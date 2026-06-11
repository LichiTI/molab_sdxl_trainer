import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GSPLayerStats:
    """GSP 层级统计"""
    residual_norm: float = 0.0      # 残差范数
    projected_norm: float = 0.0     # 投影后范数
    residual_ratio: float = 0.0     # 残差比例
    effective_k: int = 0            # 有效 k 值
    last_update_step: int = 0       # 上次 SVD 更新步数
    # V2.8: Manifold Telemetry
    stable_rank: float = 0.0        # (sum(S))^2 / sum(S^2)
    drift_velocity: float = 0.0     # 1 - cos(V_t, V_{t-1})
    spectrum_gini: float = 0.0      # S 的基尼系数 (不平衡度)

import logging
logger = logging.getLogger(__name__)

from .svd_utils import compute_effective_rank_cpu


class GradientSubspaceProjection:
    """
    GSP: Gradient Subspace Projection (V2.5 Enhanced Version)
    
    将梯度投影到由 SVD 定义的"有效子空间" (Active Subspace) 上。
    这相当于隐式地执行自然梯度下降的近似，忽略无效方向的噪声干扰。
    
    V2.5 新增功能：
    1. 残差监控 (Residual Monitoring)：检测子空间漂移
    2. 自适应 k_ratio：根据残差动态调整投影强度
    3. 懒惰 SVD 更新：只在子空间漂移时重新计算
    
    Math: g_proj = V_k @ V_k^T @ g
    """
    
    def __init__(
        self,
        k_ratio: float = 0.5,           # 初始保留比例
        update_interval: int = 100,     # 常规 SVD 更新间隔
        enable_residual_monitor: bool = True,
        residual_threshold: float = 0.3,  # 残差超过此阈值触发自适应
        adaptive_k: bool = True,          # 启用自适应 k_ratio
        min_k_ratio: float = 0.2,
        max_k_ratio: float = 0.8,
        lazy_update: bool = True,         # 懒惰更新模式
        lazy_threshold: float = 0.5,      # 残差超过此阈值触发 SVD 更新
        # V2.6: Sparse Manifold Regularization
        enable_sparse_sampling: bool = False,  # 启用稀疏采样
        sparsity_ratio: float = 0.5,           # 采样率 (0.5 = 每步只投影 50% 的层)
        # V3.0 Master Polish
        warmup_steps: int = 500,               # 预热步数
        svd_algo: str = "rsvd",                # 'rsvd', 'full', 'brands'
        precision_guard: bool = True,          # 强制 FP32 计算
        # V3.2: Subspace curvature preconditioning
        precondition_mode: str = "none",       # none | svd | grad_ema | hybrid
        svd_precond_beta: float = 0.5,
        precond_min_scale: float = 0.25,
        precond_max_scale: float = 4.0,
        coord_curv_beta: float = 0.95,
        precond_clip: float = 3.0,
        precond_eps: float = 1e-6,
        # V3.3: Adaptive sparse projection/cache. Enabled by default for
        # MN-LoRA because SDXL can expose ~2k LoRA tensors; full V/S cache is
        # usually more expensive than the benefit. The default keeps the
        # hottest ~40% layers by gradient/activity score.
        # 40-step SDXL smoke note (2026-05-25):
        # - adaptive sparse recovered ~12% wall time with ~46% projection skips.
        # - If short-run fitting looks good but speed gain is modest, first try
        #   reducing adaptive_sparse_warmup_steps from 10 to 4-6 so sparse tiers
        #   start earlier in small/quick runs.
        # - If the skip rate is high but speed barely improves, the remaining
        #   cost is likely SVD/update overhead. The next tuning direction is to
        #   lower SVD update frequency for cold layers, not just skip projection.
        adaptive_sparse_enabled: bool = True,
        adaptive_sparse_warmup_steps: int = 10,
        adaptive_sparse_refresh_interval: int = 10,
        adaptive_sparse_hot_ratio: float = 0.40,
        adaptive_sparse_warm_ratio: float = 0.0,
        adaptive_sparse_warm_interval: int = 4,
        adaptive_sparse_cold_interval: int = 16,
        adaptive_sparse_min_hot_layers: int = 16,
        adaptive_sparse_grad_beta: float = 0.95,
        adaptive_sparse_zero_cold_after: int = 3,
        adaptive_sparse_residual_weight: float = 0.45,
        adaptive_sparse_drift_weight: float = 0.30,
        adaptive_sparse_grad_weight: float = 0.25,
    ):
        self.base_k_ratio = k_ratio
        self.k_ratio = k_ratio
        self.layer_k_ratio: Dict[str, float] = {}
        self.update_interval = update_interval
        
        # V2.5 新增
        self.enable_residual_monitor = enable_residual_monitor
        self.residual_threshold = residual_threshold
        self.adaptive_k = adaptive_k
        self.min_k_ratio = min_k_ratio
        self.max_k_ratio = max_k_ratio
        self.lazy_update = lazy_update
        self.lazy_threshold = lazy_threshold
        
        # V2.6: Sparse Manifold
        self.enable_sparse_sampling = enable_sparse_sampling
        self.sparsity_ratio = sparsity_ratio
        self._sparse_counter: int = 0  # 用于轮询采样

        # V3.0: Master Polish
        self.warmup_steps = warmup_steps
        self.svd_algo = svd_algo
        self.precision_guard = precision_guard

        # V3.2: Lightweight natural-gradient-like scaling in the GSP basis.
        self.precondition_mode = str(precondition_mode or "none").lower()
        if self.precondition_mode not in {"none", "svd", "grad_ema", "hybrid"}:
            logger.warning("[GSP] Unknown precondition_mode=%s; falling back to none", precondition_mode)
            self.precondition_mode = "none"
        self.svd_precond_beta = float(svd_precond_beta)
        self.precond_min_scale = float(precond_min_scale)
        self.precond_max_scale = float(precond_max_scale)
        self.coord_curv_beta = float(coord_curv_beta)
        self.precond_clip = float(precond_clip)
        self.precond_eps = float(precond_eps)

        # V3.3: Focus projection/preconditioning budget on active, unstable layers.
        self.adaptive_sparse_enabled = bool(adaptive_sparse_enabled)
        self.adaptive_sparse_warmup_steps = max(0, int(adaptive_sparse_warmup_steps))
        self.adaptive_sparse_refresh_interval = max(1, int(adaptive_sparse_refresh_interval))
        self.adaptive_sparse_hot_ratio = max(0.0, min(1.0, float(adaptive_sparse_hot_ratio)))
        self.adaptive_sparse_warm_ratio = max(0.0, min(1.0, float(adaptive_sparse_warm_ratio)))
        self.adaptive_sparse_warm_interval = max(1, int(adaptive_sparse_warm_interval))
        self.adaptive_sparse_cold_interval = max(0, int(adaptive_sparse_cold_interval))
        self.adaptive_sparse_min_hot_layers = max(0, int(adaptive_sparse_min_hot_layers))
        self.adaptive_sparse_grad_beta = max(0.0, min(0.9999, float(adaptive_sparse_grad_beta)))
        self.adaptive_sparse_zero_cold_after = max(1, int(adaptive_sparse_zero_cold_after))
        self.adaptive_sparse_residual_weight = float(adaptive_sparse_residual_weight)
        self.adaptive_sparse_drift_weight = float(adaptive_sparse_drift_weight)
        self.adaptive_sparse_grad_weight = float(adaptive_sparse_grad_weight)
        
        # 缓存
        self.V_cache: Dict[str, torch.Tensor] = {}
        self.S_cache: Dict[str, torch.Tensor] = {}
        self.coord_curv_cache: Dict[str, torch.Tensor] = {}
        self.V_prev_cache: Dict[str, torch.Tensor] = {} # V2.8: For Drift Conv
        self.layer_stats: Dict[str, GSPLayerStats] = {}
        self._global_step = 0
        self._precond_calls: int = 0
        self._precond_ratio_count: int = 0
        self._precond_zero_norm_calls: int = 0
        self._precond_clip_count: int = 0
        self._precond_ratio_sum: float = 0.0
        self._precond_ratio_max: float = 0.0
        self._precond_ratio_min: float = float("inf")
        self._precond_last_ratio: float = 1.0
        self.layer_grad_norm_ema: Dict[str, float] = {}
        self.layer_zero_grad_streak: Dict[str, int] = {}
        self.layer_sparse_tier: Dict[str, str] = {}
        self._adaptive_sparse_last_refresh: int = -1
        self._adaptive_sparse_project_count: int = 0
        self._adaptive_sparse_skip_count: int = 0
        self._adaptive_sparse_hot_calls: int = 0
        self._adaptive_sparse_warm_calls: int = 0
        self._adaptive_sparse_cold_calls: int = 0
        self._adaptive_sparse_warm_skips: int = 0
        self._adaptive_sparse_cold_skips: int = 0
        self._adaptive_sparse_svd_update_count: int = 0
        self._adaptive_sparse_svd_skip_count: int = 0
        self._adaptive_sparse_cache_evictions: int = 0
        
    def update_subspace(self, layer_name: str, weight: torch.Tensor, force: bool = False):
        """
        SVD 更新 (should be done on CPU or asynchronously to avoid blocking)
        
        Args:
            layer_name: 层名称
            weight: 权重张量
            force: 强制更新（忽略懒惰更新逻辑）
        """
        if weight.dim() == 4:
            w = weight.view(weight.size(0), -1)
        elif weight.dim() == 2:
            w = weight
        else:
            return
        
        # 懒惰更新检查
        if self.lazy_update and not force:
            stats = self.layer_stats.get(layer_name)
            if stats and stats.residual_ratio < self.lazy_threshold:
                # 残差足够小，跳过更新
                return
            
        try:
            # Calculate target_ratio (adaptive)
            current_ratio = self.layer_k_ratio.get(layer_name, self.base_k_ratio)
            if self.adaptive_k and layer_name in self.layer_stats:
                stats = self.layer_stats[layer_name]
                if stats.residual_ratio > self.residual_threshold:
                    # 残差过大，增加 k_ratio
                    current_ratio = min(current_ratio * 1.2, self.max_k_ratio)
                    self.layer_k_ratio[layer_name] = current_ratio
                elif stats.residual_ratio < self.residual_threshold * 0.5:
                    # 残差很小，可以减少 k_ratio
                    current_ratio = max(current_ratio * 0.9, self.min_k_ratio)
                    self.layer_k_ratio[layer_name] = current_ratio
            
            # Use optimized SVD based on choice
            S_k = None
            if self.svd_algo == "rsvd":
                # torch.svd_lowrank is robust and fast
                q = int(min(w.shape) * current_ratio)
                q = max(min(q, 128), 1) # Cap for performance
                U, S, V = torch.svd_lowrank(w.to(torch.float32), q=q, niter=2)
                V_k = V[:, :q]
                S_k = S[:q]
                k_eff = q
            elif self.svd_algo == "brands":
                # Fallback to randomized for now if brands not fully implemented natively
                V_k, S_k, k_eff = compute_effective_rank_cpu(weight, ratio=current_ratio, use_randomized=True)
            else:
                # Full SVD
                V_k, S_k, k_eff = compute_effective_rank_cpu(weight, ratio=current_ratio, use_randomized=False)
            
            # Move to device and cache
            V_k = V_k.to(weight.device)
            S_k = S_k.to(weight.device) if S_k is not None else None
            
            # --- V2.8 Manifold Telemetry ---
            stable_rank = 0.0
            spectrum_gini = 0.0
            drift_velocity = 0.0
            
            # 1. Stable Rank & Gini
            if S_k is not None and len(S_k) > 0:
                s_sum = S_k.sum()
                s2_sum = (S_k ** 2).sum()
                if s2_sum > 0:
                    stable_rank = float((s_sum ** 2) / s2_sum)
                
                # Gini
                n = len(S_k)
                if n > 1:
                    # Gini formula for sorted vector:
                    # G = (2 * sum( (i+1)*x_i ) ) / (n * sum(x)) - (n+1)/n 
                    # Assumes non-increasing sort? SVD returns descending.
                    # Standard Gini: 1 - 2/(n-1) * sum( (n-i)*y_i / total_y ) for ascending
                    # Direct approx: 1 - stable_rank / rank? No.
                    # Simple concentration metric: Top-10% energy ratio
                    pass

            # 2. Drift Velocity (Subspace Angle)
            if layer_name in self.V_cache:
                # V_prev exists
                V_prev = self.V_cache[layer_name]
                # Compare V_k (current) with V_prev
                # Both are [dim, rank], rank might differ.
                # Project V_curr onto V_prev: P = V_prev @ V_prev.T
                # Similarity = || V_prev.T @ V_curr ||_F^2 / min(k1, k2)
                
                try:
                    # Align shapes if needed (min rank)
                    # For simple drift, calculate Subspace Distance
                    # D = sqrt(max(0, k - ||V1.T @ V2||_F^2))
                    
                    # Ensure same device/dtype
                    if V_prev.device != V_k.device:
                        V_prev = V_prev.to(V_k.device)
                        
                    # Compute correlation matrix
                    # [k_prev, dim] @ [dim, k_curr] -> [k_prev, k_curr]
                    corr = V_prev.T @ V_k 
                    
                    # Frobenius norm squared
                    overlap_energy = (corr ** 2).sum().item()
                    
                    # Normalized overlap (0 to 1)
                    # Denom: min(rank_prev, rank_curr) or just rank_curr?
                    # Since V are orthonormal, max energy is min(k1, k2)
                    max_energy = min(V_prev.shape[1], V_k.shape[1])
                    
                    if max_energy > 0:
                        similarity = overlap_energy / max_energy
                        drift_velocity = 1.0 - max(0.0, min(1.0, similarity))
                        
                except Exception as e:
                    pass # Ignore size mismatch errors during drift calc
            
            # Cache Rotation
            # We use V_cache as current, so effectively V_cache IS V_prev for the NEXT step.
            # No need for separate V_prev_cache if we update V_cache at the end.
            # But wait, self.V_cache is used for projection in the same step?
            # Yes, update_subspace is called periodically.
            # So V_cache holds the subspace used "until now".
            # We are computing "New Subspace".
            # Drift = distance(New, Old). Good.
            
            self.V_cache[layer_name] = V_k
            if S_k is not None:
                self.S_cache[layer_name] = S_k.detach()
            else:
                self.S_cache.pop(layer_name, None)
            
            # 更新统计
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = GSPLayerStats()
            
            stats_obj = self.layer_stats[layer_name]
            stats_obj.effective_k = k_eff
            stats_obj.last_update_step = self._global_step
            stats_obj.stable_rank = stable_rank
            stats_obj.drift_velocity = drift_velocity
            
            # Emit Telemetry for Backend (PID: stdout capture)
            try:
                import json
                telemetry = {
                    "layer": layer_name,
                    "k": k_eff,
                    "stable_rank": stable_rank,
                    "drift": drift_velocity,
                    "step": self._global_step
                }
                # Use standard logger but with specific prefix for regex capturing
                logger.info(f"MNLORA_TELEMETRY: {json.dumps(telemetry)}")
            except Exception: pass

        except Exception as e:
            logger.debug(f"[GSP] SVD failed for {layer_name}: {e}")

    def _apply_svd_preconditioner(self, layer_name: str, coords: torch.Tensor) -> torch.Tensor:
        S = self.S_cache.get(layer_name)
        if S is None or S.numel() == 0 or self.svd_precond_beta <= 0:
            return coords
        k = min(coords.shape[-1], int(S.numel()))
        if k <= 0:
            return coords
        S = S[:k].to(device=coords.device, dtype=coords.dtype).clamp_min(self.precond_eps)
        scale = (S / (S.mean() + self.precond_eps)).pow(self.svd_precond_beta)
        scale = scale.clamp(self.precond_min_scale, self.precond_max_scale)
        result = coords.clone()
        result[..., :k] = result[..., :k] / scale.view(1, -1)
        return result

    def _apply_grad_ema_preconditioner(self, layer_name: str, coords: torch.Tensor) -> torch.Tensor:
        curv_now = coords.detach().to(torch.float32).pow(2).mean(dim=0)
        if layer_name not in self.coord_curv_cache or self.coord_curv_cache[layer_name].numel() != curv_now.numel():
            curv = curv_now
        else:
            prev = self.coord_curv_cache[layer_name].to(device=curv_now.device, dtype=curv_now.dtype)
            beta = max(0.0, min(0.9999, self.coord_curv_beta))
            curv = beta * prev + (1.0 - beta) * curv_now
        self.coord_curv_cache[layer_name] = curv.detach().cpu()

        curv = curv.to(device=coords.device, dtype=coords.dtype).clamp_min(self.precond_eps)
        normalized = curv / (curv.mean() + self.precond_eps)
        scale = normalized.sqrt().clamp(self.precond_min_scale, self.precond_max_scale)
        return coords / scale.view(1, -1)

    def _clip_preconditioned_coords(self, original: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        self._precond_calls += 1
        original_norm_raw = original.norm()
        original_norm = original_norm_raw.clamp_min(self.precond_eps)
        updated_norm = updated.norm().clamp_min(self.precond_eps)
        ratio_tensor = updated_norm / original_norm
        try:
            has_signal = bool((original_norm_raw > self.precond_eps).detach().cpu())
            if has_signal:
                ratio_value = float(ratio_tensor.detach().cpu())
                self._precond_last_ratio = ratio_value
                self._precond_ratio_count += 1
                self._precond_ratio_sum += ratio_value
                self._precond_ratio_max = max(self._precond_ratio_max, ratio_value)
                self._precond_ratio_min = min(self._precond_ratio_min, ratio_value)
            else:
                self._precond_zero_norm_calls += 1
                ratio_value = 1.0
        except Exception:
            ratio_value = 1.0
        if self.precond_clip <= 0:
            return updated
        clip_factor = (original_norm * self.precond_clip / updated_norm).clamp_max(1.0)
        try:
            if bool((clip_factor < 1.0).detach().cpu()):
                self._precond_clip_count += 1
        except Exception:
            if ratio_value > self.precond_clip:
                self._precond_clip_count += 1
        return updated * clip_factor

    def _precondition_coords(self, layer_name: str, coords: torch.Tensor) -> torch.Tensor:
        if self.precondition_mode == "none":
            return coords
        original = coords
        updated = coords
        if self.precondition_mode in {"svd", "hybrid"}:
            updated = self._apply_svd_preconditioner(layer_name, updated)
        if self.precondition_mode in {"grad_ema", "hybrid"}:
            updated = self._apply_grad_ema_preconditioner(layer_name, updated)
        return self._clip_preconditioned_coords(original, updated)

    def _update_layer_signal(self, layer_name: str, grad_2d: torch.Tensor) -> None:
        try:
            grad_norm = float(grad_2d.detach().float().norm().cpu())
        except Exception:
            return
        prev = self.layer_grad_norm_ema.get(layer_name)
        if prev is None:
            ema = grad_norm
        else:
            beta = self.adaptive_sparse_grad_beta
            ema = beta * prev + (1.0 - beta) * grad_norm
        self.layer_grad_norm_ema[layer_name] = float(ema)
        if grad_norm <= self.precond_eps:
            self.layer_zero_grad_streak[layer_name] = self.layer_zero_grad_streak.get(layer_name, 0) + 1
        else:
            self.layer_zero_grad_streak[layer_name] = 0

    def observe_gradient(self, layer_name: str, grad: torch.Tensor) -> None:
        """Record gradient activity before a layer has an initialized V cache."""
        if grad.dim() == 4:
            g = grad.view(grad.size(0), -1)
        elif grad.dim() == 2:
            g = grad
        else:
            return
        self._update_layer_signal(layer_name, g)

    def _refresh_adaptive_sparse_tiers(self, force: bool = False) -> None:
        if not self.adaptive_sparse_enabled:
            return
        if not force and self._global_step <= self.adaptive_sparse_warmup_steps:
            return
        if not force and self._adaptive_sparse_last_refresh >= 0:
            elapsed = self._global_step - self._adaptive_sparse_last_refresh
            if elapsed < self.adaptive_sparse_refresh_interval:
                return

        layers = sorted(set(self.layer_grad_norm_ema.keys()) | set(self.V_cache.keys()))
        if not layers:
            return

        max_grad = max((self.layer_grad_norm_ema.get(name, 0.0) for name in layers), default=0.0)
        scored = []
        for name in layers:
            stats = self.layer_stats.get(name)
            residual = float(stats.residual_ratio) if stats is not None else 0.0
            drift = float(stats.drift_velocity) if stats is not None else 0.0
            grad_score = self.layer_grad_norm_ema.get(name, 0.0) / (max_grad + self.precond_eps)
            score = (
                self.adaptive_sparse_residual_weight * residual
                + self.adaptive_sparse_drift_weight * drift
                + self.adaptive_sparse_grad_weight * grad_score
            )
            zero_streak = self.layer_zero_grad_streak.get(name, 0)
            scored.append((score, zero_streak, name))

        scored.sort(key=lambda item: item[0], reverse=True)
        n_layers = len(scored)
        hot_count = max(self.adaptive_sparse_min_hot_layers, int(n_layers * self.adaptive_sparse_hot_ratio))
        hot_count = min(hot_count, n_layers)
        warm_count = int(n_layers * self.adaptive_sparse_warm_ratio)
        warm_count = min(max(warm_count, 0), max(n_layers - hot_count, 0))

        new_tiers: Dict[str, str] = {}
        for idx, (_score, zero_streak, name) in enumerate(scored):
            if zero_streak >= self.adaptive_sparse_zero_cold_after:
                tier = "cold"
            elif idx < hot_count:
                tier = "hot"
            elif idx < hot_count + warm_count:
                tier = "warm"
            else:
                tier = "cold"
            new_tiers[name] = tier

        self.layer_sparse_tier = new_tiers
        self._adaptive_sparse_last_refresh = self._global_step
        self._prune_adaptive_sparse_cache()

    def _prune_adaptive_sparse_cache(self) -> None:
        if not self.adaptive_sparse_enabled or not self.layer_sparse_tier:
            return
        keep = {name for name, tier in self.layer_sparse_tier.items() if tier in {"hot", "warm"}}
        for name in list(self.V_cache.keys()):
            if name in keep:
                continue
            self.V_cache.pop(name, None)
            self.S_cache.pop(name, None)
            self.coord_curv_cache.pop(name, None)
            self._adaptive_sparse_cache_evictions += 1

    def should_update_subspace_cache(self, layer_name: str, has_cache: bool = False) -> bool:
        """Return whether this layer should pay the SVD/V-cache update cost."""
        if not self.adaptive_sparse_enabled:
            return True
        if self._global_step <= self.adaptive_sparse_warmup_steps:
            self._adaptive_sparse_svd_skip_count += 1
            return False
        self._refresh_adaptive_sparse_tiers(force=layer_name not in self.layer_sparse_tier)
        tier = self.layer_sparse_tier.get(layer_name, "cold")
        if tier == "hot":
            self._adaptive_sparse_svd_update_count += 1
            return True
        if tier == "warm":
            if not has_cache or self._global_step % self.adaptive_sparse_warm_interval == 0:
                self._adaptive_sparse_svd_update_count += 1
                return True
            self._adaptive_sparse_svd_skip_count += 1
            return False
        if has_cache and self.adaptive_sparse_cold_interval > 0 and self._global_step % self.adaptive_sparse_cold_interval == 0:
            self._adaptive_sparse_svd_update_count += 1
            return True
        self._adaptive_sparse_svd_skip_count += 1
        return False

    def _should_project_adaptive_sparse(self, layer_name: str) -> bool:
        if not self.adaptive_sparse_enabled:
            return True
        if self._global_step <= self.adaptive_sparse_warmup_steps:
            self._adaptive_sparse_project_count += 1
            return True

        self._refresh_adaptive_sparse_tiers()
        tier = self.layer_sparse_tier.get(layer_name, "hot")
        if tier == "hot":
            self._adaptive_sparse_hot_calls += 1
            self._adaptive_sparse_project_count += 1
            return True
        if tier == "warm":
            self._adaptive_sparse_warm_calls += 1
            if self._global_step % self.adaptive_sparse_warm_interval == 0:
                self._adaptive_sparse_project_count += 1
                return True
            self._adaptive_sparse_warm_skips += 1
            self._adaptive_sparse_skip_count += 1
            return False

        self._adaptive_sparse_cold_calls += 1
        # Future tuning hook:
        # Cold layers are already projected less often here, but they can still
        # pay SVD refresh costs from the optimizer path. If telemetry shows many
        # cold skips with limited wall-time gain, add a cold-layer SVD throttle
        # near the update_subspace caller so these layers also refresh slower.
        if self.adaptive_sparse_cold_interval > 0 and self._global_step % self.adaptive_sparse_cold_interval == 0:
            self._adaptive_sparse_project_count += 1
            return True
        self._adaptive_sparse_cold_skips += 1
        self._adaptive_sparse_skip_count += 1
        return False

    def project_gradient(
        self,
        layer_name: str,
        grad: torch.Tensor,
        return_residual: bool = False
    ) -> torch.Tensor:
        """
        投影梯度到子空间
        
        Args:
            layer_name: 层名称
            grad: 原始梯度
            return_residual: 是否返回残差信息
            
        Returns:
            投影后的梯度（和可选的残差）
        """
        # V2.6: Sparse Sampling - 随机跳过部分层的投影
        if self.enable_sparse_sampling:
            self._sparse_counter += 1
            # 使用 hash 确保同一层在同一步内的行为一致
            layer_hash = hash(layer_name) % 1000
            threshold = int(self.sparsity_ratio * 1000)
            if (self._sparse_counter + layer_hash) % 1000 >= threshold:
                # 本层本次跳过投影，直接返回原始梯度
                return grad
        
        orig_shape = grad.shape
        if grad.dim() == 4:
            g = grad.view(grad.size(0), -1)
        elif grad.dim() == 2:
            g = grad
        else:
            return grad

        self._update_layer_signal(layer_name, g)

        if layer_name not in self.V_cache:
            return grad
            
        V_k = self.V_cache[layer_name]  # [in_features, k]

        if not self._should_project_adaptive_sparse(layer_name):
            return grad
            
        # Projection: P = V V^T
        # g_proj = g @ V @ V^T
        
        # V3.0 Master Polish: Numerical Stability Guard
        if self.precision_guard:
            orig_dtype = g.dtype
            # Execute in FP32 to prevent drift
            g_float = g.to(torch.float32)
            V_float = V_k.to(torch.float32)
            coords = g_float @ V_float
            coords = self._precondition_coords(layer_name, coords)
            g_proj = coords @ V_float.T
            g_proj = g_proj.to(orig_dtype)
        else:
            coords = g @ V_k
            coords = self._precondition_coords(layer_name, coords)
            g_proj = coords @ V_k.T
        
        # 残差监控
        if self.enable_residual_monitor:
            residual = g - g_proj
            residual_norm = residual.norm().item()
            proj_norm = g_proj.norm().item()
            original_norm = g.norm().item()
            
            # 计算残差比例
            ratio = residual_norm / (original_norm + 1e-8)
            
            # 更新统计
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = GSPLayerStats()
            
            stats = self.layer_stats[layer_name]
            stats.residual_norm = residual_norm
            stats.projected_norm = proj_norm
            stats.residual_ratio = ratio
            
            # 如果残差过大且启用了自适应，混合一部分原始梯度
            # V2.7: 使用 Phase-Aware 阈值 (需要从外部传入 phase，或者在 step 中记录 phase)
            # 由于 project_gradient 不接受 phase 参数，我们需要在 step 中保存 current_phase
            thresh = self.get_effective_threshold(getattr(self, 'current_phase', 'steady'))
            
            if self.adaptive_k and ratio > thresh:
                # 混合系数：残差越大，保留越多原始梯度
                alpha = min(0.5, (ratio - thresh) / thresh)
                g_proj = (1 - alpha) * g_proj + alpha * g
        
        return g_proj.view(orig_shape)
    
    def step(self, global_step: int, phase: str = "steady"):
        """
        更新全局步数与训练阶段
        
        Args:
            global_step: 全局训练步数
            phase: 训练阶段 ('warmup', 'steady', 'decay')
        """
        self._global_step = global_step
        self.current_phase = phase
        
        # Phase-Aware Logic (V2.7)
        # 根据训练阶段动态调整 GSP 行为
        if phase == "warmup":
            # Warmup: 允许更大的残差，加速子空间探索
            # 临时放宽 20%
            pass # 保持原样或显示 log
            
        elif phase == "decay":
            # Decay: 收敛阶段，收紧子空间以锁定流形
            # 下调 residual_threshold，强制更激进的投影
            # 注意：这里我们动态修改实例变量，但在 reset 时不需要恢复，
            # 因为 GSP 通常是 per-training session 的。
            # 为了安全，我们仅在检测到切换时做一次性调整，或使用临时变量。
            # 简单起见，这里假设外部调用是连续的。
            pass

    def get_effective_threshold(self, phase: str = "steady") -> float:
        """获取当前阶段的有效残差阈值"""
        if phase == "decay":
            return self.residual_threshold * 0.8  # 收紧 20%
        elif phase == "warmup":
            return self.residual_threshold * 1.2  # 放宽 20%
        return self.residual_threshold

    
    def get_stats(self) -> Dict[str, GSPLayerStats]:
        """获取所有层的统计信息"""
        return self.layer_stats

    def get_telemetry_snapshot(self) -> Dict[str, object]:
        """Return a JSON-safe summary of GSP/MN-LoRA preconditioner state."""
        stats = list(self.layer_stats.values())
        residuals = [float(s.residual_ratio) for s in stats]
        projected_norms = [float(s.projected_norm) for s in stats]
        effective_ks = [int(s.effective_k) for s in stats]
        stable_ranks = [float(s.stable_rank) for s in stats if float(s.stable_rank) > 0.0]
        drifts = [float(s.drift_velocity) for s in stats]

        def _avg(values: list[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        precond_avg = self._precond_ratio_sum / self._precond_ratio_count if self._precond_ratio_count else 1.0
        precond_min = self._precond_ratio_min if self._precond_ratio_count and self._precond_ratio_min != float("inf") else 1.0
        coord_values = []
        for tensor in self.coord_curv_cache.values():
            try:
                coord_values.append(tensor.detach().float().mean())
            except Exception:
                continue
        coord_mean = 0.0
        coord_max = 0.0
        coord_min = 0.0
        if coord_values:
            try:
                stacked = torch.stack(coord_values)
                coord_mean = float(stacked.mean())
                coord_max = float(stacked.max())
                coord_min = float(stacked.min())
            except Exception:
                coord_mean = 0.0
                coord_max = 0.0
                coord_min = 0.0

        sparse_tier_counts = {"hot": 0, "warm": 0, "cold": 0}
        for tier in self.layer_sparse_tier.values():
            if tier in sparse_tier_counts:
                sparse_tier_counts[tier] += 1

        return {
            "mode": self.precondition_mode,
            "global_step": int(self._global_step),
            "v_cache_layers": len(self.V_cache),
            "s_cache_layers": len(self.S_cache),
            "coord_curv_layers": len(self.coord_curv_cache),
            "layer_stats_count": len(stats),
            "avg_residual_ratio": _avg(residuals),
            "max_residual_ratio": max(residuals) if residuals else 0.0,
            "avg_projected_norm": _avg(projected_norms),
            "avg_effective_k": _avg([float(k) for k in effective_ks]),
            "max_effective_k": max(effective_ks) if effective_ks else 0,
            "avg_stable_rank": _avg(stable_ranks),
            "avg_drift_velocity": _avg(drifts),
            "precondition_calls": int(self._precond_calls),
            "precondition_ratio_count": int(self._precond_ratio_count),
            "precondition_zero_norm_calls": int(self._precond_zero_norm_calls),
            "precondition_clip_count": int(self._precond_clip_count),
            "precondition_clip_rate": float(self._precond_clip_count / self._precond_calls) if self._precond_calls else 0.0,
            "precondition_norm_ratio_avg": float(precond_avg),
            "precondition_norm_ratio_min": float(precond_min),
            "precondition_norm_ratio_max": float(self._precond_ratio_max if self._precond_ratio_count else 1.0),
            "precondition_norm_ratio_last": float(self._precond_last_ratio),
            "coord_curv_mean": coord_mean,
            "coord_curv_min": coord_min,
            "coord_curv_max": coord_max,
            "adaptive_sparse_enabled": bool(self.adaptive_sparse_enabled),
            "adaptive_sparse_warmup_steps": int(self.adaptive_sparse_warmup_steps),
            "adaptive_sparse_last_refresh": int(self._adaptive_sparse_last_refresh),
            "adaptive_sparse_project_count": int(self._adaptive_sparse_project_count),
            "adaptive_sparse_skip_count": int(self._adaptive_sparse_skip_count),
            "adaptive_sparse_skip_rate": (
                float(self._adaptive_sparse_skip_count / (self._adaptive_sparse_project_count + self._adaptive_sparse_skip_count))
                if (self._adaptive_sparse_project_count + self._adaptive_sparse_skip_count)
                else 0.0
            ),
            "adaptive_sparse_hot_layers": int(sparse_tier_counts["hot"]),
            "adaptive_sparse_warm_layers": int(sparse_tier_counts["warm"]),
            "adaptive_sparse_cold_layers": int(sparse_tier_counts["cold"]),
            "adaptive_sparse_hot_calls": int(self._adaptive_sparse_hot_calls),
            "adaptive_sparse_warm_calls": int(self._adaptive_sparse_warm_calls),
            "adaptive_sparse_cold_calls": int(self._adaptive_sparse_cold_calls),
            "adaptive_sparse_warm_skips": int(self._adaptive_sparse_warm_skips),
            "adaptive_sparse_cold_skips": int(self._adaptive_sparse_cold_skips),
            "adaptive_sparse_svd_update_count": int(self._adaptive_sparse_svd_update_count),
            "adaptive_sparse_svd_skip_count": int(self._adaptive_sparse_svd_skip_count),
            "adaptive_sparse_cache_evictions": int(self._adaptive_sparse_cache_evictions),
            "adaptive_sparse_cached_layers": int(len(self.V_cache)),
        }
    
    def get_layer_residual(self, layer_name: str) -> float:
        """获取指定层的残差比例"""
        if layer_name in self.layer_stats:
            return self.layer_stats[layer_name].residual_ratio
        return 0.0
    
    def get_average_residual(self) -> float:
        """获取所有层的平均残差比例"""
        if not self.layer_stats:
            return 0.0
        ratios = [s.residual_ratio for s in self.layer_stats.values()]
        return sum(ratios) / len(ratios)
    
    def should_trigger_svd_update(self, layer_name: str) -> bool:
        """检查是否应该触发 SVD 更新（用于懒惰更新）"""
        if not self.lazy_update:
            return True
        
        if layer_name not in self.layer_stats:
            return True
            
        return self.layer_stats[layer_name].residual_ratio > self.lazy_threshold
    
    def reset_k_ratio(self):
        """重置 k_ratio 到初始值"""
        self.k_ratio = self.base_k_ratio
        self.layer_k_ratio.clear()


class GSPMonitor:
    """
    GSP 监控器
    
    用于可视化和分析 GSP 的行为
    """
    
    def __init__(self, gsp: GradientSubspaceProjection):
        self.gsp = gsp
        self.history: Dict[str, list] = {}
    
    def record(self, step: int):
        """记录当前状态"""
        for name, stats in self.gsp.get_stats().items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append({
                'step': step,
                'residual_ratio': stats.residual_ratio,
                'effective_k': stats.effective_k,
            })
    
    def get_layer_history(self, layer_name: str) -> list:
        """获取指定层的历史"""
        return self.history.get(layer_name, [])
    
    def get_drift_warning_layers(self, threshold: float = 0.5) -> list:
        """获取子空间漂移严重的层"""
        warnings = []
        for name, stats in self.gsp.get_stats().items():
            if stats.residual_ratio > threshold:
                warnings.append({
                    'layer': name,
                    'residual_ratio': stats.residual_ratio,
                    'recommendation': 'Consider forcing SVD update or increasing k_ratio'
                })
        return warnings


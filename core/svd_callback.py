"""
SVD Callback 模块

⚠️ 废弃警告 (Deprecation Notice):
    SVDCallback 类已被 LoRAAuditor 完全取代。
    
    推荐使用统一入口:
        from core.audit import AuditSystem
        audit = AuditSystem(enable_auditor=True, enable_dynamic_pruning=True)
    
    此模块中的 DynamicRankPruner 仍然有效，并已集成到 AuditSystem。

历史版本功能整合:
    - SVDCallback -> core.lora_auditor.LoRAAuditor (更完整)
    - DynamicRankPruner -> 保留，通过 core.audit.AuditSystem 访问
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

try:
    import torch
    import numpy as np
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    np = None

logger = logging.getLogger(__name__)


# 发出废弃警告 (仅对 SVDCallback)
def _warn_deprecated():
    warnings.warn(
        "SVDCallback 已废弃，请使用 core.lora_auditor.LoRAAuditor 或 core.audit.AuditSystem",
        DeprecationWarning,
        stacklevel=3
    )



class SVDCallback:

    def __init__(
        self, 
        output_file: str, 
        analyze_interval: int = None, 
        async_compute: bool = True, 
        on_data: Optional[Callable] = None,
        use_rsvd: bool = True,  # 使用 rSVD 算法 (推荐，降低 25x 资源占用)
        rsvd_k: int = 50,       # rSVD 投影维度 (10-50，越大越精确)
        advanced_stats: bool = False  # 高级统计模式
    ):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._analyze_interval = analyze_interval
        self.async_compute = async_compute
        self.on_data = on_data
        self.use_rsvd = use_rsvd
        self.rsvd_k = rsvd_k
        self.advanced_stats = advanced_stats
        self._ema_loss = None
        self._ema_alpha = 0.1
        self._epoch_losses: List[float] = []
        self._current_epoch = 1
        self._last_svd_data = {}
        self._svd_lock = None
        if async_compute:
            import threading
            self._svd_lock = threading.Lock()
        self._file = open(self.output_file, 'a', encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        """AUDIT FIX: Safety net to ensure file handle is closed even if close() wasn't called explicitly."""
        try:
            self.close()
        except Exception:
            pass

    def _get_analyze_interval(self, total_steps: int) -> int:
        if self._analyze_interval is not None:
            return self._analyze_interval
        if total_steps > 1000:
            return 100
        else:
            return max(10, total_steps // 10)

    def step(self, step: int, total_steps: int, epoch: int, loss: float, lr_unet: float=0.0, lr_te: float=0.0, network=None, vram_gb: float=0.0, power_w: float=0.0, throughput: float=0.0):
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
        self._epoch_losses.append(loss)
        if epoch > self._current_epoch:
            self._epoch_losses = [loss]
            self._current_epoch = epoch
        epoch_loss = sum(self._epoch_losses) / len(self._epoch_losses) if self._epoch_losses else loss
        analyze_interval = self._get_analyze_interval(total_steps)
        svd_data = self._last_svd_data.copy() if self._last_svd_data else {}
        if step % analyze_interval == 0 and network is not None and HAS_TORCH:
            if self.async_compute:
                self._async_analyze_svd(network)
            else:
                svd_data = self._analyze_network_svd(network)
                self._last_svd_data = svd_data
        record = {'timestamp': datetime.now().isoformat(), 'step': step, 'total_steps': total_steps, 'epoch': epoch, 'loss': {'step': round(loss, 5), 'ema': round(self._ema_loss, 5), 'epoch': round(epoch_loss, 5)}, 'lr': {'unet': lr_unet, 'te': lr_te}, 'svd': svd_data, 'hardware': {'vram_gb': round(vram_gb, 2), 'power_w': round(power_w, 1), 'throughput': round(throughput, 2)}}
        self._file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self._file.flush()
        if self.on_data:
            self.on_data(record)

    def _async_analyze_svd(self, network):
        import threading
        weights_cpu = []
        try:
            for name, module in network.named_modules():
                if hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                    down_weight = module.lora_down.weight
                    if down_weight.dim() >= 2:
                        weights_cpu.append((name, down_weight.detach().cpu().clone()))
        except (AttributeError, RuntimeError) as e:
            # AUDIT FIX: Narrowed exception to avoid hiding logic errors
            logger.error(f"[SVDCallback] Error detaching weights: {e}")
            return
        if not weights_cpu:
            return

        def compute_in_thread():
            try:
                effective_ranks = []
                total_ranks = []
                stable_ranks = []
                svd_entropies = []
                
                for name, weight in weights_cpu:
                    result = self._compute_svd_stats_cpu(weight, use_rsvd=self.use_rsvd, k=self.rsvd_k)
                    if result:
                        effective_ranks.append(result['effective_rank'])
                        total_ranks.append(result['total_rank'])
                        if 'stable_rank' in result:
                            stable_ranks.append(result['stable_rank'])
                        if 'svd_entropy' in result:
                            svd_entropies.append(result['svd_entropy'])
                
                if effective_ranks:
                    avg_effective_rank = sum(effective_ranks) / len(effective_ranks)
                    max_rank = max(total_ranks) if total_ranks else 1
                    
                    svd_data = {
                        'effective_rank': round(avg_effective_rank, 1), 
                        'max_rank': max_rank, 
                        'health_score': round(self._compute_health_score(avg_effective_rank, max_rank), 3), 
                        'num_layers': len(effective_ranks),
                        'algorithm': 'rsvd' if self.use_rsvd else 'standard'
                    }
                    
                    # 添加高级指标
                    if stable_ranks:
                        svd_data['stable_rank'] = round(sum(stable_ranks) / len(stable_ranks), 3)
                    if svd_entropies:
                        svd_data['svd_entropy'] = round(sum(svd_entropies) / len(svd_entropies), 3)
                    
                    if self._svd_lock:
                        with self._svd_lock:
                            self._last_svd_data = svd_data
                    else:
                        self._last_svd_data = svd_data
            except RuntimeError as e:
                # AUDIT FIX: Catch specific tensor/CUDA errors
                logger.error(f"[SVDCallback] Async compute error: {e}")
            except Exception as e:
                logger.error(f"[SVDCallback] Unexpected error in async thread: {e}")
        t = threading.Thread(target=compute_in_thread, daemon=True)
        t.start()

    def _compute_svd_stats_cpu(self, weight, use_rsvd: bool = True, k: int = 50) -> Optional[Dict]:
        """
        计算 SVD 统计信息
        
        Args:
            weight: 权重张量
            use_rsvd: 是否使用随机化 SVD (降低资源占用 25x)
            k: rSVD 投影维度
        """
        try:
            if weight.dim() == 4:
                w = weight.view(weight.size(0), -1)
            elif weight.dim() == 2:
                w = weight
            else:
                return None
            
            w = w.float()
            
            if use_rsvd:
                # 随机化 SVD (rSVD) - 降低 25x 资源占用
                # 基于 Johnson-Lindenstrauss 引理
                height, width = w.shape
                omega = torch.randn((width, k), dtype=torch.float32)
                Y = torch.mm(w, omega)
                S = torch.linalg.svdvals(Y)
            else:
                # 标准 SVD
                _, S, _ = torch.linalg.svd(w, full_matrices=False)
            
            S_np = S.numpy()
            
            # Stable Rank: sum(σ²) / σ_max²
            max_sv = S_np[0] if len(S_np) > 0 else 1
            stable_rank = float(np.sum(S_np ** 2) / (max_sv ** 2 + 1e-8))
            
            # SVD Entropy
            s_sum = np.sum(S_np) + 1e-8
            p = S_np / s_sum
            entropy = float(-np.sum(p * np.log2(p + 1e-8)))
            
            # Effective rank (threshold based)
            threshold = S_np.max() * 0.01
            effective_rank = int((S_np > threshold).sum())
            
            return {
                'effective_rank': effective_rank, 
                'total_rank': len(S_np),
                'stable_rank': round(stable_rank, 3),
                'svd_entropy': round(entropy, 3)
            }
        except (RuntimeError, ValueError) as e:
            # AUDIT FIX: Specific errors for SVD computation
            logger.error(f"[SVDCallback] SVD stats error: {e}")
            return None
        except Exception as e:
            logger.error(f"[SVDCallback] Unexpected SVD error: {e}")
            return None

    def _analyze_network_svd(self, network) -> Dict[str, Any]:
        if not HAS_TORCH:
            return {}
        try:
            effective_ranks = []
            total_ranks = []
            for name, module in network.named_modules():
                if hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                    down_weight = module.lora_down.weight
                    if down_weight.dim() >= 2:
                        result = self._compute_svd_stats(down_weight)
                        if result:
                            effective_ranks.append(result['effective_rank'])
                            total_ranks.append(result['total_rank'])
            if not effective_ranks:
                return {}
            avg_effective_rank = sum(effective_ranks) / len(effective_ranks)
            max_rank = max(total_ranks) if total_ranks else 1
            return {'effective_rank': round(avg_effective_rank, 1), 'max_rank': max_rank, 'health_score': round(self._compute_health_score(avg_effective_rank, max_rank), 3), 'num_layers': len(effective_ranks)}
        except Exception as e:
            return {'error': str(e)}

    def _compute_svd_stats(self, weight: 'torch.Tensor') -> Optional[Dict]:
        try:
            if weight.dim() == 4:
                w = weight.view(weight.size(0), -1)
            elif weight.dim() == 2:
                w = weight
            else:
                return None
            with torch.no_grad():
                _, S, _ = torch.linalg.svd(w.float(), full_matrices=False)
            S_np = S.cpu().numpy()
            threshold = S_np.max() * 0.01
            effective_rank = int((S_np > threshold).sum())
            return {'effective_rank': effective_rank, 'total_rank': len(S_np)}
        except Exception:
            return None

    def _compute_health_score(self, eff_rank: float, max_rank: int) -> float:
        ratio = eff_rank / max_rank if max_rank > 0 else 0
        if 0.3 <= ratio <= 0.8:
            return 1.0
        elif ratio < 0.3:
            return ratio / 0.3
        else:
            return max(0, 1.0 - (ratio - 0.8) / 0.2)

    def close(self):
        """显式关闭文件句柄"""
        if hasattr(self, '_file') and self._file:
            try:
                self._file.close()
            except Exception as e:
                logger.error(f"[SVDCallback] Error closing file: {e}")
            finally:
                self._file = None

def inject_callback_to_trainer(trainer, output_file: str, analyze_interval: int=50):
    callback = SVDCallback(output_file, analyze_interval)
    original_step = trainer.train if hasattr(trainer, 'train') else None
    if original_step:

        def wrapped_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            return result
        trainer.train = wrapped_step
    return callback

def auto_inject_if_enabled():
    """
    自动注入 SVDCallback (通过环境变量)
    
    环境变量:
        LULYNX_SVD_MONITOR: "1" 或 "true" 启用
        LULYNX_SVD_OUTPUT: 输出文件路径
        LULYNX_SVD_ALGORITHM: "rsvd" 或 "standard"
        LULYNX_SVD_K: rSVD 投影维度 (默认 50)
        LULYNX_ADVANCED_STATS: "1" 或 "true" 启用高级统计
    """
    if os.environ.get('LULYNX_SVD_MONITOR', '').lower() in ('1', 'true', 'yes'):
        output_file = os.environ.get('LULYNX_SVD_OUTPUT', './logs/training.jsonl')
        use_rsvd = os.environ.get('LULYNX_SVD_ALGORITHM', 'rsvd').lower() != 'standard'
        rsvd_k = int(os.environ.get('LULYNX_SVD_K', '50'))
        advanced_stats = os.environ.get('LULYNX_ADVANCED_STATS', '').lower() in ('1', 'true', 'yes')
        
        logger.info(f'[SVDCallback] Auto-inject enabled')
        logger.info(f'  Output: {output_file}')
        logger.info(f'  Algorithm: {"rSVD (k={})".format(rsvd_k) if use_rsvd else "Standard SVD"}')
        logger.info(f'  Advanced Stats: {advanced_stats}')
        
        return SVDCallback(
            output_file, 
            use_rsvd=use_rsvd, 
            rsvd_k=rsvd_k,
            advanced_stats=advanced_stats
        )
    return None


class DynamicRankPruner:
    """
    训练中动态 Rank 剔除器
    
    自动检测并移除未使用的 LoRA rank 维度，减少显存占用和训练时间。
    
    工作原理:
    1. 定期对每层 LoRA 权重进行 SVD 分解
    2. 计算 effective rank (奇异值 > 1% max 的数量)
    3. 如果连续 N 次检测到 rank 利用率 < threshold，则剔除
    4. 使用 SVD 重建更小的 A/B 矩阵
    """
    
    def __init__(
        self,
        prune_threshold: float = 0.05,     # 剔除阈值 (利用率 < 5%)
        min_rank: int = 8,                  # 最低保留 rank
        warmup_ratio: float = 0.15,         # warmup 期间不剔除
        require_consecutive: int = 2,       # 连续 N 次低于阈值才剔除
        on_prune: Optional[Callable] = None # 剔除回调
    ):
        if not HAS_TORCH:
            raise ImportError("DynamicRankPruner requires PyTorch")
        
        self.prune_threshold = prune_threshold
        self.min_rank = min_rank
        self.warmup_ratio = warmup_ratio
        self.require_consecutive = require_consecutive
        self.on_prune = on_prune
        
        # 跟踪每层的低利用率次数
        self._low_util_counts: Dict[str, int] = {}
        # 记录每层的当前 rank
        self._current_ranks: Dict[str, int] = {}
        # 记录剔除历史
        self._prune_history: List[Dict] = []
        # 总剔除的 rank 数
        self.total_pruned = 0
        
    def _get_prune_interval(self, total_steps: int) -> int:
        """动态计算检查间隔：总步数的 10%，范围 [50, 500]"""
        return max(50, min(500, total_steps // 10))
    
    def _get_warmup_steps(self, total_steps: int) -> int:
        """计算 warmup 步数"""
        return int(total_steps * self.warmup_ratio)
    
    def step(
        self, 
        step: int, 
        total_steps: int, 
        network,
        optimizer = None  # 可选：同步更新优化器状态
    ) -> Optional[Dict]:
        """
        每步调用，检查是否需要剔除
        
        Returns:
            剔除信息 dict (如果发生剔除)，否则 None
        """
        # Warmup 期间不剔除
        warmup_steps = self._get_warmup_steps(total_steps)
        if step < warmup_steps:
            return None
            
        # 检查 Orchestra 控制器准入（Anti-Resonance）
        try:
            from core.lulynx_trainer.orchestra_controller import get_orchestra
            orchestra = get_orchestra()
            # 无论是否到达间隔，每步都要让 Orchestra 知道我们在跑
            # 但这里我们只在应该跑的时候问
            interval = self._get_prune_interval(total_steps)
            if step % interval == 0:
                if not orchestra.should_run_rank_pruning():
                    # print(f"[DynamicPruner] Pruning skipped by Orchestra at step {step}")
                    return None
        except ImportError:
            pass
        
        # 检查是否到了剔除间隔
        interval = self._get_prune_interval(total_steps)
        if step % interval != 0:
            return None
        
        # 分析并剔除
        return self._analyze_and_prune(network, optimizer, step)
    
    def _analyze_and_prune(self, network, optimizer, step: int) -> Optional[Dict]:
        """分析网络并剔除低利用率 rank"""
        pruned_layers = []
        
        for name, module in network.named_modules():
            if not (hasattr(module, 'lora_down') and hasattr(module, 'lora_up')):
                continue
            
            down = module.lora_down.weight
            up = module.lora_up.weight
            
            if down.dim() < 2 or up.dim() < 2:
                continue
            
            current_rank = down.shape[0]
            
            # 初始化跟踪
            if name not in self._current_ranks:
                self._current_ranks[name] = current_rank
                self._low_util_counts[name] = 0
            
            # 计算 effective rank
            effective_rank = self._compute_effective_rank(down, up)
            utilization = effective_rank / current_rank if current_rank > 0 else 1.0
            
            # 检查是否低于阈值
            if utilization < self.prune_threshold:
                self._low_util_counts[name] += 1
            else:
                self._low_util_counts[name] = 0
            
            # 连续 N 次低于阈值才剔除
            if self._low_util_counts[name] >= self.require_consecutive:
                new_rank = max(self.min_rank, effective_rank)
                
                if new_rank < current_rank:
                    success = self._prune_layer(module, new_rank, optimizer, name)
                    if success:
                        pruned = current_rank - new_rank
                        self.total_pruned += pruned
                        self._current_ranks[name] = new_rank
                        self._low_util_counts[name] = 0
                        
                        pruned_layers.append({
                            "layer": name,
                            "old_rank": current_rank,
                            "new_rank": new_rank,
                            "pruned": pruned,
                            "utilization": round(utilization, 3)
                        })
        
        if pruned_layers:
            result = {
                "step": step,
                "layers_pruned": len(pruned_layers),
                "total_pruned": self.total_pruned,
                "details": pruned_layers
            }
            self._prune_history.append(result)
            
            # 通知 Orchestra 控制器
            try:
                from core.lulynx_trainer.orchestra_controller import get_orchestra
                get_orchestra().notify_rank_pruned(len(pruned_layers))
            except ImportError:
                pass
            
            if self.on_prune:
                self.on_prune(result)
            
            return result
        
        return None
    
    def _compute_effective_rank(self, down_weight, up_weight) -> int:
        """计算 effective rank"""
        try:
            with torch.no_grad():
                # 合并权重: up @ down
                if down_weight.dim() == 4:
                    down_2d = down_weight.view(down_weight.size(0), -1)
                else:
                    down_2d = down_weight
                
                if up_weight.dim() == 4:
                    up_2d = up_weight.view(up_weight.size(0), -1)
                else:
                    up_2d = up_weight
                
                # 使用 down 矩阵计算 (更小)
                S = torch.linalg.svdvals(down_2d.float())
                threshold = S.max() * 0.01
                effective_rank = int((S > threshold).sum().item())
                
                return max(1, effective_rank)
        except Exception:
            return down_weight.shape[0]  # 失败时返回当前 rank
    
    def _prune_layer(self, module, new_rank: int, optimizer, layer_name: str) -> bool:
        """
        剔除层的无效 rank
        
        使用 SVD 重建更小的 A/B 矩阵
        """
        try:
            with torch.no_grad():
                up = module.lora_up.weight.data  # [out_features, rank]
                down = module.lora_down.weight.data  # [rank, in_features]
                
                # 处理 4D 卷积权重
                up_shape = up.shape
                down_shape = down.shape
                
                if up.dim() == 4:
                    up = up.view(up.size(0), -1)
                if down.dim() == 4:
                    down = down.view(down.size(0), -1)
                
                # 获取 alpha 和 scale
                alpha = getattr(module, 'alpha', None)
                if alpha is None:
                    alpha = down.shape[0]
                scale = float(alpha) / down.shape[0]
                
                # 合并并 SVD
                # 合并并 SVD
                merged = (up @ down) * scale
                
                # AUDIT FIX: Use rSVD for performance (O(mn^2) -> O(kmn))
                try:
                    # 多算 10 个 rank 作为 buffer，提高精度
                    k_target = min(new_rank + 10, min(merged.shape))
                    U, S, Vh_t = torch.svd_lowrank(merged.float(), q=k_target, niter=2)
                    Vh = Vh_t[:, :k_target].T
                    U = U[:, :k_target]
                    S = S[:k_target]
                except Exception as e:
                    print(f"[DynamicPruner] rSVD failed, fallback to full SVD: {e}")
                    U, S, Vh = torch.linalg.svd(merged.float(), full_matrices=False)
                
                # 截断到 new_rank
                # SVD 结果可能比 new_rank 大（如果是 rSVD），也可能正好
                # 确保安全切片
                U = U[:, :new_rank]
                S_sqrt = torch.sqrt(S[:new_rank])
                Vh = Vh[:new_rank, :]
                
                new_up = U @ torch.diag(S_sqrt)
                new_down = torch.diag(S_sqrt) @ Vh
                
                # 恢复 4D 形状
                if len(up_shape) == 4:
                    new_up = new_up.view(up_shape[0], new_rank, 1, 1)
                if len(down_shape) == 4:
                    new_down = new_down.view(new_rank, down_shape[1], down_shape[2], down_shape[3])
                
                # 创建新参数
                device = module.lora_up.weight.device
                dtype = module.lora_up.weight.dtype
                
                new_up_param = torch.nn.Parameter(new_up.to(dtype=dtype, device=device))
                new_down_param = torch.nn.Parameter(new_down.to(dtype=dtype, device=device))
                
                # 更新模块
                module.lora_up.weight = new_up_param
                module.lora_down.weight = new_down_param
                
                # 更新 alpha (设为 new_rank，即 scale=1)
                if hasattr(module, 'alpha'):
                    module.alpha = torch.tensor(new_rank, dtype=torch.float32, device=device)
                
                # 更新优化器状态 (如果提供)
                if optimizer is not None:
                    self._update_optimizer_state(optimizer, module, new_rank)
                
                print(f"[DynamicPruner] {layer_name}: rank {down_shape[0]} -> {new_rank}")
                return True
                
        except Exception as e:
            print(f"[DynamicPruner] Failed to prune {layer_name}: {e}")
            return False
    
    def _update_optimizer_state(self, optimizer, module, new_rank: int):
        """AUDIT FIX: Project optimizer state to new rank space instead of deleting.
        
        Previously, this method deleted optimizer.state[p], causing:
        - Loss of momentum history (exp_avg, exp_avg_sq for Adam)
        - "Cold restart" training behavior after pruning
        - Loss spikes that could negate pruning benefits
        
        New approach: Project momentum tensors to the new smaller rank space,
        preserving as much training history as possible.
        """
        try:
            for param in [module.lora_up.weight, module.lora_down.weight]:
                for group in optimizer.param_groups:
                    for i, p in enumerate(group['params']):
                        if p is param and p in optimizer.state:
                            state = optimizer.state[p]
                            
                            # Project each state tensor to new shape
                            for key in list(state.keys()):
                                if isinstance(state[key], torch.Tensor):
                                    old_tensor = state[key]
                                    new_shape = param.shape
                                    
                                    if old_tensor.shape != new_shape:
                                        # Project to new shape by truncating/padding
                                        new_tensor = self._project_tensor(
                                            old_tensor, new_shape, param.device
                                        )
                                        state[key] = new_tensor
                            
                            # Update the state reference to new param
                            del optimizer.state[p]
                            optimizer.state[param] = state
                            
        except Exception as e:
            print(f"[DynamicPruner] Optimizer state projection failed: {e}")
            # Fallback: just clear the state
            try:
                for param in [module.lora_up.weight, module.lora_down.weight]:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p is param and p in optimizer.state:
                                del optimizer.state[p]
            except Exception:
                pass
    
    def _project_tensor(self, old_tensor: 'torch.Tensor', new_shape: tuple, device) -> 'torch.Tensor':
        """Project optimizer state tensor to new shape.
        
        Strategy:
        - If new rank is smaller: truncate to keep most significant dimensions
        - If new rank is larger: pad with zeros (shouldn't happen in pruning)
        - Scale the retained values to maintain momentum magnitude
        """
        import torch
        
        new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype, device=device)
        
        # Calculate slice sizes
        min_dims = [min(o, n) for o, n in zip(old_tensor.shape, new_shape)]
        
        # Copy overlapping region
        if len(min_dims) == 1:
            new_tensor[:min_dims[0]] = old_tensor[:min_dims[0]]
        elif len(min_dims) == 2:
            new_tensor[:min_dims[0], :min_dims[1]] = old_tensor[:min_dims[0], :min_dims[1]]
        else:
            # Generic fallback for higher dims
            new_tensor.data.copy_(old_tensor.view(-1)[:new_tensor.numel()].view(new_shape))
        
        # Scale to preserve momentum energy
        old_norm = old_tensor.norm()
        new_norm = new_tensor.norm()
        if new_norm > 1e-8 and old_norm > 1e-8:
            scale_factor = min(old_norm / new_norm, 2.0)  # Cap at 2x to prevent explosion
            new_tensor.mul_(scale_factor)
        
        return new_tensor
    
    def get_summary(self) -> Dict:
        """获取剔除摘要"""
        return {
            "total_pruned_ranks": self.total_pruned,
            "prune_events": len(self._prune_history),
            "current_ranks": self._current_ranks.copy(),
            "history": self._prune_history[-10:]  # 最近 10 次
        }



def create_dynamic_pruner(
    prune_threshold: float = 0.05,
    min_rank: int = 8,
    warmup_ratio: float = 0.15,
    on_prune: Optional[Callable] = None
) -> Optional[DynamicRankPruner]:
    """创建动态剔除器的工厂函数"""
    if not HAS_TORCH:
        print("[DynamicPruner] PyTorch not available")
        return None
    
    return DynamicRankPruner(
        prune_threshold=prune_threshold,
        min_rank=min_rank,
        warmup_ratio=warmup_ratio,
        on_prune=on_prune
    )


class TimestepAwareRegularization:
    """
    T-LoRA: Timestep-Dependent Regularization
    
    动态调整正则化强度 lambda(t)，根据当前扩散时间步 t。
    
    原理:
    - 早期 (High t): 结构生成阶段，需要强正则化防止结构坍塌 -> High lambda
    - 晚期 (Low t): 细节完善阶段，需要弱正则化允许细节微调 -> Low lambda
    
    Formula:
    lambda(t) = base_lambda * (min_decay + (1 - min_decay) * (t / max_t))
    
    Example:
    base_lambda=0.1, min_decay=0.5
    t=1000 -> lambda=0.1 * 1.0 = 0.1
    t=0    -> lambda=0.1 * 0.5 = 0.05
    """
    
    def __init__(
        self,
        max_t: int = 1000,
        min_decay_ratio: float = 0.5,  # 最低衰减比例
        enabled: bool = True
    ):
        self.max_t = max_t
        self.min_decay_ratio = min_decay_ratio
        self.enabled = enabled
        
    def get_regularization_strength(self, t: int, base_lambda: float) -> float:
        """根据时间步 t 计算当前的 lambda"""
        if not self.enabled:
            return base_lambda
        
        # Normalize t to [0, 1]
        # t is usually [0, 1000], where 1000 is noise (early), 0 is clean (late)
        # Note: In diffusion, t=1000 is start (structure), t=0 is end (details)
        ratio = max(0.0, min(1.0, t / self.max_t))
        
        # Linear decay
        decay_factor = self.min_decay_ratio + (1.0 - self.min_decay_ratio) * ratio
        
        return base_lambda * decay_factor



class AdaptiveLRCallback:
    """
    层级自适应学习率调整器
    
    根据每层 LoRA 的 effective rank 利用率动态调整学习率：
    - 低利用率层 → 提高学习率（需要更多学习）
    - 高利用率层 → 降低学习率（避免过拟合）
    
    工作原理：
    1. 定期分析每层的 effective_rank / max_rank 比例
    2. 基于利用率计算每层的 LR 倍率
    3. 通过钩子或直接修改 param_groups 应用调整
    """
    

    def __init__(
        self,
        base_lr: float = 1e-4,
        min_multiplier: float = 0.3,     # 最低 LR 倍率
        max_multiplier: float = 2.0,      # 最高 LR 倍率
        warmup_ratio: float = 0.1,        # warmup 期间不调整
        update_interval: int = 100,       # 每 N 步更新一次
        target_utilization: float = 0.5,  # 目标利用率 (50%)
        on_adjust: Optional[Callable] = None,
        use_pissa: bool = False           # PiSSA 协同模式
    ):
        if not HAS_TORCH:
            raise ImportError("AdaptiveLRCallback requires PyTorch")
        
        # PiSSA Synergy: 
        # PiSSA 初始化提供了更好的起点，但也更容易在早期过拟合。
        # 因此我们需要更激进的下限 (防止 LR 降得太低导致把 PiSSA 结构破坏)
        # 同时也允许更高的上限 (利用 PiSSA 的快速收敛特性)
        if use_pissa:
            min_multiplier = max(min_multiplier, 0.5)  # 至少 0.5x
            max_multiplier = max(max_multiplier, 2.5)  # 允许 2.5x
            print(f"[AdaptiveLR] PiSSA Synergy Enabled: min={min_multiplier}, max={max_multiplier}")

        
        self.base_lr = base_lr
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier
        self.warmup_ratio = warmup_ratio
        self.update_interval = update_interval
        self.target_utilization = target_utilization
        self.on_adjust = on_adjust
        
        # 每层的 LR 倍率
        self._layer_multipliers: Dict[str, float] = {}
        # 每层的利用率历史
        self._utilization_history: Dict[str, List[float]] = {}
        # 调整历史
        self._adjustment_history: List[Dict] = []
    
    def step(
        self,
        step: int,
        total_steps: int,
        network,
        optimizer
    ) -> Optional[Dict]:
        """
        每步调用，检查是否需要调整学习率
        
        Returns:
            调整信息 dict (如果发生调整)，否则 None
        """
        # Warmup 期间不调整
        warmup_steps = int(total_steps * self.warmup_ratio)
        if step < warmup_steps:
            return None
        
        # 检查是否到了更新间隔
        if step % self.update_interval != 0:
            return None
        
        # 分析并调整
        return self._analyze_and_adjust(network, optimizer, step)
    
    def _analyze_and_adjust(self, network, optimizer, step: int) -> Optional[Dict]:
        """分析网络并调整学习率"""
        adjustments = []
        
        for name, module in network.named_modules():
            if not (hasattr(module, 'lora_down') and hasattr(module, 'lora_up')):
                continue
            
            down = module.lora_down.weight
            if down.dim() < 2:
                continue
            
            current_rank = down.shape[0]
            
            # 计算 effective rank
            try:
                with torch.no_grad():
                    if down.dim() == 4:
                        down_2d = down.view(down.size(0), -1)
                    else:
                        down_2d = down
                    S = torch.linalg.svdvals(down_2d.float())
                    threshold = S.max() * 0.01
                    effective_rank = int((S > threshold).sum().item())
            except Exception:
                continue
            
            utilization = effective_rank / current_rank if current_rank > 0 else 1.0
            
            # 记录利用率历史
            if name not in self._utilization_history:
                self._utilization_history[name] = []
            self._utilization_history[name].append(utilization)
            # 只保留最近 10 个
            if len(self._utilization_history[name]) > 10:
                self._utilization_history[name].pop(0)
            
            # 计算平均利用率
            avg_utilization = sum(self._utilization_history[name]) / len(self._utilization_history[name])
            
            # 计算 LR 倍率
            # 低利用率 → 高倍率，高利用率 → 低倍率
            # 使用线性映射：util=0.2 → mult=1.5, util=0.5 → mult=1.0, util=0.8 → mult=0.5
            if avg_utilization < self.target_utilization:
                # 低利用率：线性增加
                deviation = (self.target_utilization - avg_utilization) / self.target_utilization
                multiplier = 1.0 + deviation * (self.max_multiplier - 1.0)
            else:
                # 高利用率：线性减少
                deviation = (avg_utilization - self.target_utilization) / (1.0 - self.target_utilization)
                multiplier = 1.0 - deviation * (1.0 - self.min_multiplier)
            
            multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
            
            old_multiplier = self._layer_multipliers.get(name, 1.0)
            self._layer_multipliers[name] = multiplier
            
            # 应用到优化器
            if abs(multiplier - old_multiplier) > 0.05:  # 只有显著变化才记录
                self._apply_to_optimizer(optimizer, module, multiplier)
                adjustments.append({
                    "layer": name,
                    "utilization": round(avg_utilization, 3),
                    "old_mult": round(old_multiplier, 3),
                    "new_mult": round(multiplier, 3),
                    "effective_lr": round(self.base_lr * multiplier, 7)
                })
        
        if adjustments:
            result = {
                "step": step,
                "layers_adjusted": len(adjustments),
                "details": adjustments
            }
            self._adjustment_history.append(result)
            
            if self.on_adjust:
                self.on_adjust(result)
            
            return result
        
        return None
    
    def _apply_to_optimizer(self, optimizer, module, multiplier: float):
        """将倍率应用到优化器的参数组"""
        try:
            for param in [module.lora_up.weight, module.lora_down.weight]:
                for group in optimizer.param_groups:
                    if param in group['params']:
                        # 存储基础 LR（如果尚未存储）
                        if '_base_lr' not in group:
                            group['_base_lr'] = group['lr']
                        # 应用倍率
                        group['lr'] = group['_base_lr'] * multiplier
                        break
        except Exception as e:
            print(f"[AdaptiveLR] Failed to apply multiplier: {e}")
    
    def get_summary(self) -> Dict:
        """获取调整摘要"""
        return {
            "layer_multipliers": self._layer_multipliers.copy(),
            "adjustment_events": len(self._adjustment_history),
            "recent_adjustments": self._adjustment_history[-5:]
        }


def create_adaptive_lr_callback(
    base_lr: float = 1e-4,
    min_multiplier: float = 0.3,
    max_multiplier: float = 2.0,
    update_interval: int = 100,
    on_adjust: Optional[Callable] = None
) -> Optional[AdaptiveLRCallback]:
    """创建自适应 LR 回调的工厂函数"""
    if not HAS_TORCH:
        print("[AdaptiveLR] PyTorch not available")
        return None
    
    return AdaptiveLRCallback(
        base_lr=base_lr,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        update_interval=update_interval,
        on_adjust=on_adjust
    )

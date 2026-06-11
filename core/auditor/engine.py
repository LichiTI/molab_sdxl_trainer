import torch
import torch.nn.functional as F
import random
import logging
from typing import Dict, Any, Optional
from .types import AuditConfig, AuditMode, SVDAlgorithm

logger = logging.getLogger("MetricEngine")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class MetricEngine:
    """指标计算引擎"""
    
    def __init__(self, config: AuditConfig):
        self.config = config
        
        # Gradient Coherence 状态
        self._grad_proj_matrix: Optional[torch.Tensor] = None
        self._last_grad_proj: Optional[torch.Tensor] = None
        
        # CLIP Drift 基准
        self._ref_eos_feat: Optional[torch.Tensor] = None
        self._ref_mean_feat: Optional[torch.Tensor] = None
        
        # Activation Drift 基准
        self._ref_act_features: Optional[torch.Tensor] = None
    
    # ========== 权重拓扑指标 ==========
    
    def compute_topology(self, weight_tensor: 'torch.Tensor', mode: AuditMode) -> Dict[str, float]:
        """
        计算权重拓扑指标：Stable Rank, SVD Entropy, Spectral Smoothness
        共享一次 SVD/rSVD 计算
        """
        if not HAS_TORCH:
            return {}
        
        try:
            # 移至 CPU 计算，避免显存尖峰
            W = weight_tensor.detach().float().cpu()
            
            # 处理 4D 权重 (Conv)
            if W.dim() == 4:
                W = W.view(W.size(0), -1)
            elif W.dim() != 2:
                return {}
            
            # 计算奇异值
            if self.config.svd_algorithm == SVDAlgorithm.RSVD:
                S = self._compute_rsvd(W, mode)
            elif self.config.svd_algorithm == SVDAlgorithm.BRANDS:
                # TODO: 接入 IncrementalSVDEngine 完整实现
                # 目前 fallback 到 rSVD
                S = self._compute_rsvd(W, mode)
            else:
                S = self._compute_standard_svd(W)
            
            if S is None or len(S) == 0:
                return {}
            
            # 计算指标
            results = {}
            
            # 1. Stable Rank: sum(σ²) / σ_max²
            max_sv = S[0]
            if max_sv > 1e-8:
                stable_rank = (torch.sum(S ** 2) / (max_sv ** 2 + 1e-8)).item()
                results["stable_rank"] = round(stable_rank, 3)
            
            # 2. SVD Entropy: -sum(p * log2(p))
            s_sum = torch.sum(S) + 1e-8
            p = S / s_sum
            entropy = -torch.sum(p * torch.log2(p + 1e-8)).item()
            results["svd_entropy"] = round(entropy, 3)
            
            # 3. Spectral Smoothness: σ_1 - σ_2
            if len(S) >= 2:
                spectral_gap = (S[0] - S[1]).item()
                results["spectral_smoothness"] = round(spectral_gap, 5)
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_rsvd(self, W: 'torch.Tensor', mode: AuditMode) -> Optional['torch.Tensor']:
        """
        随机化 SVD (rSVD)
        
        数学原理: Johnson-Lindenstrauss 引理
        将大矩阵投影到低维空间，其奇异值分布近似不变
        
        σ = SVDvals(W · Ω), Ω ~ N(0, 1)
        """
        k = self.config.rsvd_k_pro if mode == AuditMode.PRO else self.config.rsvd_k_lite
        height, width = W.shape
        
        # 构造随机投影矩阵 Ω (width, k)
        omega = torch.randn((width, k), dtype=torch.float32)
        
        # 降维投影 Y = W · Ω, shape: (height, k)
        Y = torch.mm(W, omega)
        
        # 计算小矩阵的奇异值
        S = torch.linalg.svdvals(Y)
        
        return S
    
    def _compute_standard_svd(self, W: 'torch.Tensor') -> Optional['torch.Tensor']:
        """标准 SVD 计算"""
        S = torch.linalg.svdvals(W)
        return S
    
    def compute_dead_neuron_rate(self, weight_tensor: 'torch.Tensor', mode: AuditMode) -> float:
        # ... logic omitted for brevity in replace ...
        stride = self.config.dead_neuron_stride_pro if mode == AuditMode.PRO else self.config.dead_neuron_stride_lite
        epsilon = self.config.dead_neuron_epsilon
        
        # Zero-copy 切片采样
        w_sampled = weight_tensor.detach().view(-1)[::stride]
        
        dead_num = (torch.abs(w_sampled) < epsilon).float().sum()
        total_num = w_sampled.numel()
        
        return round((dead_num / total_num).item(), 4)
    
    def compute_rms(self, weight_tensor: 'torch.Tensor') -> float:
        """
        计算均方根 (Root Mean Square)
        """
        if not HAS_TORCH:
            return 0.0
        
        w = weight_tensor.detach().float()
        rms = torch.sqrt(torch.mean(w**2))
        return round(rms.item(), 6)
    
    # ========== 训练动力学指标 ==========
    
    def compute_update_ratio(self, optimizer) -> float:
        """
        权重更新比
        
        数学定义: Ratio = ||ΔW||_F / ||W||_F = ||η·∇W||_F / ||W||_F
        """
        if not HAS_TORCH:
            return 0.0
        
        update_norm_sq = 0.0
        param_norm_sq = 0.0
        
        for group in optimizer.param_groups:
            lr = group.get('lr', 0)
            for p in group['params']:
                if p.grad is not None:
                    update_norm_sq += (torch.norm(p.grad).item() * lr) ** 2
                    param_norm_sq += torch.norm(p.data).item() ** 2
        
        if param_norm_sq < 1e-8:
            return 0.0
        
        ratio = (update_norm_sq ** 0.5) / (param_norm_sq ** 0.5 + 1e-8)
        return round(ratio, 6)
    
    def compute_grad_coherence(self, grad_tensor: 'torch.Tensor', mode: AuditMode) -> float:
        """
        梯度一致性
        
        数学定义: CosineSimilarity(v_t, v_{t-1})
        
        使用随机子空间投影将梯度压缩到低维
        """
        if not HAS_TORCH:
            return 1.0
        
        dim = self.config.grad_proj_dim_pro if mode == AuditMode.PRO else self.config.grad_proj_dim_lite
        
        # 展平并移至 CPU
        g = grad_tensor.detach().cpu().view(1, -1)
        
        # 初始化投影矩阵 (只需一次)
        if self._grad_proj_matrix is None or self._grad_proj_matrix.shape[0] != g.shape[1]:
            self._grad_proj_matrix = torch.randn((g.shape[1], 8), dtype=torch.float32)
        
        # 投影到低维空间
        P_use = self._grad_proj_matrix[:, :dim]
        v_curr = torch.mm(g, P_use)
        
        # 计算余弦相似度
        if self._last_grad_proj is None:
            self._last_grad_proj = v_curr
            return 1.0
        
        sim = F.cosine_similarity(v_curr, self._last_grad_proj, dim=1).item()
        self._last_grad_proj = v_curr
        
        return round(sim, 4)
    
    def compute_gsnr(self, grad_tensor: 'torch.Tensor') -> float:
        """
        梯度信噪比 (Gradient Signal-to-Noise Ratio)
        
        数学定义: GSNR = E[g]² / Var(g)
        """
        if not HAS_TORCH:
            return 0.0
        
        g = grad_tensor.detach()
        mean_sq = torch.mean(g).item() ** 2
        var = torch.var(g).item()
        
        gsnr = mean_sq / (var + 1e-8)
        return round(gsnr, 4)
    
    # ========== 语义与画质指标 ==========
    
    def compute_noise_pred_std(self, noise_pred: 'torch.Tensor') -> float:
        """噪声预测标准差"""
        if not HAS_TORCH:
            return 0.0
        return round(noise_pred.detach().std().item(), 4)
    
    def compute_attn_entropy(self, attn_map: 'torch.Tensor', step: int, mode: AuditMode) -> float:
        """
        注意力熵
        
        数学定义: H(A) = -(1/HW) * sum(A * log(A + ε))
        
        使用动态轮询采样单个 Head
        """
        if not HAS_TORCH:
            return 0.0
        
        try:
            # attn_map shape: (Batch, Heads, H, W) 或类似
            if attn_map.dim() < 3:
                return 0.0
            
            num_heads = attn_map.shape[1] if attn_map.dim() >= 2 else 1
            
            if mode == AuditMode.PRO:
                # 轮询采样
                head_idx = step % num_heads
            else:
                # 随机采样
                head_idx = random.randint(0, num_heads - 1)
            
            # 切片 (GPU 内操作)
            if attn_map.dim() >= 4:
                probs = attn_map[:, head_idx, :, :]
            else:
                probs = attn_map
            
            # 归一化 (如果需要)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum()
            
            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-8)) / probs.numel()
            
            return round(entropy.item(), 4)
        except Exception:
            return 0.0
    
    # ========== 几何与安全指标 ==========
    
    def compute_act_drift(self, features: 'torch.Tensor', mode: AuditMode) -> float:
        """
        激活值漂移
        
        使用 Global Average Pooling 降维后计算
        """
        if not HAS_TORCH:
            return 0.0
        
        try:
            # GAP 降维: (B, C, H, W) -> (B, C)
            if features.dim() == 4:
                pooled = torch.mean(features.detach(), dim=[2, 3])
            else:
                pooled = features.detach()
            
            # LITE 模式进一步降维
            if mode == AuditMode.LITE:
                pooled = pooled[:, :128]
            
            pooled = pooled.cpu()
            
            # 初始化基准
            if self._ref_act_features is None:
                self._ref_act_features = pooled.clone()
                return 0.0
            
            # 计算漂移 (Frobenius 范数)
            drift = torch.norm(pooled - self._ref_act_features).item()
            return round(drift, 4)
        except Exception:
            return 0.0

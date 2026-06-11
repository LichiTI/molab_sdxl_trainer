"""
流形几何约束 (Manifold Constraint / Geometric Lock)

基于 Johnson-Lindenstrauss 引理的随机投影流形约束。
使用 Newton-Schulz 迭代计算稳定的 Log-Euclidean 距离。

技术原理:
1. 将高维特征 (4096D) 投影到低维空间 (128D)
2. 使用黎曼几何的 Log-Euclidean 距离约束 Gram 矩阵
3. Newton-Schulz 迭代替代 EVD，确保数值稳定性

矩阵对数算法选项:
- SQRT_PROXY: log(A) ≈ 2(A^{1/2} - I) 快速但仅在 A ≈ I 时精确
- PADE_SERIES: Padé [4/4] 近似，中等精度 (推荐)
- SCALING_SQUARING: Higham 稳定算法，最高精度但较慢
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Any
from enum import Enum


class LogMethod(Enum):
    """矩阵对数计算方法"""
    SQRT_PROXY = "sqrt_proxy"           # 快速: log(A) ≈ 2(sqrt(A) - I)
    PADE_SERIES = "pade_series"         # 中等: Padé [4/4] 近似
    SCALING_SQUARING = "scaling_squaring"  # 最高精度: Higham 方法


class ManifoldConstraint:
    """
    流形约束模块 - 在低维投影空间中约束权重矩阵的拓扑结构
    
    使用方式:
        constraint = ManifoldConstraint(proj_dim=128, log_method=LogMethod.PADE_SERIES)
        loss = constraint.compute_loss(student_features, teacher_features)
    """
    
    def __init__(
        self,
        proj_dim: int = 128,
        anchor_layers: Optional[List[str]] = None,
        newton_schulz_iterations: int = 5,
        epsilon: float = 1e-6,
        device: str = "cuda",
        log_method: LogMethod = LogMethod.PADE_SERIES,
        sparse_freq: int = 1
    ):
        """
        Args:
            proj_dim: 投影维度 (推荐 128，基于 JL 引理)
            anchor_layers: 锚点层名称列表 (None = 自动选择)
            newton_schulz_iterations: NS 迭代次数 (推荐 5)
            epsilon: 数值稳定性小量
            device: 计算设备
            log_method: 矩阵对数计算方法
            sparse_freq: 稀疏计算频率 (1 = 每步计算)
        """
        self.proj_dim = proj_dim
        self.anchor_layers = anchor_layers
        self.ns_iters = newton_schulz_iterations
        self.epsilon = epsilon
        self.device = device
        self.log_method = log_method
        self.sparse_freq = sparse_freq
        
        self._step_counter = 0
        self._last_loss = torch.tensor(0.0, device=self.device)
        
        # 投影矩阵缓存 (按输入维度)
        self._projection_matrices: Dict[int, torch.Tensor] = {}
        
        # 基线 Gram 矩阵缓存 (训练开始时记录)
        self._baseline_grams: Dict[str, torch.Tensor] = {}
    
    def get_projection_matrix(self, input_dim: int) -> torch.Tensor:
        """
        获取或创建正交投影矩阵
        
        使用 QR 分解确保行向量正交，防止特征碰撞
        """
        if input_dim not in self._projection_matrices:
            # 随机高斯矩阵
            H = torch.randn(input_dim, self.proj_dim, device=self.device)
            # QR 分解获取正交矩阵
            Q, _ = torch.linalg.qr(H)
            # 取前 proj_dim 列的转置作为投影矩阵
            R = Q[:, :self.proj_dim].t()  # [proj_dim, input_dim]
            self._projection_matrices[input_dim] = R
        
        return self._projection_matrices[input_dim]
    
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """
        将特征投影到低维空间
        
        Args:
            features: [batch, seq_len, hidden_dim] 或 [batch, hidden_dim]
        
        Returns:
            projected: [batch, seq_len, proj_dim] 或 [batch, proj_dim]
        """
        input_dim = features.shape[-1]
        R = self.get_projection_matrix(input_dim)
        
        # 应用投影: proj = features @ R.T
        return torch.matmul(features, R.t())
    
    def compute_gram(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算投影后的 Gram 矩阵
        
        Args:
            features: [batch, seq_len, hidden_dim]
        
        Returns:
            gram: [batch, proj_dim, proj_dim]
        """
        # 投影到低维
        proj = self.project(features)  # [batch, seq_len, proj_dim]
        
        # 计算 Gram 矩阵: G = F^T @ F (归一化)
        proj_t = proj.transpose(-2, -1)  # [batch, proj_dim, seq_len]
        gram = torch.bmm(proj_t, proj)  # [batch, proj_dim, proj_dim]
        
        # 归一化 (按 seq_len)
        gram = gram / proj.shape[1]
        
        return gram
    
    def newton_schulz_sqrt(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Newton-Schulz 迭代计算矩阵平方根及其逆
        
        比 EVD 更稳定，特别是在混合精度下
        
        Args:
            A: SPD 矩阵 [batch, n, n]
        
        Returns:
            Y: A^(1/2)
            Z: A^(-1/2)
        """
        # 强制 FP32 计算
        original_dtype = A.dtype
        A = A.to(torch.float32)
        
        n = A.shape[-1]
        I = torch.eye(n, device=A.device, dtype=torch.float32).unsqueeze(0)
        
        # 谱归一化 (确保收敛)
        norm_A = torch.linalg.matrix_norm(A, ord=2, keepdim=True).unsqueeze(-1)
        A_scaled = A / (norm_A + self.epsilon)
        
        # 初始化
        Y = A_scaled
        Z = I.expand_as(A)
        
        # Newton-Schulz 迭代
        for _ in range(self.ns_iters):
            T = 3 * I - Z @ Y
            Y = 0.5 * Y @ T
            Z = 0.5 * T @ Z
        
        # 还原尺度
        Y = Y * torch.sqrt(norm_A + self.epsilon)
        Z = Z / torch.sqrt(norm_A + self.epsilon)
        
        return Y.to(original_dtype), Z.to(original_dtype)
    
    def log_euclidean_distance(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        计算 Log-Euclidean 距离
        
        D_LE(A, B) = ||log(A) - log(B)||_F
        
        使用选择的矩阵对数算法
        """
        # 确保 SPD (添加小量正则)
        I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        A = A + self.epsilon * I
        B = B + self.epsilon * I
        
        # 根据选择的方法计算矩阵对数
        log_A = self._compute_matrix_log(A)
        log_B = self._compute_matrix_log(B)
        
        # Frobenius 范数
        diff = log_A - log_B
        distance = torch.sqrt((diff ** 2).sum(dim=(-2, -1)) + self.epsilon)
        
        return distance.mean()
    
    def _compute_matrix_log(self, A: torch.Tensor) -> torch.Tensor:
        """根据选择的方法计算矩阵对数"""
        if self.log_method == LogMethod.SQRT_PROXY:
            return self._matrix_log_sqrt_proxy(A)
        elif self.log_method == LogMethod.PADE_SERIES:
            return self._matrix_log_pade(A)
        elif self.log_method == LogMethod.SCALING_SQUARING:
            return self._matrix_log_scaling_squaring(A)
        else:
            # 默认使用 Padé
            return self._matrix_log_pade(A)
    
    def _matrix_log_sqrt_proxy(self, A: torch.Tensor) -> torch.Tensor:
        """
        快速近似: log(A) ≈ 2(A^{1/2} - I)
        
        优点: 最快速度，低内存
        缺点: 仅在 A ≈ I 时精确，远离单位阵时误差增大
        适用: 实时监控、快速迭代
        """
        n = A.shape[-1]
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        sqrt_A, _ = self.newton_schulz_sqrt(A)
        return 2 * (sqrt_A - I)
    
    def _matrix_log_pade(self, A: torch.Tensor) -> torch.Tensor:
        """
        Padé [4/4] 近似矩阵对数
        
        基于 log(A) = log(I + (A-I)) ≈ Padé(A-I)
        对于 ||A - I|| < 1，精度显著优于 sqrt proxy
        
        优点: 平衡精度与速度 (推荐默认)
        缺点: 需要 A 不能离 I 太远
        """
        original_dtype = A.dtype
        A = A.to(torch.float32)
        
        n = A.shape[-1]
        I = torch.eye(n, device=A.device, dtype=torch.float32)
        
        # 将 A 缩放到接近 I，使用 Newton-Schulz 计算根
        norm_A = torch.linalg.matrix_norm(A, ord='fro', keepdim=True).unsqueeze(-1)
        s = max(0, int(torch.ceil(torch.log2(norm_A.max() + 1)).item()))
        
        # A^{1/2^s} 使其接近 I
        B = A
        for _ in range(s):
            B_sqrt, _ = self.newton_schulz_sqrt(B)
            B = B_sqrt
        
        # X = B - I，此时 ||X|| < 1
        X = B - I
        
        # Padé [4/4] 系数 (Higham style)
        # log(I+X) ≈ X @ (c0*I + c1*X + c2*X^2 + c3*X^3 + c4*X^4) 
        #          / (d0*I + d1*X + d2*X^2 + d3*X^3 + d4*X^4)
        # 简化使用 [2/2] 近似以保持效率:
        # log(I+X) ≈ X(6I + X) / (6I + 4X + X^2)
        X2 = X @ X
        numer = X @ (6 * I + X)
        denom = 6 * I + 4 * X + X2
        
        # 求解 log_B = denom^{-1} @ numer
        try:
            log_B = torch.linalg.solve(denom, numer)
        except Exception:
            # Fallback: 使用伪逆
            log_B = torch.linalg.lstsq(denom, numer).solution
        
        # 还原缩放: log(A) = 2^s * log(B)
        log_A = log_B * (2 ** s)
        
        return log_A.to(original_dtype)
    
    def _matrix_log_scaling_squaring(self, A: torch.Tensor) -> torch.Tensor:
        """
        Higham 稳定算法 - Scaling and Squaring Method
        
        参考: N.J. Higham, "Functions of Matrices: Theory and Computation"
        
        优点: 最高精度，数值稳定
        缺点: 较慢，需要更多计算
        适用: 最终评估、高精度需求
        """
        original_dtype = A.dtype
        A = A.to(torch.float32)
        
        n = A.shape[-1]
        I = torch.eye(n, device=A.device, dtype=torch.float32)
        
        # 步骤 1: 缩放 - 计算平方根直到 ||A^{1/2^s} - I|| < 0.5
        max_s = 10  # 最大缩放次数
        B = A
        s = 0
        
        for s in range(max_s):
            diff_norm = torch.linalg.matrix_norm(B - I, ord='fro')
            if (diff_norm < 0.5).all():
                break
            B_sqrt, _ = self.newton_schulz_sqrt(B)
            B = B_sqrt
        else:
            # 未能收敛，使用当前值
            pass
        
        # 步骤 2: 使用 Padé [4/4] 对 log(B) 进行高精度计算
        X = B - I
        X2 = X @ X
        X3 = X2 @ X
        X4 = X3 @ X
        
        # Padé [4/4] 系数 (来自 Higham)
        c0, c1, c2, c3, c4 = 1.0, 1.0, 0.5, 1/6, 1/24
        d0, d1, d2, d3, d4 = 1.0, -0.5, 1/12, -1/120, 1/5040
        
        # 简化: 使用 [3/3] 近似以平衡速度
        # log(I+X) ≈ X(60I + 60X + 11X^2) / (60I + 90X + 36X^2 + 3X^3)
        numer = X @ (60 * I + 60 * X + 11 * X2)
        denom = 60 * I + 90 * X + 36 * X2 + 3 * X3
        
        try:
            log_B = torch.linalg.solve(denom, numer)
        except Exception:
            log_B = torch.linalg.lstsq(denom, numer).solution
        
        # 步骤 3: 还原缩放
        log_A = log_B * (2 ** s)
        
        return log_A.to(original_dtype)
    
    @staticmethod
    def get_method_info(method: LogMethod = None) -> Dict[str, str]:
        """获取矩阵对数方法的详细说明"""
        info = {
            LogMethod.SQRT_PROXY: {
                "name": "平方根代理 (Sqrt Proxy)",
                "formula": "log(A) ≈ 2(√A - I)",
                "speed": "最快",
                "accuracy": "低 (仅在 A ≈ I 时)",
                "use_case": "实时监控、快速原型"
            },
            LogMethod.PADE_SERIES: {
                "name": "帕德级数 (Padé Series)",
                "formula": "log(I+X) ≈ X(6I+X)/(6I+4X+X²)",
                "speed": "中等",
                "accuracy": "中高",
                "use_case": "推荐默认、日常训练"
            },
            LogMethod.SCALING_SQUARING: {
                "name": "缩放平方法 (Scaling-Squaring)",
                "formula": "Higham 稳定算法 + Padé [3/3]",
                "speed": "较慢",
                "accuracy": "最高",
                "use_case": "最终评估、学术研究"
            }
        }
        if method:
            return info.get(method, {})
        return info
    
    def capture_baseline(self, model: nn.Module, sample_inputs: Any) -> None:
        """
        在训练开始前捕获基线 Gram 矩阵
        
        Args:
            model: 底座模型
            sample_inputs: 代表性输入样本
        """
        self._baseline_grams.clear()
        
        # 注册 hook 捕获中间特征
        hooks = []
        features = {}
        
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    features[name] = output.detach()
            return hook
        
        for name, module in model.named_modules():
            if self._should_track_layer(name):
                hooks.append(module.register_forward_hook(create_hook(name)))
        
        try:
            with torch.no_grad():
                if isinstance(sample_inputs, dict):
                    model(**sample_inputs)
                else:
                    model(sample_inputs)

            for name, feat in features.items():
                gram = self.compute_gram(feat)
                self._baseline_grams[name] = gram.detach()
        finally:
            for hook in hooks:
                hook.remove()
    
    def _should_track_layer(self, name: str) -> bool:
        """判断是否应该追踪该层"""
        if self.anchor_layers is not None:
            return any(anchor in name for anchor in self.anchor_layers)
        
        # 自动选择策略 (针对 Flux/SDXL)
        name_lower = name.lower()
        
        # Flux: Double Blocks 4, 10 和 Single Block 0
        if "double" in name_lower and any(f".{i}." in name for i in [4, 10]):
            return True
        if "single" in name_lower and ".0." in name:
            return True
        
        # SDXL: Mid Block (最敏感)
        if "mid_block" in name_lower:
            return True
        
        return False
    
    def compute_loss(
        self,
        current_features: Dict[str, torch.Tensor],
        weight: float = 1.0
    ) -> torch.Tensor:
        """
        计算流形约束损失 (支持稀疏采样)
        """
        if not self._baseline_grams:
            return torch.tensor(0.0, device=self.device)

        # 稀疏检查
        self._step_counter += 1
        if self._step_counter % self.sparse_freq != 0:
            # 返回缓存的 Loss (detach 以防止旧图残留，但在这里它是一个 scalar)
            return weight * self._last_loss.detach()
        
        
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for name, feat in current_features.items():
            if name in self._baseline_grams:
                current_gram = self.compute_gram(feat)
                baseline_gram = self._baseline_grams[name]
                
                distance = self.log_euclidean_distance(current_gram, baseline_gram)
                total_loss = total_loss + distance
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        self._last_loss = total_loss
        return weight * total_loss

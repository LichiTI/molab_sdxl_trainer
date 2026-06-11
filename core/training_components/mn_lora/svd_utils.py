
import torch
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

def randomized_svd(
    M: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    center: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算矩阵 M 的 Randomized SVD 前 k 个奇异值/向量。
    
    算法参考: Halko et al., "Finding structure with randomness: Probabilistic
    algorithms for constructing approximate matrix decompositions" (2011).
    
    Args:
        M: 输入矩阵 [m, n]
        k: 目标秩 (rank)
        n_oversamples: 过采样参数，通常 5-10
        n_iter: 幂迭代次数，增加可提高精度 (subsidiary subspace iteration)
        center: 是否先对矩阵进行中心化 (减去均值)
        
    Returns:
        U, S, Vh
        U: [m, k]
        S: [k]
        Vh: [k, n]
    """
    # V3.1 Fix: FP16 Safety - QR/SVD operations require at least FP32
    orig_dtype = M.dtype
    if M.dtype != torch.float32 and M.dtype != torch.float64:
        M = M.float()
    
    m, n = M.shape
    k_over = min(k + n_oversamples, min(m, n))
    
    # 1. 随机投影矩阵生成
    # Omega: [n, k_over]
    Omega = torch.randn(n, k_over, device=M.device, dtype=M.dtype)
    
    # 2. 采样矩阵 Y = M @ Omega
    Y = M @ Omega
    
    # 3. 幂迭代 (Power Iteration) 以增强奇异值衰减，提高精度
    # Y = (M M^T)^q M Omega
    for _ in range(n_iter):
        Y = M @ (M.T @ Y)
        
    # 4. QR 分解获取正交基 Q
    # Q: [m, k_over]
    Q, _ = torch.linalg.qr(Y)
    
    # 5. 投影原始矩阵到小矩阵 B
    # B = Q^T M
    # B: [k_over, n]
    B = Q.T @ M
    
    # 6. 对小矩阵 B 进行完整 SVD
    # U_hat: [k_over, k_over], S: [k_over], Vh: [k_over, n]
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # 7. 恢复原始 U
    # U = Q @ U_hat
    U = Q @ U_hat
    
    # 8. 截断到 k
    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    
    # V3.1 Fix: Cast back to original dtype
    if U.dtype != orig_dtype:
        U = U.to(orig_dtype)
        S = S.to(orig_dtype)
        Vh = Vh.to(orig_dtype)
    
    return U, S, Vh

def compute_effective_rank_cpu(
    weight: torch.Tensor, 
    ratio: float = 0.5, 
    min_k: int = 1,
    use_randomized: bool = True,
    randomized_threshold: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    在 CPU 上计算有效秩子空间 V_k。
    专为节省显存设计。
    
    Args:
        weight: 权重张量
        ratio: 保留能量/秩的比例
        min_k: 最小保留秩
        use_randomized: 是否对大矩阵使用随机化 SVD
        randomized_threshold: 触发随机化 SVD 的最小维度
        
    Returns:
        V_k: [in_features, k] 右奇异向量矩阵
        S_k: [k] 奇异值向量 (Top-k)
        k: 有效秩
    """
    # 展平
    if weight.dim() == 4:
        w = weight.view(weight.size(0), -1) # [out, in]
    elif weight.dim() == 2:
        w = weight
    else:
        # 非矩阵，无法做 SVD，返回空或原样
        # 这里返回全零作为 fallback，或者抛出异常
        # 简单起见，如果维度不对，我们返回一个伪造的单位矩阵切片
        # 但实际上 GSP 应该在调用前过滤
        raise ValueError(f"Unsupported tensor dimension: {weight.dim()}")

    # 移至 CPU 计算
    w_cpu = w.detach().float().cpu()
    m, n = w_cpu.shape
    
    # 估算目标秩 k
    # 注意：这里我们基于维度做简单比例截断。
    # 更高级的做法是基于奇异值能量分布，但这需要先计算 SVD。
    # 对于 Randomized SVD，我们需要先指定 k。
    target_k = max(min_k, int(min(m, n) * ratio))
    
    # 决定是否使用 Randomized SVD
    use_rand = use_randomized and (min(m, n) > randomized_threshold)
    
    try:
        if use_rand:
            # Randomized SVD (Faster, low memory footprint)
            # Randomized SVD is already custom logic, let's wrap its result access in try-block logic if needed
            # But the underlying SVD inside randomized_svd should be safe too.
            # TODO: Refactor randomized_svd to use safe_svd internally or here.
            # For now, let's stick to using safe_svd for key components or rely on the try-except block here.
            # Actually, let's just call safe_svd wrapper if not using rand.
            _, S, Vh = randomized_svd(w_cpu, k=target_k, n_oversamples=10, n_iter=2)
            
            # Randomized SVD 已经截断到 target_k 了
            k = target_k
            V_k = Vh.T # [n, k]
            S_k = S    # [k]
            
        else:
            # Full SVD (Slow, exact)
            # Use SAFE SVD wrapper
            _, S, Vh = safe_svd(w_cpu, full_matrices=False)
            
            # 截断
            k = target_k
            V_k = Vh[:k, :].T # [n, k]
            S_k = S[:k]       # [k]
            
        return V_k, S_k, k
        
    except Exception as e:
        logger.warning(f"[SVD Utils] SVD failed: {e}")
        # Fallback must still be an orthonormal basis; otherwise g @ V @ V.T
        # is no longer an orthogonal projection and can amplify gradients.
        fallback = torch.randn(n, target_k, device=weight.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(fallback, mode="reduced")
        dummy_S = torch.ones(target_k, device=weight.device)
        return Q[:, :target_k], dummy_S, target_k

def safe_svd(
    matrix: torch.Tensor, 
    full_matrices: bool = False,
    method: str = "torch"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Safe SVD wrapper with NaN checking and fallback.
    
    Args:
        matrix: Input tensor
        full_matrices: Whether to compute full U and V
        method: "torch" or "randomized" (if randomized, calls randomized_svd)
        
    Returns:
        U, S, Vh
    """
    if not torch.isfinite(matrix).all():
        # Input contains NaNs or Infs
        # Fallback: Zero output or identity
        m, n = matrix.shape
        k = min(m, n)
        U = torch.eye(m, k, device=matrix.device, dtype=matrix.dtype)
        S = torch.zeros(k, device=matrix.device, dtype=matrix.dtype)
        Vh = torch.eye(k, n, device=matrix.device, dtype=matrix.dtype)
        return U, S, Vh
        
    try:
        if method == "randomized":
            # Randomized SVD should be called directly via randomized_svd function
            # This wrapper is primarily for standard SVD with safety checks
            import logging
            logging.getLogger(__name__).warning(
                "safe_svd called with method='randomized' - use randomized_svd() directly"
            )
            # Fall through to standard SVD as fallback
            
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=full_matrices)
        return U, S, Vh
    except Exception:
        # Fallback on failure (e.g. convergence error)
        m, n = matrix.shape
        k = min(m, n)
        return (
            torch.eye(m, k, device=matrix.device, dtype=matrix.dtype),
            torch.zeros(k, device=matrix.device, dtype=matrix.dtype),
            torch.eye(k, n, device=matrix.device, dtype=matrix.dtype)
        )

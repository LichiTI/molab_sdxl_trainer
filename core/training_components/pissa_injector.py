"""
PiSSA (Principal Singular values for Adaptation) Injector
=======================================================
Initialized standard LoRA layers with SVD components of the base model weights.
This allows fine-tuning on the "Principal Components" from step 0.

Paper: PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models

Warehouse implementation: SVD utilities inlined, no external package dependencies.
Uses explicit layer mapping (base_module_map) instead of name-prefix heuristics.
"""

from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inlined SVD utilities (self-contained, no core.mn_lora dependency)
# ---------------------------------------------------------------------------

def _randomized_svd(
    M: torch.Tensor,
    k: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomized SVD (Halko et al., 2011). Returns U[:, :k], S[:k], Vh[:k, :]."""
    orig_dtype = M.dtype
    if M.dtype not in (torch.float32, torch.float64):
        M = M.float()

    m, n = M.shape
    k_over = min(k + n_oversamples, min(m, n))

    Omega = torch.randn(n, k_over, device=M.device, dtype=M.dtype)
    Y = M @ Omega
    for _ in range(n_iter):
        Y = M @ (M.T @ Y)

    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ M
    U_hat, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
    if U.dtype != orig_dtype:
        U = U.to(orig_dtype)
        S = S.to(orig_dtype)
        Vh = Vh.to(orig_dtype)
    return U, S, Vh


def _safe_svd(
    matrix: torch.Tensor,
    full_matrices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SVD with NaN/Inf guard and convergence-error fallback."""
    if not torch.isfinite(matrix).all():
        m, n = matrix.shape
        k = min(m, n)
        return (
            torch.eye(m, k, device=matrix.device, dtype=matrix.dtype),
            torch.zeros(k, device=matrix.device, dtype=matrix.dtype),
            torch.eye(k, n, device=matrix.device, dtype=matrix.dtype),
        )
    try:
        return torch.linalg.svd(matrix, full_matrices=full_matrices)
    except Exception:
        m, n = matrix.shape
        k = min(m, n)
        return (
            torch.eye(m, k, device=matrix.device, dtype=matrix.dtype),
            torch.zeros(k, device=matrix.device, dtype=matrix.dtype),
            torch.eye(k, n, device=matrix.device, dtype=matrix.dtype),
        )


# ---------------------------------------------------------------------------
# PiSSA Injector
# ---------------------------------------------------------------------------

class PissaInjector:
    def __init__(self, rank: int, device: str = "cuda", svd_algo: str = "rsvd"):
        self.rank = rank
        self.device = device
        self.svd_algo = svd_algo

    def _compute_svd(self, weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight_2d = weight.view(weight.shape[0], -1).float()
        if self.svd_algo == "rsvd":
            U, S, Vh = _randomized_svd(weight_2d, k=rank, n_oversamples=10, n_iter=2)
            return U[:, :rank], S[:rank], Vh[:rank, :]
        else:
            U, S, Vh = _safe_svd(weight_2d, full_matrices=False)
            return U[:, :rank], S[:rank], Vh[:rank, :]

    def inject_network(
        self,
        network: nn.Module,
        unet: nn.Module,
        text_encoder: Optional[nn.Module] = None,
        base_module_map: Optional[Dict[str, nn.Module]] = None,
    ) -> int:
        """
        Inject PiSSA initialization into a LoRA-style Network.

        Args:
            network: The LoRA network (must expose named_modules with lora_down/lora_up).
            unet: The base UNet model.
            text_encoder: The base Text Encoder model (optional).
            base_module_map: Explicit mapping ``{lora_module_name: base_module}``.
                When provided, this mapping is used directly.
                When *None*, falls back to the LoRA module's ``base_module``
                attribute (if present).

        Returns:
            Number of layers processed.
        """
        logger.info(f"[PiSSA] Starting initialization (Rank={self.rank}, Algo={self.svd_algo})...")
        count = 0

        for lora_name, lora_mod in network.named_modules():
            if not (hasattr(lora_mod, "lora_down") and hasattr(lora_mod, "lora_up")):
                continue

            # --- Resolve the corresponding base module ---
            found_base_mod: Optional[nn.Module] = None

            if base_module_map is not None:
                found_base_mod = base_module_map.get(lora_name)
            elif hasattr(lora_mod, "base_module") and lora_mod.base_module is not None:
                found_base_mod = lora_mod.base_module

            if found_base_mod is None:
                logger.debug(f"[PiSSA] Skipping {lora_name}: no base module mapping")
                continue

            # --- SVD + Injection ---
            try:
                W = found_base_mod.weight.data.to(self.device)
                U_r, S_r, Vh_r = self._compute_svd(W, self.rank)

                sqrt_S = torch.sqrt(S_r)
                A_init = U_r * sqrt_S.unsqueeze(0)        # [out, rank]
                B_init = sqrt_S.unsqueeze(1) * Vh_r       # [rank, in]

                if isinstance(found_base_mod, nn.Conv2d):
                    lora_mod.lora_up.weight.data.copy_(A_init.reshape(lora_mod.lora_up.weight.shape))
                    lora_mod.lora_down.weight.data.copy_(B_init.reshape(lora_mod.lora_down.weight.shape))
                else:
                    lora_mod.lora_up.weight.data.copy_(A_init)
                    lora_mod.lora_down.weight.data.copy_(B_init)

                # Residual: W_res = W - A @ B
                resid = W.float() - (A_init @ B_init).float()
                found_base_mod.weight.data.copy_(resid.to(found_base_mod.weight.dtype))
                found_base_mod.weight.requires_grad = False

                count += 1
                if count % 10 == 0:
                    logger.info(f"[PiSSA] Processed {count} layers...")

            except Exception as e:
                logger.error(f"[PiSSA] Failed to inject {lora_name}: {e}")

        logger.info(f"[PiSSA] Initialization complete. Processed {count} layers.")
        return count



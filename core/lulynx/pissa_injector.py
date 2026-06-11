
"""
PiSSA (Principal Singular values for Adaptation) Injector
=======================================================
Initialized standard LoRA layers with SVD components of the base model weights.
This allows fine-tuning on the "Principal Components" from step 0.

Paper: PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models
"""

from typing import Dict, List, Optional, Tuple
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def _randomized_svd(A: torch.Tensor, k: int, n_oversamples: int = 10, n_iter: int = 2):
    """Randomized SVD via torch.svd_lowrank (Halko et al.)."""
    U, S, Vh = torch.svd_lowrank(A, q=k + n_oversamples, niter=n_iter)
    return U[:, :k], S[:k], Vh[:, :k].T

def _safe_svd(A: torch.Tensor, full_matrices: bool = False):
    """Full SVD with NaN guard."""
    U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)
    if torch.isnan(S).any():
        logger.warning("[safe_svd] NaN in singular values, replacing with zeros")
        S = torch.nan_to_num(S, nan=0.0)
    return U, S, Vh

class PissaInjector:
    def __init__(self, rank: int, device: str = "cuda", svd_algo: str = "rsvd"):
        self.rank = rank
        self.device = device
        self.svd_algo = svd_algo

    def _compute_svd(self, weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute SVD based on selected algorithm"""
        weight_2d = weight.view(weight.shape[0], -1).float()

        if self.svd_algo == "rsvd":
            U, S, Vh = _randomized_svd(weight_2d, k=rank, n_oversamples=10, n_iter=2)
            return U[:, :rank], S[:rank], Vh[:rank, :]
        else:
            U, S, Vh = _safe_svd(weight_2d, full_matrices=False)
            return U[:, :rank], S[:rank], Vh[:rank, :]

    def inject_network(self, network: nn.Module, unet: nn.Module, text_encoder: Optional[nn.Module] = None):
        """
        Inject PiSSA initialization into a LoRA-style Network.
        
        Args:
            network: The LoRA network object (containing lora_down/up modules)
            unet: The base UNet model
            text_encoder: The base Text Encoder model (optional)
            
        Returns:
            count: Number of layers processed
        """
        logger.info(f"[PiSSA] 🚀 Starting initialization (Rank={self.rank}, Algo={self.svd_algo})...")
        count = 0
        
        # 1. Map base model modules for easy access
        # Key: "unet" or "te", Value: dict{submodule_name: module}
        base_models = {"unet": unet}
        if text_encoder:
            base_models["te"] = text_encoder
            
        # Recursive helper to find linear/conv layers in base model
        def flatten_modules(root_model, prefix=""):
            modules = {}
            for name, mod in root_model.named_modules():
                # Store Linear and Conv2d layers
                if isinstance(mod, (nn.Linear, nn.Conv2d)):
                     modules[name] = mod
            return modules
            
        flat_base_modules = {
            "unet": flatten_modules(unet),
            "te": flatten_modules(text_encoder) if text_encoder else {}
        }

        # 2. Iterate LoRA modules
        # LoRA Network usually has 'unet_loras' and 'text_encoder_loras' lists,
        # OR named_modules() with specific naming convention.
        # Assuming we can iterate named_modules and parse naming config.
        # Standard LoRA naming: "lora_unet_down_blocks_0_resnets_0_time_emb_proj"
        # This maps to UNet: "down_blocks.0.resnets.0.time_emb_proj"
        
        for lora_name, lora_mod in network.named_modules():
            # Check if it's a LoRA module (has up/down)
            if not (hasattr(lora_mod, "lora_down") and hasattr(lora_mod, "lora_up")):
                continue
                
            # Determine if it belongs to UNet or TE
            target_key = None
            base_name_raw = None
            
            if "lora_unet_" in lora_name:
                target_key = "unet"
                # Strip prefix: "lora_unet_" -> ""
                base_name_raw = lora_name.replace("lora_unet_", "")
            elif "lora_te_" in lora_name:
                target_key = "te"
                base_name_raw = lora_name.replace("lora_te_", "")
                
            if not target_key or not base_name_raw:
                continue
                
            # Convert LoRA name to Pytorch dot-notation
            # Example: "down_blocks_0_resnets_0_time_emb_proj" -> "down_blocks.0.resnets.0.time_emb_proj"
            # NOTE: This conversion is heuristic and brittle.
            # A more robust way requires the Network to store 'org_module' references.
            # But let's try the heuristic replace first.

            found_base_mod = None

            # Simple heuristic: Split by '_' and try to reconstruct path
            # This is hard. Let's assume the network passed in (if it's Lulynx Native)
            # has attached metadata or stores 'org_module' in the LoRA module.

            if hasattr(lora_mod, 'org_module'):
                 # Best case: the original module reference is attached
                 found_base_mod = lora_mod.org_module[0] # Usually a list [org_module]
            else:
                 # Fallback: Try exact name match from a known map if available
                 # Or just skip with warning
                 # logger.warning(f"Could not map {lora_name} to base module. Skipping PiSSA init.")
                 continue # Safer to skip than guess wrong
            
            # 3. Perform SVD and Injection
            try:
                # Base Weight
                W = found_base_mod.weight.data.to(self.device)
                
                # SVD
                U_r, S_r, Vh_r = self._compute_svd(W, self.rank)
                
                # Check actual rank obtained (might be < self.rank)
                k = S_r.shape[0]
                if k < self.rank:
                    pass # Just use what we got
                    
                sqrt_S = torch.sqrt(S_r)
                
                # A = U * sqrt(S)  (out, rank)
                # B = sqrt(S) * Vh (rank, in)
                A_init = U_r * sqrt_S.unsqueeze(0)
                B_init = sqrt_S.unsqueeze(1) * Vh_r
                
                # Inject into LoRA
                # lora_up is (out, rank), lora_down is (rank, in)
                # NOTE: Linear layers are (out, in).
                # A_init shape: [out, rank]
                # B_init shape: [rank, in]
                # A @ B shape: [out, in] -> Matches W
                
                # LoRA weight layout:
                # lora_up.weight: [out, rank, 1, 1] for Conv2d or [out, rank] for Linear
                # lora_down.weight: [rank, in, 1, 1] for Conv2d or [rank, in] for Linear
                
                if isinstance(found_base_mod, nn.Conv2d):
                    lora_mod.lora_up.weight.data.copy_(A_init.reshape(lora_mod.lora_up.weight.shape))
                    lora_mod.lora_down.weight.data.copy_(B_init.reshape(lora_mod.lora_down.weight.shape))
                else:
                    lora_mod.lora_up.weight.data.copy_(A_init)
                    lora_mod.lora_down.weight.data.copy_(B_init)
                
                # Subtract from Base Weight (Residual)
                # W_res = W - (A @ B)
                # We do this calculation in high precision if possible
                
                resid = W.float() - (A_init @ B_init).float()
                found_base_mod.weight.data.copy_(resid.to(found_base_mod.weight.dtype))
                
                # Freeze Base Weight (it should be frozen anyway in LoRA, but double check)
                found_base_mod.weight.requires_grad = False
                
                count += 1
                
                if count % 10 == 0:
                    logger.info(f"[PiSSA] Processed {count} layers...")
                    
            except Exception as e:
                logger.error(f"[PiSSA] Failed to inject {lora_name}: {e}")
                
        logger.info(f"[PiSSA] ✅ Initialization complete. Processed {count} layers.")
        return count


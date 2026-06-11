"""
Model Merging Utility for Lulynx
Supports Weighted Sum and Add Difference for Checkpoints and LoRAs.
"""

import torch
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from safetensors.torch import load_file, save_file

logger = logging.getLogger("ModelMerger")

class ModelMerger:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def weighted_sum(self, model_a_path: str, model_b_path: str, output_path: str, alpha: float = 0.5, precision: str = "fp16") -> bool:
        """
        Calculates: Result = (1 - alpha) * A + alpha * B
        """
        try:
            logger.info(f"Merging models (Weighted Sum): {model_a_path} and {model_b_path} with alpha={alpha}")
            
            if not Path(model_a_path).exists() or not Path(model_b_path).exists():
                 logger.error("Input models not found")
                 return False

            state_dict_a = load_file(model_a_path, device=self.device)
            state_dict_b = load_file(model_b_path, device=self.device)
            
            merged = {}
            # Use keys from A as the primary set
            for key in state_dict_a.keys():
                if key in state_dict_b:
                    a = state_dict_a[key].to(torch.float32)
                    b = state_dict_b[key].to(torch.float32)
                    
                    # Ensure shapes match
                    if a.shape != b.shape:
                        logger.warning(f"Shape mismatch for key {key}: {a.shape} vs {b.shape}. Skipping.")
                        merged[key] = state_dict_a[key]
                        continue
                        
                    res = (1.0 - alpha) * a + alpha * b
                    
                    if precision == "fp16":
                        res = res.to(torch.float16)
                    elif precision == "bf16":
                        res = res.to(torch.bfloat16)
                        
                    merged[key] = res
                else:
                    merged[key] = state_dict_a[key]
            
            save_file(merged, output_path)
            return True
        except Exception as e:
            logger.error(f"Weighted sum merge failed: {e}")
            return False

    def add_difference(self, model_a_path: str, model_b_path: str, model_c_path: str, output_path: str, alpha: float = 1.0, precision: str = "fp16") -> bool:
        """
        Calculates: Result = A + (B - C) * alpha
        Commonly used for extracting 'difference' from a finetuned model (B) relative to base (C) and applying to A.
        """
        try:
            logger.info(f"Merging models (Add Difference): A={model_a_path}, B={model_b_path}, C={model_c_path} with alpha={alpha}")
            
            if not Path(model_a_path).exists() or not Path(model_b_path).exists() or not Path(model_c_path).exists():
                 logger.error("Input models not found")
                 return False

            state_dict_a = load_file(model_a_path, device=self.device)
            state_dict_b = load_file(model_b_path, device=self.device)
            state_dict_c = load_file(model_c_path, device=self.device)
            
            merged = {}
            for key in state_dict_a.keys():
                if key in state_dict_b and key in state_dict_c:
                    a = state_dict_a[key].to(torch.float32)
                    b = state_dict_b[key].to(torch.float32)
                    c = state_dict_c[key].to(torch.float32)
                    
                    if a.shape == b.shape == c.shape:
                        res = a + (b - c) * alpha
                        
                        if precision == "fp16":
                            res = res.to(torch.float16)
                        elif precision == "bf16":
                            res = res.to(torch.bfloat16)
                        merged[key] = res
                    else:
                        merged[key] = state_dict_a[key]
                else:
                    merged[key] = state_dict_a[key]
            
            save_file(merged, output_path)
            return True
        except Exception as e:
            logger.error(f"Add difference merge failed: {e}")
            return False

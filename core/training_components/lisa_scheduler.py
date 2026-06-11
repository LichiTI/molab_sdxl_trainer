
import torch
import torch.nn as nn
import random
import logging
from typing import List

logger = logging.getLogger("LISAScheduler")

class LISAScheduler:
    """
    LISA (Layerwise Importance Sampling) Scheduler
    
    Dynamically freezes/unfreezes layers during training to save memory (activations).
    References: "LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning"
    """
    
    def __init__(self, interval: int = 1, active_ratio: float = 0.2, verbose: bool = False):
        self.interval = interval
        self.active_ratio = active_ratio
        self.verbose = verbose
        self.trainable_modules: List[nn.Module] = []
        self._initialized = False
        
    def setup(self, model: nn.Module):
        """
        Identify trainable modules (layers) at the start.
        Ideally called after the model is fully prepared (LoRA merged/attached).
        """
        self.trainable_modules = []
        
        # Find all modules that contain immediate parameters with requires_grad=True
        # We look for "leaf-like" modules that hold weights.
        for name, module in model.named_modules():
            # Check if this module *directly* holds trainable parameters
            # recurse=False ensures we don't count container modules unless they maintain their own params
            params = list(module.parameters(recurse=False))
            has_trainable = any(p.requires_grad for p in params)
            
            if has_trainable:
                self.trainable_modules.append(module)
        
        count = len(self.trainable_modules)
        logger.info(f"[LISA] Initialized. Found {count} trainable sub-modules/layers.")
        self._initialized = True
        
    def step(self, current_step: int):
        """
        Apply LISA freezing strategy.
        Should be called at the start of a step (before forward pass).
        """
        if not self._initialized or not self.trainable_modules:
            return
            
        # Only update distribution at intervals
        if self.interval > 1 and current_step % self.interval != 0:
            return

        total_layers = len(self.trainable_modules)
        n_active = max(1, int(total_layers * self.active_ratio))
        
        # Randomly sample active layers
        active_indices = set(random.sample(range(total_layers), n_active))
        
        enabled_count = 0
        
        for i, module in enumerate(self.trainable_modules):
            should_train = i in active_indices
            
            for param in module.parameters(recurse=False):
                # We can safely toggle because these were identified as trainable at setup
                param.requires_grad = should_train
                
            if should_train:
                enabled_count += 1
                
        if self.verbose and current_step % 100 == 0:
            logger.info(f"[LISA] Step {current_step}: Active Layers {enabled_count}/{total_layers}")

"""
Feature Statistics Aligner (P0-03)
Implements statistical alignment loss for minimizing distribution shift between LLM and CLIP features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureStatisticsAligner(nn.Module):
    """
    Computes loss to align the statistics (Mean, Variance) of Student (LLM) features 
    to Teacher (CLIP) features.
    
    Why? Cosine similarity only aligns direction. But Attention Softmax is sensitive to 
    magnitude and variance (Scaling). If LLM has huge value range, it breaks Softmax.
    """
    def __init__(self, moment_weight=0.5, contrastive_weight=1.0):
        super().__init__()
        self.moment_weight = moment_weight
        self.contrastive_weight = contrastive_weight
        
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: [B, N, D] (Projected LLM)
            teacher_features: [B, N, D] (CLIP)
        Returns:
            loss: scalar
        """
        # 1. Cosine Distance (Direction Alignment)
        # Flatten to [B*N, D]
        s_flat = student_features.view(-1, student_features.shape[-1])
        t_flat = teacher_features.view(-1, teacher_features.shape[-1])
        
        # Normalized Cosine Similarity
        # Target is 1.0
        cos_sim = F.cosine_similarity(s_flat, t_flat, dim=-1)
        # Loss = 1 - mean(cos_sim)
        cosine_loss = 1.0 - cos_sim.mean()
        
        # 2. Moment Matching (Distribution Alignment)
        # Global Mean & Std across batch & sequence (or just feature dim?)
        # We want the *feature vector distribution* to match.
        # So we align mean/std along dim=-1 (embedding dim)
        
        s_mean = student_features.mean(dim=-1) # [B, N]
        s_std = student_features.std(dim=-1)   # [B, N]
        
        t_mean = teacher_features.mean(dim=-1) # [B, N]
        t_std = teacher_features.std(dim=-1)   # [B, N]
        
        mean_loss = F.mse_loss(s_mean, t_mean)
        std_loss = F.mse_loss(s_std, t_std)
        
        stats_loss = mean_loss + std_loss
        
        # Total Loss
        total_loss = (self.contrastive_weight * cosine_loss) + (self.moment_weight * stats_loss)
        
        return total_loss, {
            "cos_loss": cosine_loss.item(),
            "stats_loss": stats_loss.item(),
            "s_mean": s_mean.mean().item(),
            "t_mean": t_mean.mean().item()
        }

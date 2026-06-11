import asyncio
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from core.scorers.aesthetic_scorer import get_scorer as _get_internal_scorer, HybridAestheticScorer
import logging

logger = logging.getLogger("ScorerService")

# Re-export get_scorer
get_scorer = _get_internal_scorer

async def score_images_batch(scorer: HybridAestheticScorer, image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Async wrapper for batch scoring images.
    Returns a list of dicts: {"path": str, "score": float}
    """
    def _process():
        results = []
        valid_images = []
        valid_paths = []
        
        for p in image_paths:
            try:
                # Open image and ensure RGB
                img = Image.open(p).convert("RGB")
                valid_images.append(img)
                valid_paths.append(p)
            except Exception as e:
                logger.error(f"Failed to load image {p}: {e}")
                
        if not valid_images:
            return []
            
        # Run batch scoring
        scores = scorer.score_batch(valid_images)
        
        for p, s in zip(valid_paths, scores):
            results.append({
                "path": p,
                "score": s.percentile,
                "tier": s.tier.value,
                "raw": s.raw_score
            })
            
        return results

    return await asyncio.to_thread(_process)

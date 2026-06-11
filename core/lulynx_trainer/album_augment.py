# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Albumentations Pipeline — optional image augmentation for training.

Wraps an albumentations.Compose pipeline built from a JSON config.
Falls back gracefully if albumentations is not installed.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ALBUMENTATIONS_AVAILABLE: Optional[bool] = None


def available() -> bool:
    """Check if albumentations is importable."""
    global _ALBUMENTATIONS_AVAILABLE
    if _ALBUMENTATIONS_AVAILABLE is None:
        try:
            import albumentations  # noqa: F401
            _ALBUMENTATIONS_AVAILABLE = True
        except ImportError:
            _ALBUMENTATIONS_AVAILABLE = False
    return _ALBUMENTATIONS_AVAILABLE


_SUPPORTED_TRANSFORMS = {
    "GaussianBlur", "GaussNoise", "RandomBrightnessContrast",
    "HueSaturationValue", "Sharpen", "CoarseDropout",
    "Rotate", "Affine", "Perspective", "ElasticTransform",
    "RandomGamma", "CLAHE", "Blur", "MedianBlur",
    "ChannelShuffle", "RGBShift", "Normalize",
    "ColorJitter", "Downscale", "ImageCompression",
}


class AlbumentationsPipeline:
    """Configurable albumentations augmentation pipeline."""

    def __init__(self, pipeline_json: str, mask_replay: bool = True) -> None:
        if not available():
            raise ImportError(
                "albumentations is required for augmentation pipeline. "
                "Install with: pip install albumentations"
            )
        import albumentations as A

        self._mask_replay = mask_replay
        transforms = self._parse_pipeline(pipeline_json, A)
        if not transforms:
            self._pipeline = None
            return
        self._pipeline = A.Compose(transforms)

    @staticmethod
    def _parse_pipeline(pipeline_json: str, A: Any) -> List[Any]:
        """Convert JSON config to albumentations transform list."""
        if not pipeline_json or not pipeline_json.strip():
            return []
        try:
            raw = json.loads(pipeline_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Invalid albumentations_pipeline JSON, skipping augmentation")
            return []
        if not isinstance(raw, list):
            return []

        transforms = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            params = item.get("params", {})
            if name not in _SUPPORTED_TRANSFORMS:
                logger.warning("Unsupported albumentations transform: %s, skipping", name)
                continue
            cls = getattr(A, name, None)
            if cls is None:
                logger.warning("albumentations.%s not found, skipping", name)
                continue
            try:
                transforms.append(cls(**params))
            except Exception as e:
                logger.warning("Failed to create %s(%s): %s, skipping", name, params, e)
                continue
        return transforms

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation pipeline to image and optional mask."""
        if self._pipeline is None:
            return image, mask

        if mask is not None and self._mask_replay:
            result = self._pipeline(image=image, mask=mask)
            return result["image"], result["mask"]
        else:
            result = self._pipeline(image=image)
            return result["image"], mask

    @property
    def enabled(self) -> bool:
        return self._pipeline is not None

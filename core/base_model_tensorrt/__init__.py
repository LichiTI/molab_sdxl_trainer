"""Base-model TensorRT spike helpers.

These helpers are experiment planning/probing utilities only.  They keep family
knowledge out of the launcher and provide a reusable boundary for Anima, Newbie,
and future DiT/UNet families.
"""

from .probe import probe_base_model_tensorrt

__all__ = ["probe_base_model_tensorrt"]

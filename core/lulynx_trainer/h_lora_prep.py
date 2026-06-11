"""Legacy wrapper for Concept Geometry metadata preparation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .concept_geometry_prep import build_concept_geometry, main
except ImportError:  # pragma: no cover - direct script loading
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from concept_geometry_prep import build_concept_geometry, main


build_h_lora_geometry = build_concept_geometry


if __name__ == "__main__":
    raise SystemExit(main())

"""Legacy wrapper for the Concept Geometry v2 disentanglement smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .anima_concept_geometry_v2_disentangle_smoke import main
except ImportError:  # pragma: no cover - direct script loading
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anima_concept_geometry_v2_disentangle_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())

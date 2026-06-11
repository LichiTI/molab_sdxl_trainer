"""Legacy wrapper for the Anima Concept Geometry benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .anima_concept_geometry_benchmark import main
except ImportError:  # pragma: no cover - direct script loading
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from anima_concept_geometry_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

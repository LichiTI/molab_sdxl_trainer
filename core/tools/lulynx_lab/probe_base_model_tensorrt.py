"""CLI wrapper for base-model TensorRT spike probing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _prepare_path() -> None:
    backend_root = Path(__file__).resolve().parents[3]
    project_root = backend_root.parent
    for item in (str(project_root), str(backend_root)):
        if item not in sys.path:
            sys.path.insert(0, item)


_prepare_path()

from core.base_model_tensorrt import probe_base_model_tensorrt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe base-model TensorRT experiment readiness.")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-root", default="")
    parser.add_argument("--model-family", default="anima")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    result = probe_base_model_tensorrt(
        model_path=args.model_path,
        model_family=args.model_family,
        model_root=args.model_root,
        output_dir=args.output_dir,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

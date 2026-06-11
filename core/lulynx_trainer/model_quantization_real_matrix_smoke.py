"""Bounded real-weight smoke for the Launcher model quantization contract.

The default run scans ``models/`` for small real safetensors files, quantizes
the first candidate through the five main image-model formats, and writes a
machine-readable report under ``.runs/``.  It intentionally stays on the
Launcher toolbox quantization path and does not touch training dispatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.tools.model_quantizer import quantize_model_file  # noqa: E402


MAIN_FORMATS = [
    "fp16",
    "bf16",
    "fp8_e4m3fn",
    "lulynx_int8_rowwise",
    "lulynx_uint4_rowwise",
]


def _default_report_path() -> Path:
    return Path(
        os.environ.get(
            "LULYNX_MODEL_QUANT_REAL_MATRIX_REPORT",
            str(ROOT / ".runs" / "model_quantization_real_matrix" / "real_matrix.json"),
        )
    )


def _default_output_dir() -> Path:
    return Path(
        os.environ.get(
            "LULYNX_MODEL_QUANT_REAL_MATRIX_OUTPUT_DIR",
            str(ROOT / ".runs" / "model_quantization_real_matrix" / "outputs"),
        )
    )


def _discover_candidates(models_dir: Path, *, max_mb: float, max_candidates: int) -> list[Path]:
    if not models_dir.is_dir():
        return []
    max_bytes = int(max_mb * 1024 * 1024)
    candidates = [
        path
        for path in models_dir.rglob("*.safetensors")
        if path.is_file() and path.stat().st_size <= max_bytes
    ]
    candidates.sort(key=lambda item: (item.stat().st_size, str(item).lower()))
    return candidates[: max(1, int(max_candidates))]


def run_real_matrix(
    *,
    models_dir: Path,
    output_dir: Path,
    max_mb: float,
    max_candidates: int,
    formats: list[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _discover_candidates(models_dir, max_mb=max_mb, max_candidates=max_candidates)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "model_quantization_real_matrix_v1",
        "started_at": started,
        "models_dir": str(models_dir),
        "output_dir": str(output_dir),
        "max_mb": max_mb,
        "candidate_count": len(candidates),
        "formats": list(formats),
        "results": [],
    }
    passed = 0
    failed = 0
    skipped = 0
    run_tag = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    for candidate in candidates:
        candidate_result: dict[str, Any] = {
            "path": str(candidate),
            "relative_path": _relative(candidate, ROOT),
            "size_bytes": candidate.stat().st_size,
            "size_mb": round(candidate.stat().st_size / 1024 / 1024, 3),
            "formats": [],
        }
        for fmt in formats:
            output_path = output_dir / f"{candidate.stem}.{run_tag}.{fmt}.safetensors"
            started_at = time.perf_counter()
            try:
                result = quantize_model_file(
                    str(candidate),
                    str(output_path),
                    fmt,
                    decode_dtype="fp16",
                    overwrite=False,
                )
                elapsed = time.perf_counter() - started_at
                validation = result.get("validation") if isinstance(result, dict) else None
                ok = bool(isinstance(validation, dict) and validation.get("ok") is True)
                if ok:
                    passed += 1
                else:
                    failed += 1
                candidate_result["formats"].append(
                    {
                        "format": fmt,
                        "status": "passed" if ok else "failed",
                        "elapsed_sec": round(elapsed, 3),
                        "output_path": str(output_path),
                        "output_size_bytes": int(result.get("output_size_bytes") or 0),
                        "compression_ratio": result.get("compression_ratio"),
                        "converted_tensors": int(result.get("converted_tensors") or 0),
                        "skipped_tensors": int(result.get("skipped_tensors") or 0),
                        "validation_ok": ok,
                        "failed_checks": list(validation.get("failed_checks") or [])[:16] if isinstance(validation, dict) else ["missing_validation_report"],
                        "metadata_format": str(validation.get("metadata_format") or "") if isinstance(validation, dict) else "",
                        "quantized_entry_count": int(validation.get("quantized_entry_count") or 0) if isinstance(validation, dict) else 0,
                        "trainer_loader_compatible": bool(validation.get("trainer_loader_compatible")) if isinstance(validation, dict) else False,
                    }
                )
            except RuntimeError as exc:
                if fmt == "fp8_e4m3fn" and "float8_e4m3fn" in str(exc):
                    skipped += 1
                    candidate_result["formats"].append(
                        {
                            "format": fmt,
                            "status": "skipped",
                            "reason": str(exc),
                            "elapsed_sec": round(time.perf_counter() - started_at, 3),
                        }
                    )
                    continue
                failed += 1
                candidate_result["formats"].append(_failure_record(fmt, exc, started_at))
            except Exception as exc:  # noqa: BLE001 - smoke report should preserve failure type.
                failed += 1
                candidate_result["formats"].append(_failure_record(fmt, exc, started_at))
        report["results"].append(candidate_result)
    report["summary"] = {
        "candidate_count": len(candidates),
        "format_attempt_count": passed + failed,
        "format_passed_count": passed,
        "format_failed_count": failed,
        "format_skipped_count": skipped,
        "ok": bool(candidates) and failed == 0 and passed > 0,
    }
    return report


def _failure_record(fmt: str, exc: BaseException, started_at: float) -> dict[str, Any]:
    return {
        "format": fmt,
        "status": "failed",
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a bounded real-weight model quantization matrix.")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--report", default=str(_default_report_path()))
    parser.add_argument("--output-dir", default=str(_default_output_dir()))
    parser.add_argument("--max-mb", type=float, default=float(os.environ.get("LULYNX_MODEL_QUANT_REAL_MATRIX_MAX_MB", "256")))
    parser.add_argument("--max-candidates", type=int, default=int(os.environ.get("LULYNX_MODEL_QUANT_REAL_MATRIX_MAX_CANDIDATES", "1")))
    parser.add_argument("--formats", default=",".join(MAIN_FORMATS))
    args = parser.parse_args(argv)

    formats = [item.strip() for item in str(args.formats).split(",") if item.strip()]
    report = run_real_matrix(
        models_dir=Path(args.models_dir),
        output_dir=Path(args.output_dir),
        max_mb=float(args.max_mb),
        max_candidates=int(args.max_candidates),
        formats=formats,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if bool(report["summary"].get("ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())

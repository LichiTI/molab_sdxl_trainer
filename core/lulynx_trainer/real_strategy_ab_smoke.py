# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Short real-model A/B runner for selective checkpointing and GaLore/SVD.

This wrapper shells into real_model_training_smoke.py so both experiment and
baseline use the normal LulynxTrainer.start() path. It hard-caps every case to
40 training steps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _python_exe(repo_root: Path) -> Path:
    candidate = repo_root / "backend" / "env" / "python-flashattention" / "python.exe"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "json_error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _run_case(*, python_exe: Path, script: Path, case_name: str, args: argparse.Namespace, extra: List[str], report_dir: Path, source_data_override: Optional[Path] = None) -> Dict[str, Any]:
    case_report = report_dir / f"{case_name}.json"
    cmd = [
        str(python_exe),
        str(script),
        "--family",
        args.family,
        "--adapter",
        args.adapter,
        "--steps",
        str(max(1, min(int(args.steps), 40))),
        "--allow-short-steps",
        "--sample-limit",
        str(max(1, int(args.sample_limit))),
        "--rank",
        str(max(1, int(args.rank))),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--output-root",
        str(report_dir / "runs" / case_name),
        "--json",
        str(case_report),
        "--stop-on-failure",
        "--disable-vram-auto-enhance",
        "--cuda-cache-release-strategy",
        "off",
    ]
    source_data = str(source_data_override) if source_data_override is not None else args.source_data
    if source_data:
        cmd.extend(["--source-data", source_data])
    if args.resolution:
        cmd.extend(["--resolution", str(int(args.resolution))])
    cmd.extend(extra)
    started = time.perf_counter()
    completed = subprocess.run(cmd, cwd=str(_repo_root()), text=True, capture_output=True)
    payload = _load_json(case_report) if case_report.exists() else {"ok": False, "missing_report": str(case_report)}
    return {
        "case": case_name,
        "ok": completed.returncode == 0 and bool(payload.get("ok", False)),
        "returncode": completed.returncode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "source_data": source_data,
        "source_data_reused": source_data_override is not None,
        "command": cmd,
        "report": str(case_report),
        "stdout_tail": completed.stdout.splitlines()[-40:],
        "stderr_tail": completed.stderr.splitlines()[-40:],
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("selective", "galore"), required=True)
    parser.add_argument("--family", default="anima", choices=("anima", "newbie", "sdxl"))
    parser.add_argument("--adapter", default="lora")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=2)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--source-data", default="")
    parser.add_argument(
        "--reuse-newbie-cache",
        action="store_true",
        default=True,
        help="Reuse the first Newbie case train_dir as source-data for later cases.",
    )
    parser.add_argument(
        "--no-reuse-newbie-cache",
        dest="reuse_newbie_cache",
        action="store_false",
        help="Force every Newbie case to rebuild/cache from the original source data.",
    )
    parser.add_argument("--svd-rank", type=int, default=4)
    parser.add_argument("--svd-update-interval", type=int, default=1)
    parser.add_argument("--out", default="temp/real_strategy_ab_smoke.json")
    args = parser.parse_args()

    repo_root = _repo_root()
    python_exe = _python_exe(repo_root)
    script = repo_root / "backend" / "core" / "lulynx_trainer" / "real_model_training_smoke.py"
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_dir = output.parent / output.stem
    report_dir.mkdir(parents=True, exist_ok=True)

    cases: List[tuple[str, List[str]]]
    if args.experiment == "selective":
        cases = [
            ("baseline_full", ["--checkpoint-policy", "full"]),
            ("selective", ["--checkpoint-policy", "selective"]),
        ]
    else:
        cases = [
            ("baseline_auto", ["--advanced-optimizer-strategy", "off"]),
            (
                "galore_svd",
                [
                    "--advanced-optimizer-strategy",
                    "galore",
                    "--svd-grad-proj-rank",
                    str(max(1, int(args.svd_rank))),
                    "--svd-grad-proj-update-interval",
                    str(max(1, int(args.svd_update_interval))),
                ],
            ),
        ]

    started = time.perf_counter()
    results = []
    reusable_newbie_source: Optional[Path] = None
    for name, extra in cases:
        result = _run_case(
            python_exe=python_exe,
            script=script,
            case_name=name,
            args=args,
            extra=extra,
            report_dir=report_dir,
            source_data_override=reusable_newbie_source,
        )
        results.append(result)
        if (
            args.family == "newbie"
            and bool(args.reuse_newbie_cache)
            and reusable_newbie_source is None
            and result.get("ok")
        ):
            payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
            first_case = (payload.get("results") or [{}])[0] if isinstance(payload.get("results"), list) else {}
            train_dir = first_case.get("train_dir") if isinstance(first_case, dict) else ""
            if train_dir:
                reusable_newbie_source = Path(str(train_dir))
    payload = {
        "probe": "real_strategy_ab_smoke",
        "experiment": args.experiment,
        "family": args.family,
        "adapter": args.adapter,
        "steps": max(1, min(int(args.steps), 40)),
        "ok": all(item["ok"] for item in results),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "reuse_newbie_cache": bool(args.reuse_newbie_cache),
        "reusable_newbie_source": str(reusable_newbie_source) if reusable_newbie_source is not None else "",
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


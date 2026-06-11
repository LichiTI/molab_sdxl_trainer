# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Command-suite runner for repeatable training benchmark presets.

This module complements :mod:`benchmark_pipeline`: that file measures one
in-process step function, while this runner orchestrates existing smoke and
benchmark commands, captures environment metadata, and writes one comparable
JSON report per suite run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROGRESS_PREFIX = "lulynx_benchmark_progress:"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _prepend_pythonpath(env: dict[str, str], *paths: Path) -> None:
    parts = [str(path) for path in paths if path]
    current = env.get("PYTHONPATH", "")
    if current:
        parts.append(current)
    env["PYTHONPATH"] = os.pathsep.join(parts)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "read_error": f"{type(exc).__name__}: {exc}"}


def _parse_json_text(text: str) -> dict[str, Any] | list[Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        if isinstance(payload, (dict, list)):
            return payload
    except Exception:
        pass
    start_candidates = [idx for idx in (raw.find("{"), raw.find("[")) if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    for end in range(len(raw), start, -1):
        candidate = raw[start:end].strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
            if isinstance(payload, (dict, list)):
                return payload
        except Exception:
            continue
    return None


def _tail(text: str, max_lines: int) -> list[str]:
    if max_lines <= 0:
        return []
    lines = text.splitlines()
    return lines[-max_lines:]


def _emit_progress(enabled: bool, event: str, **payload: Any) -> None:
    if not enabled:
        return
    print(
        PROGRESS_PREFIX + json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _extract_losses(log_tail: Iterable[Any]) -> list[float]:
    losses: list[float] = []
    for raw in log_tail or []:
        text = str(raw)
        marker = "Loss:"
        if marker not in text:
            continue
        try:
            losses.append(float(text.split(marker, 1)[1].strip().split()[0]))
        except Exception:
            pass
    return losses


def _python_imports_torch(python_path: Path) -> bool:
    if not python_path.exists():
        return False
    try:
        completed = subprocess.run(
            [str(python_path), "-c", "import torch"],
            text=True,
            capture_output=True,
            timeout=20,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _detect_project_python(root: Path) -> str:
    candidates = (
        root / "backend" / "env" / "python-flashattention" / "python.exe",
        root / "backend" / "env" / "python" / "python.exe",
        root / "backend" / "env" / "python_launcher" / "python.exe",
    )
    for candidate in candidates:
        if _python_imports_torch(candidate):
            return str(candidate)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def _torch_environment() -> dict[str, Any]:
    report: dict[str, Any] = {"available": False}
    try:
        import torch

        report.update(
            {
                "available": True,
                "version": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            report.update(
                {
                    "device_index": int(idx),
                    "device_name": str(props.name),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_vram_mb": round(total_bytes / (1024 * 1024), 2),
                    "free_vram_mb": round(free_bytes / (1024 * 1024), 2),
                }
            )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _torch_environment_for_python(python_path: str) -> dict[str, Any]:
    code = r"""
import json
report = {"available": False}
try:
    import torch
    report.update({
        "available": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
    })
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        report.update({
            "device_index": int(idx),
            "device_name": str(props.name),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mb": round(total_bytes / (1024 * 1024), 2),
            "free_vram_mb": round(free_bytes / (1024 * 1024), 2),
        })
except Exception as exc:
    report["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(report, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [python_path, "-c", code],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode == 0:
            return json.loads(completed.stdout)
        return {
            "available": False,
            "error": (completed.stderr or completed.stdout or "torch probe failed").strip(),
        }
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _system_environment(root: Path, project_python: str) -> dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.executable,
        "project_python": project_python,
        "cwd": str(root),
        "torch": _torch_environment_for_python(project_python),
        "runner_torch": _torch_environment(),
    }


def _summarize_payload(payload: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(payload, list):
        rows: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            row = {
                key: item.get(key)
                for key in (
                    "profile",
                    "effective",
                    "enabled",
                    "change_count",
                    "checkpoint_policy",
                    "te_vae_offload_strategy",
                    "swap_granularity",
                    "swap_ratio",
                    "train_batch_size",
                    "enable_mixed_resolution_training",
                    "advisor_patch_low_vram_profile",
                )
                if key in item
            }
            if row:
                rows.append(row)
        summary = {"payload_type": "list", "items_count": len(payload)}
        if rows:
            summary["rows"] = rows[:16]
        return summary
    summary: dict[str, Any] = {}
    for key in (
        "ok",
        "benchmark",
        "probe",
        "scope",
        "family",
        "experiment",
        "device",
        "cuda_available",
        "duration_seconds",
        "resolved_steps",
        "interpretation",
    ):
        if key in payload:
            summary[key] = payload.get(key)

    results = payload.get("results")
    if isinstance(results, list):
        summary["results_count"] = len(results)
        success_count = 0
        metrics: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            if bool(item.get("success", item.get("ok", False))):
                success_count += 1
            metric: dict[str, Any] = {}
            for key in (
                "label",
                "family",
                "adapter",
                "backend",
                "resolved_backend",
                "success",
                "ok",
                "artifact",
                "global_step",
                "resolved_steps",
                "duration_seconds",
                "forward_ms",
                "backward_ms",
                "mean_step_ms",
                "median_step_ms",
                "p95_step_ms",
                "peak_memory_mb",
                "peak_allocated_mb",
                "samples_per_second",
                "steps_per_second",
                "max_abs_diff_vs_sdpa",
            ):
                if key in item:
                    metric[key] = item.get(key)
            losses = _extract_losses(item.get("log_tail") or [])
            if losses:
                metric["loss_first"] = losses[0]
                metric["loss_last"] = losses[-1]
                metric["loss_count"] = len(losses)
            if metric:
                metrics.append(metric)
        summary["successful_results"] = success_count
        if metrics:
            summary["metrics"] = metrics[:12]

    shape = payload.get("shape")
    if isinstance(shape, dict):
        summary["shape"] = shape
    benchmark = payload.get("benchmark")
    if isinstance(benchmark, dict):
        summary["benchmark"] = {key: benchmark.get(key) for key in sorted(benchmark.keys())[:24]}
    runs = payload.get("runs")
    if isinstance(runs, dict):
        run_metrics: dict[str, Any] = {}
        for name, value in runs.items():
            if not isinstance(value, dict):
                continue
            run_metrics[str(name)] = {
                key: value.get(key)
                for key in (
                    "success",
                    "steps_completed",
                    "total_wall_seconds",
                    "mean_step_ms",
                    "steady_mean_step_ms",
                    "samples_per_second",
                    "steady_samples_per_second",
                    "peak_vram_mb",
                    "final_loss",
                    "output_size_bytes",
                )
                if key in value
            }
        if run_metrics:
            summary["run_metrics"] = run_metrics
    strategies = payload.get("strategies")
    if isinstance(strategies, dict):
        strategy_metrics: dict[str, Any] = {}
        for name, value in strategies.items():
            if not isinstance(value, dict):
                continue
            strategy_metrics[str(name)] = {
                key: value.get(key)
                for key in (
                    "success",
                    "resolved_strategy",
                    "mean_ms",
                    "median_ms",
                    "peak_allocated_mb",
                    "failed_reason",
                )
                if key in value
            }
        if strategy_metrics:
            summary["strategy_metrics"] = strategy_metrics
    comparison = payload.get("comparison")
    if isinstance(comparison, dict):
        summary["comparison"] = comparison
    return summary


@dataclass
class SuiteCase:
    id: str
    command: list[str]
    title: str = ""
    category: str = ""
    cwd: str = "{repo}"
    env: dict[str, str] = field(default_factory=dict)
    output_json: str = ""
    timeout_seconds: int = 900
    expected_exit_codes: list[int] = field(default_factory=lambda: [0])
    required: bool = True
    tags: list[str] = field(default_factory=list)
    run_by_default: bool = True
    capture_stdout_json: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SuiteCase":
        command = payload.get("command")
        if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
            raise ValueError("case.command must be a list of strings")
        case_id = str(payload.get("id") or "").strip()
        if not case_id:
            raise ValueError("case.id is required")
        return cls(
            id=case_id,
            title=str(payload.get("title") or ""),
            category=str(payload.get("category") or ""),
            command=list(command),
            cwd=str(payload.get("cwd") or "{repo}"),
            env={str(k): str(v) for k, v in dict(payload.get("env") or {}).items()},
            output_json=str(payload.get("output_json") or ""),
            timeout_seconds=int(payload.get("timeout_seconds") or 900),
            expected_exit_codes=[int(v) for v in payload.get("expected_exit_codes", [0])],
            required=bool(payload.get("required", True)),
            tags=[str(item) for item in payload.get("tags", [])],
            run_by_default=bool(payload.get("run_by_default", True)),
            capture_stdout_json=bool(payload.get("capture_stdout_json", False)),
        )


@dataclass
class SuitePreset:
    suite_id: str
    description: str
    out_dir: str
    cases: list[SuiteCase]
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: Path) -> "SuitePreset":
        payload = _read_json(path)
        suite_id = str(payload.get("suite_id") or path.stem).strip()
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list):
            raise ValueError("preset.cases must be a list")
        return cls(
            suite_id=suite_id,
            description=str(payload.get("description") or ""),
            out_dir=str(payload.get("out_dir") or f"temp/benchmark_suite/{suite_id}"),
            cases=[SuiteCase.from_dict(item) for item in raw_cases if isinstance(item, dict)],
            notes=[str(item) for item in payload.get("notes", [])],
        )


class BenchmarkSuiteRunner:
    def __init__(self, preset: SuitePreset, *, root: Path | None = None) -> None:
        self.root = (root or _repo_root()).resolve()
        self.preset = preset
        self.project_python = _detect_project_python(self.root)
        self.out_dir = self._resolve_path(preset.out_dir, {"suite_id": preset.suite_id})

    def _context(self, case: SuiteCase | None = None) -> dict[str, str]:
        case_id = case.id if case else ""
        case_out = self.out_dir / f"{case_id}.json" if case_id else self.out_dir
        return {
            "repo": str(self.root),
            "python": sys.executable,
            "project_python": self.project_python,
            "out_dir": str(self.out_dir),
            "suite_id": self.preset.suite_id,
            "case_id": case_id,
            "case_out": str(case_out),
        }

    def _format(self, value: str, context: dict[str, str]) -> str:
        result = value
        for key, replacement in context.items():
            result = result.replace("{" + key + "}", replacement)
        return result

    def _resolve_path(self, value: str, context: dict[str, str]) -> Path:
        raw = self._format(value, context)
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        return path

    def _select_cases(
        self,
        case_ids: Iterable[str],
        *,
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
    ) -> list[SuiteCase]:
        selected = {str(item) for item in case_ids if str(item).strip()}
        included = {str(item) for item in include_tags if str(item).strip()}
        excluded = {str(item) for item in exclude_tags if str(item).strip()}
        if not selected:
            cases = [case for case in self.preset.cases if case.run_by_default or (set(case.tags) & included)]
        else:
            cases = [case for case in self.preset.cases if case.id in selected]
        if included and selected:
            cases = [case for case in cases if case.run_by_default or (set(case.tags) & included)]
        if excluded:
            cases = [case for case in cases if not (set(case.tags) & excluded)]
        return cases

    def _skipped_cases(
        self,
        selected_cases: list[SuiteCase],
        *,
        case_ids: Iterable[str],
        include_tags: Iterable[str],
        exclude_tags: Iterable[str],
    ) -> list[dict[str, Any]]:
        selected_ids = {case.id for case in selected_cases}
        explicit_ids = {str(item) for item in case_ids if str(item).strip()}
        included = {str(item) for item in include_tags if str(item).strip()}
        excluded = {str(item) for item in exclude_tags if str(item).strip()}
        skipped: list[dict[str, Any]] = []
        for case in self.preset.cases:
            if case.id in selected_ids:
                continue
            reason = "not_selected"
            if excluded and (set(case.tags) & excluded):
                reason = "excluded_tag"
            elif not explicit_ids and not case.run_by_default and not (set(case.tags) & included):
                reason = "gated_by_tag"
            skipped.append(
                {
                    "id": case.id,
                    "title": case.title,
                    "category": case.category,
                    "tags": list(case.tags),
                    "run_by_default": case.run_by_default,
                    "skip_reason": reason,
                }
            )
        return skipped

    def run(
        self,
        *,
        case_ids: Iterable[str] = (),
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        dry_run: bool = False,
        fail_fast: bool = False,
        tail_lines: int = 40,
        progress_json: bool = False,
    ) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cases = self._select_cases(case_ids, include_tags=include_tags, exclude_tags=exclude_tags)
        skipped_cases = self._skipped_cases(
            cases,
            case_ids=case_ids,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "suite_id": self.preset.suite_id,
            "description": self.preset.description,
            "dry_run": bool(dry_run),
            "include_tags": [str(item) for item in include_tags],
            "exclude_tags": [str(item) for item in exclude_tags],
            "environment": _system_environment(self.root, self.project_python),
            "notes": list(self.preset.notes),
            "cases": [],
            "skipped_cases": skipped_cases,
        }
        total = len(cases)
        _emit_progress(
            progress_json,
            "suite_start",
            suite_id=self.preset.suite_id,
            total=total,
            skipped=len(skipped_cases),
            description=self.preset.description,
        )
        for index, case in enumerate(cases, start=1):
            _emit_progress(
                progress_json,
                "case_start",
                suite_id=self.preset.suite_id,
                index=index,
                total=total,
                case_id=case.id,
                title=case.title,
                category=case.category,
                tags=list(case.tags),
                timeout_seconds=case.timeout_seconds,
            )
            item = self._run_case(case, dry_run=dry_run, tail_lines=tail_lines)
            report["cases"].append(item)
            _emit_progress(
                progress_json,
                "case_done",
                suite_id=self.preset.suite_id,
                index=index,
                total=total,
                case_id=case.id,
                title=case.title,
                category=case.category,
                ok=bool(item.get("ok", False)),
                duration_seconds=float(item.get("duration_seconds") or 0.0),
                returncode=item.get("returncode"),
                timeout=bool(item.get("timeout", False)),
            )
            if fail_fast and case.required and not bool(item.get("ok", False)):
                break
        report["ok"] = all(
            bool(item.get("ok", False)) or not bool(item.get("required", True))
            for item in report["cases"]
        )
        report["case_count"] = len(report["cases"])
        _emit_progress(
            progress_json,
            "suite_done",
            suite_id=self.preset.suite_id,
            total=total,
            completed=len(report["cases"]),
            skipped=len(skipped_cases),
            ok=bool(report["ok"]),
        )
        return report

    def _run_case(self, case: SuiteCase, *, dry_run: bool, tail_lines: int) -> dict[str, Any]:
        context = self._context(case)
        command = [self._format(arg, context) for arg in case.command]
        cwd = self._resolve_path(case.cwd, context)
        output_json = self._resolve_path(case.output_json, context) if case.output_json else None
        env = os.environ.copy()
        _prepend_pythonpath(env, self.root, self.root / "backend")
        for key, value in case.env.items():
            env[key] = self._format(value, context)
        item: dict[str, Any] = {
            "id": case.id,
            "title": case.title,
            "category": case.category,
            "tags": list(case.tags),
            "run_by_default": case.run_by_default,
            "required": case.required,
            "command": command,
            "cwd": str(cwd),
            "output_json": str(output_json) if output_json else "",
            "expected_exit_codes": list(case.expected_exit_codes),
            "timeout_seconds": case.timeout_seconds,
            "capture_stdout_json": case.capture_stdout_json,
        }
        if dry_run:
            item.update({"ok": True, "skipped": True, "skip_reason": "dry_run"})
            return item

        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=max(int(case.timeout_seconds), 1),
            )
            duration = time.perf_counter() - start
            ok = int(completed.returncode) in set(case.expected_exit_codes)
            item.update(
                {
                    "ok": bool(ok),
                    "returncode": int(completed.returncode),
                    "duration_seconds": round(duration, 3),
                    "stdout_tail": _tail(completed.stdout or "", tail_lines),
                    "stderr_tail": _tail(completed.stderr or "", tail_lines),
                }
            )
            if case.capture_stdout_json and output_json and not output_json.exists():
                parsed = _parse_json_text(completed.stdout or "")
                if parsed is not None:
                    output_json.parent.mkdir(parents=True, exist_ok=True)
                    output_json.write_text(json.dumps(parsed, indent=2, sort_keys=True), encoding="utf-8")
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            item.update(
                {
                    "ok": False,
                    "timeout": True,
                    "duration_seconds": round(duration, 3),
                    "stdout_tail": _tail(str(exc.stdout or ""), tail_lines),
                    "stderr_tail": _tail(str(exc.stderr or ""), tail_lines),
                }
            )

        if output_json and output_json.exists():
            payload = _read_json(output_json)
            item["json_summary"] = _summarize_payload(payload)
        elif output_json:
            item["json_summary"] = {"ok": False, "missing": True}
            if bool(item.get("ok", False)):
                item["ok"] = False
        return item


def _default_preset(root: Path) -> Path:
    return root / "devtools" / "benchmark_presets" / "quick_training_benchmark.json"


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run a repeatable benchmark preset suite.")
    parser.add_argument("--repo-root", default="", help="Repository root used to resolve preset paths and child commands.")
    parser.add_argument("--preset", default=str(_default_preset(root)))
    parser.add_argument("--out", default="")
    parser.add_argument("--case", action="append", default=[], help="Run only this case id; repeatable.")
    parser.add_argument("--include-tag", action="append", default=[], help="Include gated cases with this tag; repeatable.")
    parser.add_argument("--exclude-tag", action="append", default=[], help="Exclude cases with this tag; repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--tail-lines", type=int, default=40)
    parser.add_argument("--progress-json", action="store_true", help="Emit structured progress lines for launcher UI consumption.")
    args = parser.parse_args(argv)

    if args.repo_root:
        root = Path(args.repo_root).resolve()

    preset_path = Path(args.preset)
    if not preset_path.is_absolute():
        preset_path = root / preset_path
    preset = SuitePreset.from_path(preset_path)
    runner = BenchmarkSuiteRunner(preset, root=root)
    report = runner.run(
        case_ids=args.case,
        include_tags=args.include_tag,
        exclude_tags=args.exclude_tag,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
        tail_lines=int(args.tail_lines),
        progress_json=bool(args.progress_json),
    )
    out = Path(args.out) if args.out else runner.out_dir / "suite_report.json"
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "suite_id": report["suite_id"], "report": str(out)}, sort_keys=True))
    return 0 if bool(report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

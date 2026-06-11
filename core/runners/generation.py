"""Generation runners for the request-native registry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from backend.core.contracts import (
    ArtifactFile,
    ArtifactManifest,
    GenerationRequest,
    GenerationResult,
    PlatformIssue,
    RunContext,
    RunStatus,
    RunnerRegistry,
)


class DryRunGenerationRunner:
    """Lightweight generation runner proving the registry execution path.

    This runner intentionally does not generate pixels. It validates the request
    boundary, checks declared paths against ``RunContext.safe_roots``, and
    returns a `GenerationResult` with a planned output artifact for dry-run and
    preview planning flows. Real Anima/Newbie/SDXL samplers can later replace or
    extend this runner behind the same schema id.
    """

    runner_id = "generation.dry-run"
    schema_ids = ("generation.image", "generation.preview")
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_models", "write_output"],
        "resources": ["model", "adapter", "vae", "output_dir"],
        "heavy_dependencies": [],
        "estimated_cost": "light",
        "metadata": {
            "real_generation": False,
            "artifact_kind": "image",
            "note": "Dry-run planner only; concrete samplers register behind the same request family later.",
        },
    }

    def run(self, request: Any, context: RunContext) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            request = GenerationRequest.model_validate(request)

        issues = self._validate_paths(request, context)
        if issues:
            return GenerationResult.failure(
                "Generation request failed path validation.",
                request_id=request.request_id,
                issues=issues,
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        if not request.dry_run:
            return GenerationResult.failure(
                "Real image generation is not connected to this runner yet.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="generation.real_runner_unavailable",
                        message="Use dry_run=true until a concrete sampler runner is registered.",
                        severity="error",
                        field="dry_run",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )

        artifact = self._planned_image_artifact(request, context)
        return GenerationResult(
            request_id=request.request_id,
            status=RunStatus.SUCCEEDED,
            message="Generation dry-run plan created.",
            images=[artifact],
            metrics={
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "batch_size": request.batch_size,
            },
            data={
                "runner_id": self.runner_id,
                "schema_id": request.schema_id,
                "arch": request.arch,
                "sampler": request.sampler,
                "seed": request.seed,
                "dry_run": True,
            },
        )

    def _validate_paths(self, request: GenerationRequest, context: RunContext) -> list[PlatformIssue]:
        issues: list[PlatformIssue] = []
        for field, value in request.required_paths().items():
            if field == "output_dir":
                if not context.is_safe_path(value):
                    issues.append(self._unsafe_path_issue(field, value))
                continue
            path = Path(value)
            if path.is_absolute() and not context.is_safe_path(path):
                issues.append(self._unsafe_path_issue(field, value))
        return issues

    @staticmethod
    def _unsafe_path_issue(field: str, value: str) -> PlatformIssue:
        return PlatformIssue(
            code="generation.path_outside_safe_roots",
            message=f"{field} is outside allowed safe roots: {value}",
            severity="error",
            field=field,
            hint="Use a path inside the project, models, data, or output roots configured for this run context.",
        )

    def _planned_image_artifact(self, request: GenerationRequest, context: RunContext) -> ArtifactManifest:
        output_dir = Path(request.output_dir or "output/generation")
        if not output_dir.is_absolute():
            output_dir = context.project_root / output_dir
        name = self._safe_output_name(request)
        output_path = output_dir / name
        return ArtifactManifest(
            artifact_kind="image",
            schema_id=request.schema_id,
            producer=self.runner_id,
            request_id=request.request_id,
            files=[
                ArtifactFile(
                    path=str(output_path),
                    role="planned-image",
                    media_type="image/png",
                    metadata={"dry_run": True},
                )
            ],
            metadata={
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "arch": request.arch,
                "sampler": request.sampler,
            },
        )

    @staticmethod
    def _safe_output_name(request: GenerationRequest) -> str:
        raw = str(request.output_name or "generation_preview.png").strip() or "generation_preview.png"
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
        if not safe.lower().endswith(".png"):
            safe += ".png"
        return safe.strip("._") or "generation_preview.png"


class SubprocessGenerationRunner:
    """Real generation runner that delegates heavy sampling to a subprocess.

    The registry can import this class without importing torch or diffusers.
    Heavy model code is loaded only by ``core.lulynx_trainer.anima_inference_cli``
    after a real request passes path validation and the explicit permission gate.
    """

    runner_id = "generation.subprocess"
    schema_ids = ("generation.image", "generation.preview")
    capability_metadata = {
        "supports_dry_run": True,
        "permissions": ["read_models", "write_output", "run_generation"],
        "resources": ["model", "adapter", "vae", "output_dir", "runtime"],
        "heavy_dependencies": ["torch", "diffusers", "transformers"],
        "estimated_cost": "heavy",
        "metadata": {
            "real_generation": True,
            "artifact_kind": "image",
            "execution_mode": "subprocess",
            "module": "core.lulynx_trainer.anima_inference_cli",
        },
    }

    def __init__(self) -> None:
        self._dry_runner = DryRunGenerationRunner()

    def run(self, request: Any, context: RunContext) -> GenerationResult:
        if not isinstance(request, GenerationRequest):
            request = GenerationRequest.model_validate(request)

        issues = self._dry_runner._validate_paths(request, context)
        if issues:
            return GenerationResult.failure(
                "Generation request failed path validation.",
                request_id=request.request_id,
                issues=issues,
                data={"runner_id": self.runner_id, "schema_id": request.schema_id},
            )
        if request.dry_run:
            return self._dry_runner.run(request, context)
        if "run_generation" not in set(context.metadata.get("permissions") or []):
            return GenerationResult.failure(
                "Generation execution requires run_generation permission.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="generation.permission_required",
                        message="run_generation permission is required for real image generation.",
                        severity="error",
                        field="permissions",
                    )
                ],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id, "required_permission": "run_generation"},
            )
        return self._run_subprocess_generation(request, context)

    def _run_subprocess_generation(self, request: GenerationRequest, context: RunContext) -> GenerationResult:
        backend_root = context.backend_root or context.project_root / "backend"
        run_dir = self._run_dir(request, context)
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "generation_request.json"
        result_path = run_dir / "generation_result.json"
        request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")

        command = [
            str(self._resolve_python(request, context, backend_root)),
            "-m",
            "core.lulynx_trainer.anima_inference_cli",
            "--request_json",
            str(request_path),
            "--run_result_json",
            str(result_path),
        ]
        completed = self._run_subprocess(command, backend_root, self._subprocess_env(context, backend_root))
        if completed.returncode != 0:
            output = _tail_text((completed.stdout or "") + "\n" + (completed.stderr or ""))
            return GenerationResult.failure(
                "Generation subprocess failed.",
                request_id=request.request_id,
                issues=[
                    PlatformIssue(
                        code="generation.subprocess_failed",
                        message=output or f"Generation subprocess exited with code {completed.returncode}",
                        severity="error",
                    )
                ],
                data={
                    "runner_id": self.runner_id,
                    "schema_id": request.schema_id,
                    "returncode": completed.returncode,
                    "stdout_tail": output,
                    "request_path": str(request_path),
                    "result_path": str(result_path),
                },
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            result = GenerationResult.model_validate(payload)
        except Exception as exc:
            return GenerationResult.failure(
                "Generation subprocess did not write a valid result envelope.",
                request_id=request.request_id,
                issues=[PlatformIssue(code="generation.result_parse_failed", message=str(exc), severity="error")],
                data={"runner_id": self.runner_id, "schema_id": request.schema_id, "result_path": str(result_path)},
            )
        result.data.setdefault("runner_id", self.runner_id)
        result.data.setdefault("schema_id", request.schema_id)
        result.data.setdefault("request_path", str(request_path))
        result.data.setdefault("result_path", str(result_path))
        return result

    def _run_dir(self, request: GenerationRequest, context: RunContext) -> Path:
        if context.work_dir is not None:
            return Path(context.work_dir)
        backend_root = context.backend_root or context.project_root / "backend"
        return backend_root / "data" / "runs" / request.request_id

    def _resolve_python(self, request: GenerationRequest, context: RunContext, backend_root: Path) -> Path:
        requested = str(
            getattr(request, "execution_profile_id", "")
            or (request.model_extra or {}).get("execution_profile_id", "")
            or context.runtime_id
            or ""
        ).strip()
        if requested:
            from backend.core.execution_manifest import get_profile_entry, resolve_python_executable

            entry = get_profile_entry(requested)
            return resolve_python_executable(entry, backend_root).resolve()
        standard = (backend_root / "env" / "python" / ("python.exe" if os.name == "nt" else "bin/python")).resolve()
        return standard if standard.is_file() else Path(sys.executable).resolve()

    def _subprocess_env(self, context: RunContext, backend_root: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(dict(context.env or {}))
        env["PYTHONPATH"] = os.pathsep.join([str(backend_root), str(context.project_root), env.get("PYTHONPATH", "")])
        return env

    def _run_subprocess(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.run(command, **popen_kwargs)


def _tail_text(text: str, *, line_count: int = 80) -> str:
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    return "\n".join(lines[-line_count:])


def create_generation_registry(*, enable_real: bool = False) -> RunnerRegistry:
    registry = RunnerRegistry()
    registry.register(SubprocessGenerationRunner() if enable_real else DryRunGenerationRunner())
    return registry

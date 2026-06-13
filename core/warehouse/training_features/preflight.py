"""Preflight Orchestrator — composes Warehouse utilities into a single
go/no-go check before training begins.

This module does not import legacy fork app code, old preflight code, or make any
assumptions about training-script paths.  It delegates to the Warehouse
subsystems already present in this package and aggregates their results
into a unified :class:`PreflightResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cache_preflight import CachePreflightReport, DatasetCachePreflight
from .dataset_inspector import DatasetInspector, DatasetReport
from .resource_check import ResourceChecker, ResourceReport
from .resume_guard import ResumeGuard, ResumeReport
from .route_contract import RouteContract
from .runtime_dependencies import DependencyReport, RuntimeDependencyChecker
from .trainer_registry import TrainerRegistry
from .types import MessageBag, ModelArchitecture, PreflightVerdict, Severity


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreflightConfig:
    """Declarative inputs for a preflight run.

    All fields are optional; omitted checks are skipped.
    """

    # Dataset
    data_dir: Path | None = None
    caption_extension: str = ".txt"

    # Cache
    architecture: ModelArchitecture | None = None
    expected_image_count: int | None = None

    # Resume
    checkpoint_dir: Path | None = None

    # Resources
    min_disk_free_gb: float = 0.0
    min_gpu_vram_mb: int = 0

    # Runtime deps
    required_packages: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)

    # Trainer / route
    trainer_key: str | None = None
    route_id: str | None = None
    route_params: dict[str, Any] = field(default_factory=dict)

    # Registries (optional external state)
    trainer_registry: TrainerRegistry | None = None
    route_contracts: Any = None  # RouteContractSet — typed as Any to avoid circular import


# ---------------------------------------------------------------------------
# Aggregated result
# ---------------------------------------------------------------------------

@dataclass
class PreflightResult:
    """Unified outcome of all preflight checks."""

    verdict: PreflightVerdict
    can_start: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # Sub-reports (None when the corresponding check was skipped)
    dataset: DatasetReport | None = None
    cache: CachePreflightReport | None = None
    resume: ResumeReport | None = None
    resources: ResourceReport | None = None
    dependencies: DependencyReport | None = None
    route_errors: list[str] = field(default_factory=list)

    # Convenience
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PreflightOrchestrator:
    """Runs a battery of preflight checks and returns a consolidated result.

    Stateless after construction.  Each :meth:`run` call is independent.

    Example::

        orch = PreflightOrchestrator()
        result = orch.run(PreflightConfig(
            data_dir=Path("/data/my_dataset"),
            architecture=ModelArchitecture.SDXL,
            checkpoint_dir=Path("/checkpoints/model"),
            required_packages=["torch", "diffusers"],
            min_disk_free_gb=50.0,
        ))
        if not result.can_start:
            for err in result.errors:
                print(f"  ERROR: {err}")
    """

    def __init__(
        self,
        *,
        dataset_inspector: DatasetInspector | None = None,
        dependency_checker: RuntimeDependencyChecker | None = None,
        resource_checker: ResourceChecker | None = None,
    ) -> None:
        self._inspector = dataset_inspector or DatasetInspector()
        self._dep_checker = dependency_checker or RuntimeDependencyChecker()
        self._res_checker = resource_checker or ResourceChecker()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, config: PreflightConfig) -> PreflightResult:
        """Execute all applicable preflight checks and return a result."""
        bag = MessageBag()
        summary: dict[str, Any] = {}

        # -- 1. Runtime dependencies ----------------------------------------
        dep_report: DependencyReport | None = None
        if config.required_packages or config.required_tools:
            dep_report = self._dep_checker.check(
                packages=config.required_packages or None,
                tools=config.required_tools or None,
            )
            bag.errors.extend(dep_report.errors)
            bag.warnings.extend(dep_report.warnings)
            bag.notes.extend(dep_report.notes)
            summary["dependencies"] = {
                "all_satisfied": dep_report.all_satisfied,
                "packages_checked": len(dep_report.packages),
                "tools_checked": len(dep_report.tools),
            }

        # -- 2. Resources ---------------------------------------------------
        res_report: ResourceReport | None = None
        if config.data_dir is not None or config.min_gpu_vram_mb > 0:
            res_report = self._res_checker.check(
                data_dir=config.data_dir,
                min_disk_free_gb=config.min_disk_free_gb,
                min_gpu_vram_mb=config.min_gpu_vram_mb,
            )
            bag.errors.extend(res_report.errors)
            bag.warnings.extend(res_report.warnings)
            bag.notes.extend(res_report.notes)
            summary["resources"] = {
                "disk_free_gb": res_report.disk.free_gb if res_report.disk else None,
                "gpu_count": len(res_report.gpus),
            }

        # -- 3. Dataset inspection ------------------------------------------
        ds_report: DatasetReport | None = None
        if config.data_dir is not None:
            ds_report = self._inspector.inspect(config.data_dir)
            bag.warnings.extend(ds_report.warnings)
            summary["dataset"] = {
                "image_count": ds_report.image_count,
                "effective_image_count": ds_report.effective_image_count,
                "caption_coverage": ds_report.caption_coverage,
                "folder_count": ds_report.folder_count,
            }

        # -- 4. Cache preflight ---------------------------------------------
        cache_report: CachePreflightReport | None = None
        if config.data_dir is not None and config.architecture is not None:
            cache_pf = DatasetCachePreflight(config.architecture)
            cache_report = cache_pf.analyze(
                config.data_dir,
                expected_count=config.expected_image_count,
            )
            bag.errors.extend(cache_report.errors)
            bag.warnings.extend(cache_report.warnings)
            bag.notes.extend(cache_report.notes)
            summary["cache"] = {
                "ready": cache_report.ready,
                "hit_rate": cache_report.cache_hit_rate,
                "missing": cache_report.missing_count,
            }

        # -- 5. Resume guard ------------------------------------------------
        resume_report: ResumeReport | None = None
        if config.checkpoint_dir is not None:
            guard = ResumeGuard()
            resume_report = guard.scan(config.checkpoint_dir)
            bag.errors.extend(resume_report.errors)
            bag.warnings.extend(resume_report.warnings)
            if resume_report.found and resume_report.latest:
                bag.notes.append(
                    f"Resume checkpoint found: step {resume_report.latest.step} "
                    f"({resume_report.latest.size_bytes} bytes)"
                )
            summary["resume"] = {
                "found": resume_report.found,
                "latest_step": resume_report.latest.step if resume_report.latest else None,
            }

        # -- 6. Trainer / route validation -----------------------------------
        route_errors: list[str] = []
        if config.trainer_key is not None and config.trainer_registry is not None:
            entry = config.trainer_registry.get(config.trainer_key)
            if entry is None:
                bag.add_error(f"Unknown trainer key: {config.trainer_key}")
            else:
                summary["trainer"] = {"key": entry.key, "display_name": entry.display_name}

        if config.route_id is not None and config.route_contracts is not None:
            contract = config.route_contracts.get(config.route_id)
            if contract is None:
                bag.add_error(f"Unknown route: {config.route_id}")
            elif config.route_params:
                route_errors = contract.validate_params(config.route_params)
                bag.errors.extend(route_errors)
            summary["route"] = {"route_id": config.route_id, "param_errors": len(route_errors)}

        # -- 7. Cross-checks ------------------------------------------------
        # Warn if dataset found but cache architecture missing
        if ds_report is not None and ds_report.image_count > 0 and config.architecture is None:
            bag.add_warning("Dataset found but architecture not specified — cache check skipped")

        # Warn if resume found but dataset missing
        if resume_report is not None and resume_report.found and ds_report is None:
            bag.add_note("Resume checkpoint found but no dataset directory specified")

        # -- 8. Verdict -----------------------------------------------------
        can_start = bag.is_clean
        if can_start:
            verdict = PreflightVerdict.GO
        elif bag.errors:
            verdict = PreflightVerdict.NO_GO
        else:
            # Only warnings — allow with caution
            verdict = PreflightVerdict.WARN
            can_start = True

        summary["verdict"] = verdict.value
        summary["error_count"] = len(bag.errors)
        summary["warning_count"] = len(bag.warnings)

        return PreflightResult(
            verdict=verdict,
            can_start=can_start,
            errors=list(bag.errors),
            warnings=list(bag.warnings),
            notes=list(bag.notes),
            dataset=ds_report,
            cache=cache_report,
            resume=resume_report,
            resources=res_report,
            dependencies=dep_report,
            route_errors=route_errors,
            summary=summary,
        )


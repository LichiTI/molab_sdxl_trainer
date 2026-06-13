"""Runtime Dependencies — verifies that Python packages and system tools
required for training are importable/executable.

Pure-stdlib.  No external dependencies.  Each check is independent and
results are collected into a structured :class:`DependencyReport`.

This module now also includes a **training-aware dependency inspector**
with a PACKAGE_REGISTRY mapping and per-config requirement collection
modeled on the fork's ``runtime_dependencies.py`` + ``runtime_dependency_rules.py``
behavioral surface.  No original code bodies were copied.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Iterable

from .types import MessageBag


# ---------------------------------------------------------------------------
# Package registry — maps module_name → pip package info
# ---------------------------------------------------------------------------

PACKAGE_REGISTRY: dict[str, dict[str, Any]] = {
    "accelerate":          {"package_name": "accelerate",            "display_name": "accelerate",           "required_by_default": True},
    "torch":               {"package_name": "torch",                 "display_name": "PyTorch",              "required_by_default": True},
    "fastapi":             {"package_name": "fastapi",               "display_name": "FastAPI",              "required_by_default": True},
    "toml":                {"package_name": "toml",                  "display_name": "toml",                 "required_by_default": True},
    "lion_pytorch":        {"package_name": "lion-pytorch",          "display_name": "lion-pytorch",         "required_by_default": True},
    "dadaptation":         {"package_name": "dadaptation",           "display_name": "dadaptation",          "required_by_default": False},
    "schedulefree":        {"package_name": "schedulefree",          "display_name": "schedulefree",         "required_by_default": True},
    "prodigyopt":          {"package_name": "prodigyopt",            "display_name": "prodigyopt",           "required_by_default": True},
    "prodigyplus":         {"package_name": "prodigy-plus-schedule-free", "display_name": "prodigyplus",     "required_by_default": True},
    "pytorch_optimizer":   {"package_name": "pytorch-optimizer",     "display_name": "pytorch-optimizer",    "required_by_default": True},
    "lycoris":             {"package_name": "lycoris-lora",          "display_name": "lycoris-lora",         "required_by_default": False},
    "safetensors":         {"package_name": "safetensors",           "display_name": "safetensors",          "required_by_default": True},
    "sentencepiece":       {"package_name": "sentencepiece",         "display_name": "sentencepiece",        "required_by_default": False},
    "sageattention":       {"package_name": "sageattention",         "display_name": "sageattention",        "required_by_default": False},
    "flash_attn":          {"package_name": "flash-attn",            "display_name": "flash-attn",           "required_by_default": False},
    "bitsandbytes":        {"package_name": "bitsandbytes",          "display_name": "bitsandbytes",         "required_by_default": False},
    "transformers":        {"package_name": "transformers",          "display_name": "transformers",         "required_by_default": True},
    "diffusers":           {"package_name": "diffusers",             "display_name": "diffusers",            "required_by_default": True},
    "requests":            {"package_name": "requests",              "display_name": "requests",             "required_by_default": False},
    "psutil":              {"package_name": "psutil",                "display_name": "psutil",               "required_by_default": False},
    "cv2":                 {"package_name": "opencv-python",         "display_name": "opencv-python",        "required_by_default": False},
    "matplotlib":          {"package_name": "matplotlib",            "display_name": "matplotlib",           "required_by_default": False},
    "scipy":               {"package_name": "scipy",                 "display_name": "scipy",                "required_by_default": False},
    "polars":              {"package_name": "polars",                "display_name": "polars",               "required_by_default": False},
    "torchvision":         {"package_name": "torchvision",           "display_name": "torchvision",          "required_by_default": False},
    "open_clip":           {"package_name": "open-clip-torch",       "display_name": "open-clip-torch",      "required_by_default": False},
    "timm":                {"package_name": "timm",                  "display_name": "timm",                 "required_by_default": False},
    "tqdm":                {"package_name": "tqdm",                  "display_name": "tqdm",                 "required_by_default": False},
    "yaml":                {"package_name": "PyYAML",                "display_name": "PyYAML",               "required_by_default": False},
    "PIL":                 {"package_name": "Pillow",                "display_name": "Pillow",               "required_by_default": False},
    "thop":                {"package_name": "ultralytics-thop",      "display_name": "ultralytics-thop",     "required_by_default": False},
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackageStatus:
    """Status of a single Python package requirement."""

    name: str
    available: bool
    version: str = ""
    error: str = ""


@dataclass(frozen=True)
class ToolStatus:
    """Status of a single system tool requirement."""

    name: str
    available: bool
    path: str = ""


@dataclass
class DependencyReport:
    """Outcome of a runtime dependency scan."""

    all_satisfied: bool
    packages: list[PackageStatus] = field(default_factory=list)
    tools: list[ToolStatus] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _check_package(name: str) -> PackageStatus:
    """Try to import *name* and return its status."""
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "")
        return PackageStatus(name=name, available=True, version=str(version))
    except ImportError as exc:
        return PackageStatus(name=name, available=False, error=str(exc))


def _check_tool(name: str) -> ToolStatus:
    """Check whether *name* is on PATH via :func:`shutil.which`."""
    resolved = shutil.which(name)
    return ToolStatus(name=name, available=resolved is not None, path=resolved or "")


def _metadata_version(package_name: str) -> str | None:
    """Return the installed package version from importlib.metadata."""
    try:
        return metadata.version(package_name)
    except (metadata.PackageNotFoundError, Exception):
        return None


def _safe_find_spec(module_name: str) -> Any:
    """Try importlib.util.find_spec without raising."""
    try:
        return importlib.util.find_spec(module_name)
    except Exception:
        return None


def _short_exc_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return message.splitlines()[0]


# ---------------------------------------------------------------------------
# Detailed package inspector (training-aware)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackageInspection:
    """Detailed status of one module from PACKAGE_REGISTRY."""

    module_name: str
    package_name: str
    display_name: str
    required_by_default: bool
    installed: bool
    importable: bool
    version: str | None
    reason: str


def inspect_runtime_package(
    module_name: str,
    *,
    probe_import: bool = True,
    runtime_environment: str = "",
) -> PackageInspection:
    """Inspect a single package by *module_name* (key in PACKAGE_REGISTRY).

    If *probe_import* is True, actually import the module to verify it works.
    *runtime_environment* allows callers to inject platform overrides (e.g.
    disabling bnb on ROCm).  When empty, defaults to normal behavior.
    """
    info = PACKAGE_REGISTRY.get(module_name, {
        "package_name": module_name.replace("_", "-"),
        "display_name": module_name,
        "required_by_default": False,
    })
    package_name: str = info["package_name"]
    display_name: str = info["display_name"]
    required_by_default: bool = bool(info.get("required_by_default", False))
    rt = runtime_environment.strip().lower()

    try:
        # Platform-conditional blocks
        if module_name == "pytorch_optimizer" and ("rocm" in rt or "xpu" in rt):
            version = _metadata_version(package_name)
            spec = _safe_find_spec(module_name)
            rt_label = "AMD ROCm" if "rocm" in rt else "Intel XPU"
            return PackageInspection(
                module_name=module_name, package_name=package_name,
                display_name=display_name, required_by_default=False,
                installed=(spec is not None or version is not None),
                importable=False, version=version,
                reason=f"{rt_label} runtime: pytorch-optimizer compatibility incomplete; auto-fallback active.",
            )
        if module_name == "bitsandbytes" and ("rocm" in rt or "xpu" in rt):
            version = _metadata_version(package_name)
            spec = _safe_find_spec(module_name)
            rt_label = "AMD ROCm" if "rocm" in rt else "Intel XPU"
            return PackageInspection(
                module_name=module_name, package_name=package_name,
                display_name=display_name, required_by_default=False,
                installed=(spec is not None or version is not None),
                importable=False, version=version,
                reason=f"{rt_label} runtime: bitsandbytes not available; 8bit/Paged optimizers auto-fallback.",
            )

        version = _metadata_version(package_name)
        spec = _safe_find_spec(module_name)
        installed = spec is not None or version is not None
        importable = False
        reason = ""

        if not installed:
            reason = "Package is not installed in the active runtime."
        elif not probe_import:
            importable = True
        else:
            try:
                importlib.import_module(module_name)
                importable = True
            except Exception as exc:
                reason = _short_exc_message(exc)

        return PackageInspection(
            module_name=module_name, package_name=package_name,
            display_name=display_name, required_by_default=required_by_default,
            installed=installed, importable=importable,
            version=version, reason=reason,
        )
    except Exception as exc:
        return PackageInspection(
            module_name=module_name, package_name=package_name,
            display_name=display_name, required_by_default=required_by_default,
            installed=False, importable=False, version=None,
            reason=f"inspection failed: {_short_exc_message(exc)}",
        )


def build_runtime_status_payload(
    module_names: Iterable[str] | None = None,
    *,
    probe_import: bool = True,
    runtime_environment: str = "",
) -> dict[str, Any]:
    """Return a dict describing the import/readiness state of tracked packages."""
    tracked = list(module_names or PACKAGE_REGISTRY.keys())
    packages = {
        name: _inspection_to_dict(inspect_runtime_package(
            name, probe_import=probe_import, runtime_environment=runtime_environment,
        ))
        for name in tracked
    }
    required_ready = all(
        p["importable"] for p in packages.values() if p["required_by_default"]
    )
    return {
        "environment": runtime_environment or "default",
        "runtime_experimental": "rocm" in runtime_environment.lower() or "xpu" in runtime_environment.lower(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "required_ready": required_ready,
        "packages": packages,
        "warnings": _runtime_payload_warnings(packages, runtime_environment=runtime_environment),
    }


def _runtime_payload_warnings(
    packages: dict[str, dict[str, Any]],
    *,
    runtime_environment: str,
) -> list[str]:
    rt = str(runtime_environment or "").strip().lower()
    warnings: list[str] = []
    if "rocm" in rt:
        for module_name in ("bitsandbytes", "pytorch_optimizer"):
            pkg = packages.get(module_name) or {}
            if pkg.get("installed"):
                warnings.append(
                    f"{module_name} is installed but is treated as incompatible residual state on AMD ROCm."
                )
    return warnings


def _inspection_to_dict(insp: PackageInspection) -> dict[str, Any]:
    return {
        "module_name": insp.module_name,
        "package_name": insp.package_name,
        "display_name": insp.display_name,
        "required_by_default": insp.required_by_default,
        "installed": insp.installed,
        "importable": insp.importable,
        "version": insp.version,
        "reason": insp.reason,
    }


# ---------------------------------------------------------------------------
# Dependency rules — maps training config → required modules
# ---------------------------------------------------------------------------

BUILTIN_LR_SCHEDULERS: frozenset[str] = frozenset({
    "linear", "cosine", "cosine_with_restarts", "polynomial",
    "constant", "constant_with_warmup", "cosine_with_min_lr",
    "loss_gated_cosine", "loss_weighted_annealed_cosine",
    "piecewise_constant", "one_cycle", "inverse_sqrt",
    "warmup_stable_decay", "restart_linear", "adafactor",
})

CUSTOM_SCHEDULER_PREFIX = "__custom__:"


def _config_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def collect_training_dependency_requirements(config: dict[str, Any]) -> dict[str, list[str]]:
    """Analyze *config* and return ``{module_name: [reason, ...]}`` for all
    packages that should be available for this training run.

    This re-implements the behavioral surface of the fork's
    ``runtime_dependency_rules.collect_training_dependency_requirements``
    without copying any code.
    """
    reqs: dict[str, list[str]] = {}

    def _add(module: str, reason: str) -> None:
        if module not in reqs:
            reqs[module] = []
        if reason not in reqs[module]:
            reqs[module].append(reason)

    # --- optimizer ---
    opt = str(config.get("optimizer_type", "")).strip()
    runtime_id = str(config.get("runtime_id") or config.get("execution_profile_id") or "").strip().lower()
    is_mps_runtime = runtime_id in {"apple-mps", "mps", "mac-mps", "darwin-mps"}
    if opt:
        lower = opt.lower()
        if "." in opt:
            _add(opt.split(".", 1)[0], f"optimizer_type={opt}")
        elif opt == "Lion":
            _add("lion_pytorch", f"optimizer_type={opt}")
        elif opt == "AdaFactor":
            _add("transformers", f"optimizer_type={opt}")
        elif (lower.endswith("8bit") or lower.startswith("paged")) and not is_mps_runtime:
            _add("bitsandbytes", f"optimizer_type={opt}")
        elif lower.startswith("dadapt"):
            _add("dadaptation", f"optimizer_type={opt}")
        elif opt == "Prodigy":
            _add("prodigyopt", f"optimizer_type={opt}")
        elif lower.endswith("schedulefree"):
            _add("schedulefree", f"optimizer_type={opt}")
        elif opt == "PytorchOptimizer" and not is_mps_runtime:
            _add("pytorch_optimizer", f"optimizer_type={opt}")

    # --- scheduler ---
    sched = str(config.get("lr_scheduler_type", "")).strip()
    if sched:
        normalized_sched = sched
        if normalized_sched.startswith(CUSTOM_SCHEDULER_PREFIX):
            normalized_sched = normalized_sched[len(CUSTOM_SCHEDULER_PREFIX):]
        if normalized_sched not in BUILTIN_LR_SCHEDULERS and "." in normalized_sched:
            _add(normalized_sched.split(".", 1)[0], f"lr_scheduler_type={sched}")

    # --- attention ---
    attn_mode = str(config.get("attn_mode", "")).strip().lower()
    if attn_mode in {"flash", "flash2", "flashattn", "flashattention", "flashattention2", "fa2"}:
        _add("flash_attn", f"attn_mode={attn_mode}")
    elif attn_mode == "sageattn":
        _add("sageattention", "attn_mode=sageattn")
    if _config_flag(config.get("use_flash_attn")):
        _add("flash_attn", "use_flash_attn=true")
    if _config_flag(config.get("flashattn")):
        _add("flash_attn", "flashattn=true")
    if _config_flag(config.get("use_sage_attn")):
        _add("sageattention", "use_sage_attn=true")
    if _config_flag(config.get("sageattn")):
        _add("sageattention", "sageattn=true")

    # --- network module ---
    net_mod = str(config.get("network_module", "")).strip()
    if net_mod.lower().startswith("lycoris."):
        _add("lycoris", f"network_module={net_mod}")

    # --- training type specific ---
    mtt = str(config.get("model_train_type", "")).strip().lower()
    if mtt.startswith("anima"):
        _add("safetensors", f"model_train_type={mtt}")
        _add("sentencepiece", f"model_train_type={mtt}")
    if mtt == "yolo":
        for mod in ("cv2", "matplotlib", "scipy", "polars", "requests", "psutil", "torchvision", "PIL", "yaml"):
            _add(mod, f"model_train_type={mtt}")
    if mtt == "aesthetic-scorer":
        for mod in ("open_clip", "timm", "transformers", "safetensors", "PIL", "tqdm"):
            _add(mod, f"model_train_type={mtt}")

    return reqs


def analyze_training_runtime_dependencies(
    config: dict[str, Any],
    *,
    runtime_environment: str = "",
) -> dict[str, Any]:
    """High-level: collect requirements from *config*, inspect each, return
    a structured report with ``ready``, ``required``, and ``missing`` lists.
    """
    requirements = collect_training_dependency_requirements(config)
    if not requirements:
        return {"ready": True, "required": [], "missing": []}

    required_records: list[dict] = []
    missing_records: list[dict] = []
    for module_name, reasons in requirements.items():
        insp = inspect_runtime_package(module_name, runtime_environment=runtime_environment)
        record = {**_inspection_to_dict(insp), "required_for": reasons}
        required_records.append(record)
        if not insp.importable:
            missing_records.append(record)

    return {
        "ready": len(missing_records) == 0,
        "required": required_records,
        "missing": missing_records,
    }


# ---------------------------------------------------------------------------
# Basic checker (backward-compatible API)
# ---------------------------------------------------------------------------

class RuntimeDependencyChecker:
    """Validates that required Python packages and system tools are present.

    Stateless after construction.  Call :meth:`check` with lists of
    package and tool names to obtain a :class:`DependencyReport`.

    Example::

        checker = RuntimeDependencyChecker()
        report = checker.check(
            packages=["torch", "diffusers"],
            tools=["nvidia-smi"],
        )
        if not report.all_satisfied:
            for e in report.errors:
                print(e)
    """

    def check(
        self,
        *,
        packages: list[str] | None = None,
        tools: list[str] | None = None,
    ) -> DependencyReport:
        bag = MessageBag()
        pkg_statuses: list[PackageStatus] = []
        tool_statuses: list[ToolStatus] = []

        for name in packages or []:
            status = _check_package(name)
            pkg_statuses.append(status)
            if not status.available:
                bag.add_error(f"Missing Python package: {name}")
            else:
                bag.add_note(f"Package {name} {status.version}".strip())

        for name in tools or []:
            status = _check_tool(name)
            tool_statuses.append(status)
            if not status.available:
                bag.add_warning(f"System tool not found on PATH: {name}")

        return DependencyReport(
            all_satisfied=bag.is_clean and not any(
                not s.available for s in pkg_statuses
            ),
            packages=pkg_statuses,
            tools=tool_statuses,
            errors=list(bag.errors),
            warnings=list(bag.warnings),
            notes=list(bag.notes),
        )

    @staticmethod
    def check_package(name: str) -> PackageStatus:
        return _check_package(name)

    @staticmethod
    def check_tool(name: str) -> ToolStatus:
        return _check_tool(name)

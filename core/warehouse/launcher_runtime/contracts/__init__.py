"""Public data contracts for the launcher/runtime layer."""

from .runtime import (
    CapabilityTag,
    CompatibilityRule,
    CompatibilityStatus,
    RuntimeCategory,
    RuntimeDef,
)
from .status import IntegrityReport, RuntimeStatus
from .launch import LaunchOptions, LaunchPlan, PlanStep
from .install import InstallProgress, InstallResult, InstallSectionSpec
from .task import ProgressEvent, TaskResult, TaskResultRecord, TaskState
from .gpu import GpuInfo, GpuStats
from .update import UpdateInfo, UpdateResult
from .plugin import PluginInfo, PluginManifest
from .diagnostics import (
    CheckSeverity,
    DiagnosticCheck,
    DiagnosticFinding,
    HealthReport,
    PreflightResult,
)

__all__ = [
    # runtime
    "CapabilityTag",
    "CompatibilityRule",
    "CompatibilityStatus",
    "RuntimeCategory",
    "RuntimeDef",
    # status
    "IntegrityReport",
    "RuntimeStatus",
    # launch
    "LaunchOptions",
    "LaunchPlan",
    "PlanStep",
    # install
    "InstallProgress",
    "InstallResult",
    "InstallSectionSpec",
    # task
    "ProgressEvent",
    "TaskResult",
    "TaskResultRecord",
    "TaskState",
    # gpu
    "GpuInfo",
    "GpuStats",
    # update
    "UpdateInfo",
    "UpdateResult",
    # plugin
    "PluginInfo",
    "PluginManifest",
    # diagnostics
    "CheckSeverity",
    "DiagnosticCheck",
    "DiagnosticFinding",
    "HealthReport",
    "PreflightResult",
]

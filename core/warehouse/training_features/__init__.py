"""Training features — Warehouse utilities for dataset analysis, caching,
checkpoint resume, trainer registration, route contracts, preflight
orchestration, and training configuration validation."""

from .cache_preflight import (
    BucketEntry,
    BucketPlan,
    CachePreflightReport,
    DatasetCachePreflight,
    LatentCacheProfile,
    resolve_cache_profile,
)
from .dataset_inspector import (
    DatasetInspector,
    DatasetReport,
    FolderStats,
)
from .preflight import (
    PreflightConfig,
    PreflightOrchestrator,
    PreflightResult,
)
from .resource_check import (
    DiskStatus,
    GpuStatus,
    ResourceChecker,
    ResourceReport,
)
from .resume_guard import (
    CheckpointEntry,
    LaunchGuardReport,
    ResumeGuard,
    ResumeReport,
    check_state_dir_complete,
    scan_output_artifacts,
)
from .route_contract import (
    ParamChoice,
    ParamRange,
    RouteContract,
    RouteContractSet,
    VramHint,
)
from .runtime_dependencies import (
    DependencyReport,
    PackageStatus,
    RuntimeDependencyChecker,
    ToolStatus,
    analyze_training_runtime_dependencies,
    collect_training_dependency_requirements,
    inspect_runtime_package,
)
from .serializers import (
    contract_set_from_dict,
    contract_set_from_json,
    contract_set_to_dict,
    contract_set_to_json,
    load_contract_set_json,
    load_registry_json,
    param_choice_from_dict,
    param_choice_to_dict,
    param_range_from_dict,
    param_range_to_dict,
    registry_from_dict,
    registry_from_json,
    registry_to_dict,
    registry_to_json,
    route_contract_from_dict,
    route_contract_to_dict,
    save_contract_set_json,
    save_registry_json,
    trainer_entry_from_dict,
    trainer_entry_to_dict,
    vram_hint_from_dict,
    vram_hint_to_dict,
)
from .trainer_registry import (
    TrainerEntry,
    TrainerRegistry,
)
from .training_config_checks import (
    ConfigCheckReport,
    TrainingPreflightConfig,
    check_anima_full_finetune_preflight,
    check_anima_requirements,
    check_channels_last,
    check_clear_cache,
    check_dataset_config_reference,
    check_eval_dataset,
    check_flux_requirements,
    check_learning_rates,
    check_masked_loss,
    check_network_targets,
    check_run_manifest_resume,
    check_sampler_compatibility,
    check_save_state,
    check_sage_attention,
    check_sdxl_clip_skip,
    check_torch_compile,
    check_validation_split,
    run_training_config_checks,
)
from .types import (
    AttentionBackend,
    Capability,
    MessageBag,
    ModelArchitecture,
    PreflightVerdict,
    RouteFamily,
    Severity,
)
from .runtime_env import (
    AcceleratorInfo,
    AttentionBackendTag,
    MemoryConfig,
    RuntimeDetector,
    RuntimeSnapshot,
    VendorTag,
)
from .batch_resolution import (
    BatchResolution,
    resolve_batch_config,
    validate_batch_config,
)
from .script_lifecycle import (
    ScriptDescriptor,
    ScriptResult,
    execute_script,
    prepared_environment,
    resolve_script_path,
)

__all__ = [
    # types
    "AttentionBackend",
    "Capability",
    "MessageBag",
    "ModelArchitecture",
    "PreflightVerdict",
    "RouteFamily",
    "Severity",
    # cache_preflight
    "BucketEntry",
    "BucketPlan",
    "CachePreflightReport",
    "DatasetCachePreflight",
    "LatentCacheProfile",
    "resolve_cache_profile",
    # dataset_inspector
    "DatasetInspector",
    "DatasetReport",
    "FolderStats",
    # preflight
    "PreflightConfig",
    "PreflightOrchestrator",
    "PreflightResult",
    # resource_check
    "DiskStatus",
    "GpuStatus",
    "ResourceChecker",
    "ResourceReport",
    # resume_guard
    "CheckpointEntry",
    "LaunchGuardReport",
    "ResumeGuard",
    "ResumeReport",
    "check_state_dir_complete",
    "scan_output_artifacts",
    # route_contract
    "ParamChoice",
    "ParamRange",
    "RouteContract",
    "RouteContractSet",
    "VramHint",
    # runtime_dependencies
    "DependencyReport",
    "PackageStatus",
    "RuntimeDependencyChecker",
    "ToolStatus",
    "analyze_training_runtime_dependencies",
    "collect_training_dependency_requirements",
    "inspect_runtime_package",
    # trainer_registry
    "TrainerEntry",
    "TrainerRegistry",
    # training_config_checks
    "ConfigCheckReport",
    "TrainingPreflightConfig",
    "check_anima_full_finetune_preflight",
    "check_anima_requirements",
    "check_channels_last",
    "check_clear_cache",
    "check_dataset_config_reference",
    "check_eval_dataset",
    "check_flux_requirements",
    "check_learning_rates",
    "check_masked_loss",
    "check_network_targets",
    "check_run_manifest_resume",
    "check_sampler_compatibility",
    "check_save_state",
    "check_sage_attention",
    "check_sdxl_clip_skip",
    "check_torch_compile",
    "check_validation_split",
    "run_training_config_checks",
    # serializers
    "contract_set_from_dict",
    "contract_set_from_json",
    "contract_set_to_dict",
    "contract_set_to_json",
    "load_contract_set_json",
    "load_registry_json",
    "param_choice_from_dict",
    "param_choice_to_dict",
    "param_range_from_dict",
    "param_range_to_dict",
    "registry_from_dict",
    "registry_from_json",
    "registry_to_dict",
    "registry_to_json",
    "route_contract_from_dict",
    "route_contract_to_dict",
    "save_contract_set_json",
    "save_registry_json",
    "trainer_entry_from_dict",
    "trainer_entry_to_dict",
    "vram_hint_from_dict",
    "vram_hint_to_dict",
    # runtime_env
    "AcceleratorInfo",
    "AttentionBackendTag",
    "MemoryConfig",
    "RuntimeDetector",
    "RuntimeSnapshot",
    "VendorTag",
    # batch_resolution
    "BatchResolution",
    "resolve_batch_config",
    "validate_batch_config",
    # script_lifecycle
    "ScriptDescriptor",
    "ScriptResult",
    "execute_script",
    "prepared_environment",
    "resolve_script_path",
]


"""Request-native platform contracts.

These models describe orchestration boundaries only. They are intended for API,
launcher, WebUI, CLI, plugin, and runner handoff layers, not for repeated
training-step hot paths.
"""

from .base import (
    ArtifactFile,
    ArtifactManifest,
    BaseRequest,
    JobEvent,
    PlatformIssue,
    RequestMetadata,
    RequestSource,
    RunResult,
    RunStatus,
)
from .base_model_tensorrt_runtime import (
    BASE_MODEL_TENSORRT_RUNTIME_ABI,
    BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID,
    BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID,
    BaseModelTensorRtRuntimeMode,
    BaseModelTensorRtRuntimeRequest,
    BaseModelTensorRtRuntimeResult,
)
from .generation import GenerationRequest, GenerationResult
from .image_gguf_runtime import (
    IMAGE_GGUF_RUNTIME_LOADER_ABI,
    IMAGE_GGUF_RUNTIME_PROVIDER,
    IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID,
    IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID,
    IMAGE_GGUF_SUPPORTED_COMPONENTS,
    ImageGGUFLoadMode,
    ImageGGUFRuntimeLoadRequest,
    ImageGGUFRuntimeLoadResult,
    build_image_gguf_runtime_loader_abi,
)
from .runner import RunContext, RunnerProtocol, RunnerRegistry
from .tools import (
    ArtifactReportRequest,
    ArtifactValidationRequest,
    DitFewStepLoraRequest,
    LabDistillerRequest,
    LabRequest,
    ToolRequest,
    TurboLoraRequest,
)
from .training import TrainingRequest
from .preprocess import DatasetArtifactManifest, PreprocessRequest
from .quality import QualityMetric, QualityReport, QualityReportRequest, QualitySample
from .runtime import (
    RepairAction,
    RuntimeAction,
    RuntimeOptimizationRequest,
    RuntimeOptimizationResolution,
    RuntimeRequest,
)
from .turbocore import TurboCoreBridgeAction, TurboCoreBridgeRequest, TurboCoreBridgeResult
from .plugin_sdk import (
    PluginArtifactHandlerRegistration,
    PluginPermissionRequest,
    PluginRequestSchemaRegistration,
    PluginResourceDetectorRegistration,
    PluginRunnerRegistration,
    PluginUiSlotRegistration,
)

__all__ = [
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactReportRequest",
    "ArtifactValidationRequest",
    "BASE_MODEL_TENSORRT_RUNTIME_ABI",
    "BASE_MODEL_TENSORRT_RUNTIME_REQUEST_SCHEMA_ID",
    "BASE_MODEL_TENSORRT_RUNTIME_RESULT_SCHEMA_ID",
    "BaseRequest",
    "BaseModelTensorRtRuntimeMode",
    "BaseModelTensorRtRuntimeRequest",
    "BaseModelTensorRtRuntimeResult",
    "DatasetArtifactManifest",
    "DitFewStepLoraRequest",
    "GenerationRequest",
    "GenerationResult",
    "IMAGE_GGUF_RUNTIME_LOADER_ABI",
    "IMAGE_GGUF_RUNTIME_PROVIDER",
    "IMAGE_GGUF_RUNTIME_REQUEST_SCHEMA_ID",
    "IMAGE_GGUF_RUNTIME_RESULT_SCHEMA_ID",
    "IMAGE_GGUF_SUPPORTED_COMPONENTS",
    "ImageGGUFLoadMode",
    "ImageGGUFRuntimeLoadRequest",
    "ImageGGUFRuntimeLoadResult",
    "JobEvent",
    "LabDistillerRequest",
    "LabRequest",
    "PlatformIssue",
    "PluginArtifactHandlerRegistration",
    "PluginPermissionRequest",
    "PluginRequestSchemaRegistration",
    "PluginResourceDetectorRegistration",
    "PluginRunnerRegistration",
    "PluginUiSlotRegistration",
    "PreprocessRequest",
    "QualityMetric",
    "QualityReport",
    "QualityReportRequest",
    "QualitySample",
    "RepairAction",
    "RequestMetadata",
    "RequestSource",
    "RunContext",
    "RunnerProtocol",
    "RunnerRegistry",
    "RunResult",
    "RunStatus",
    "RuntimeAction",
    "RuntimeOptimizationRequest",
    "RuntimeOptimizationResolution",
    "RuntimeRequest",
    "ToolRequest",
    "TrainingRequest",
    "TurboCoreBridgeAction",
    "TurboCoreBridgeRequest",
    "TurboCoreBridgeResult",
    "TurboLoraRequest",
    "build_image_gguf_runtime_loader_abi",
]


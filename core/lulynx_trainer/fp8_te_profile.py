# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""TransformerEngine FP8 capability profile.

This module only reports whether TE FP8 training is a viable experimental
route.  It does not enable FP8 or mutate model code.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class FP8TEProfile:
    requested: bool
    available: bool
    resolved: str
    fallback_reason: str = ""
    notes: List[str] = field(default_factory=list)
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes or [])
        data["capabilities"] = dict(self.capabilities or {})
        return data


def normalize_precision_experiment(value: Any) -> str:
    text = str(value or "bf16").strip().lower().replace("-", "_")
    aliases = {
        "": "bf16",
        "default": "bf16",
        "off": "bf16",
        "none": "bf16",
        "fp8": "fp8_te",
        "te_fp8": "fp8_te",
        "transformerengine": "fp8_te",
        "transformer_engine": "fp8_te",
    }
    text = aliases.get(text.replace(" ", ""), text)
    return text if text in {"bf16", "fp8_te"} else "bf16"


def _cuda_capability() -> Tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return tuple(int(v) for v in torch.cuda.get_device_capability())  # type: ignore[return-value]
    except Exception:
        return None


def _te_version() -> Tuple[bool, str, str]:
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None:
        return False, "", "transformer_engine is not installed"
    try:
        import transformer_engine  # type: ignore

        version = str(getattr(transformer_engine, "__version__", "unknown"))
        return True, version, ""
    except Exception as exc:
        return False, "", f"transformer_engine import failed: {type(exc).__name__}: {exc}"


def build_fp8_te_profile(config: Any = None, *, requested: Any = None) -> FP8TEProfile:
    if requested is None:
        requested = getattr(config, "precision_experiment", None)
        if requested is None:
            requested = "fp8_te" if bool(getattr(config, "fp8_te_enabled", False)) else "bf16"
    mode = normalize_precision_experiment(requested)
    explicitly_requested = mode == "fp8_te"

    te_available, te_version, te_reason = _te_version()
    capability = _cuda_capability()
    cuda_available = bool(torch.cuda.is_available())
    compute_capability = None if capability is None else f"{capability[0]}.{capability[1]}"
    hopper_or_newer = capability is not None and capability >= (9, 0)
    ampere_or_newer = capability is not None and capability >= (8, 0)

    capabilities = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "",
        "compute_capability": compute_capability,
        "ampere_or_newer": ampere_or_newer,
        "hopper_or_newer": hopper_or_newer,
        "transformer_engine_available": te_available,
        "transformer_engine_version": te_version,
    }

    notes = [
        "FP8 TE is experimental and must stay opt-in; bf16 remains the stable default.",
        "This profile does not replace existing weight-compression fp8_base/storage-only paths.",
    ]

    if not explicitly_requested:
        return FP8TEProfile(
            requested=False,
            available=False,
            resolved="bf16",
            notes=["No FP8 TransformerEngine experiment requested."],
            capabilities=capabilities,
        )

    if not cuda_available:
        return FP8TEProfile(
            requested=True,
            available=False,
            resolved="bf16",
            fallback_reason="TransformerEngine FP8 training requires CUDA.",
            notes=notes,
            capabilities=capabilities,
        )
    if not te_available:
        return FP8TEProfile(
            requested=True,
            available=False,
            resolved="bf16",
            fallback_reason=te_reason or "transformer_engine is not available.",
            notes=notes,
            capabilities=capabilities,
        )
    if not ampere_or_newer:
        return FP8TEProfile(
            requested=True,
            available=False,
            resolved="bf16",
            fallback_reason="TransformerEngine FP8 requires NVIDIA Ampere/Hopper-class GPU support.",
            notes=notes,
            capabilities=capabilities,
        )

    if not hopper_or_newer:
        notes.append("Ampere-class FP8 support is dependency/version sensitive; require real benchmark before enabling in training.")

    return FP8TEProfile(
        requested=True,
        available=True,
        resolved="fp8_te",
        notes=notes,
        capabilities=capabilities,
    )


__all__ = ["FP8TEProfile", "build_fp8_te_profile", "normalize_precision_experiment"]

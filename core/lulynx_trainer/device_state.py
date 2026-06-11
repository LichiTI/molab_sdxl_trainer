"""Small device-state helpers for training phases.

This is intentionally conservative: it centralizes component device/mode
transitions without implementing layer-level weight streaming. More aggressive
offload engines can build on this contract later.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import torch


@dataclass(frozen=True)
class ModuleDeviceState:
    name: str
    training: Optional[bool] = None
    requires_grad: Optional[bool] = None
    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class CapturedModuleState:
    training: bool
    requires_grad: Optional[bool]
    device: torch.device
    dtype: Optional[torch.dtype]


def _first_param(module: torch.nn.Module) -> Optional[torch.nn.Parameter]:
    try:
        return next(module.parameters())
    except StopIteration:
        return None


def capture_module_state(module: Optional[torch.nn.Module]) -> Optional[CapturedModuleState]:
    if module is None:
        return None
    param = _first_param(module)
    requires_grad: Optional[bool] = None
    device = torch.device("cpu")
    dtype: Optional[torch.dtype] = None
    if param is not None:
        requires_grad = bool(param.requires_grad)
        device = param.device
        dtype = param.dtype
    return CapturedModuleState(
        training=bool(module.training),
        requires_grad=requires_grad,
        device=device,
        dtype=dtype,
    )


def apply_module_state(module: Optional[torch.nn.Module], state: ModuleDeviceState | CapturedModuleState) -> None:
    if module is None:
        return
    if state.training is not None:
        module.train() if state.training else module.eval()
    if state.requires_grad is not None:
        module.requires_grad_(bool(state.requires_grad))
    if state.device is not None:
        kwargs: dict[str, Any] = {"device": state.device}
        if state.dtype is not None:
            kwargs["dtype"] = state.dtype
        module.to(**kwargs)


def build_loaded_model_training_states(
    *,
    device: torch.device | str,
    train_text_encoder: bool,
    keep_text_encoders_on_cpu: bool,
    keep_vae_on_cpu: bool,
) -> dict[str, ModuleDeviceState]:
    te_on_cpu = bool(keep_text_encoders_on_cpu) and not bool(train_text_encoder)
    return {
        "unet": ModuleDeviceState(
            name="unet",
            training=True,
            device=device,
        ),
        "vae": ModuleDeviceState(
            name="vae",
            training=False,
            requires_grad=False,
            device="cpu" if keep_vae_on_cpu else device,
            dtype=torch.float32 if keep_vae_on_cpu else None,
        ),
        "text_encoder_1": ModuleDeviceState(
            name="text_encoder_1",
            training=bool(train_text_encoder),
            requires_grad=None if train_text_encoder else False,
            device="cpu" if te_on_cpu else device,
            dtype=torch.float32 if te_on_cpu else None,
        ),
        "text_encoder_2": ModuleDeviceState(
            name="text_encoder_2",
            training=bool(train_text_encoder),
            requires_grad=None if train_text_encoder else False,
            device="cpu" if te_on_cpu else device,
            dtype=torch.float32 if te_on_cpu else None,
        ),
    }


def apply_loaded_model_training_states(model: Any, states: Mapping[str, ModuleDeviceState]) -> None:
    for name, state in states.items():
        apply_module_state(getattr(model, name, None), state)


@contextmanager
def module_runtime_state(
    module: Optional[torch.nn.Module],
    *,
    device: torch.device | str,
    dtype: Optional[torch.dtype] = None,
    restore: bool = True,
) -> Iterator[None]:
    if module is None:
        yield
        return
    captured = capture_module_state(module)
    runtime_state = ModuleDeviceState(
        name="runtime",
        device=device,
        dtype=dtype,
    )
    apply_module_state(module, runtime_state)
    try:
        yield
    finally:
        if restore and captured is not None:
            apply_module_state(module, captured)

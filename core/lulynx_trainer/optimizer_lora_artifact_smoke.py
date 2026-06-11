"""Artifact smoke for MN-LoRA++ and AutoProdigy LoRA outputs.

The test trains a tiny injected LoRA adapter, writes a safetensors checkpoint,
reloads it into a fresh model, and verifies the exported LoRA structure and
runtime output.
"""

from __future__ import annotations

import math
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"
MNLORA_ROOT = CORE_ROOT / "training_components" / "mn_lora"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_xformers_stub()
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_ensure_namespace("core.training_components", CORE_ROOT / "training_components")
_ensure_namespace("core.training_components.mn_lora", MNLORA_ROOT)
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
_load_module("core.lulynx_trainer.tlora", TRAINER_ROOT / "tlora.py")
_load_module("core.training_components.mn_lora.trace_guided_wd", MNLORA_ROOT / "trace_guided_wd.py")
_load_module("core.training_components.mn_lora.svd_utils", MNLORA_ROOT / "svd_utils.py")
_load_module("core.training_components.mn_lora.gradient_subspace", MNLORA_ROOT / "gradient_subspace.py")
_load_module("core.training_components.mn_lora.mn_lora_plus_plus", MNLORA_ROOT / "mn_lora_plus_plus.py")
lora_mod = _load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")
ap_mod = _load_module("core.lulynx_trainer.auto_prodigy_optimizer", TRAINER_ROOT / "auto_prodigy_optimizer.py")
mn_mod = _load_module("core.training_components.mn_lora.mn_optimizer", MNLORA_ROOT / "mn_optimizer.py")

AutoProdigy = ap_mod.AutoProdigy
LoRAInjector = lora_mod.LoRAInjector
infer_rank_from_weights = lora_mod.infer_rank_from_weights
MNLoRAOptimizer = mn_mod.MNLoRAOptimizer


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def _make_injected_model(seed: int = 1234) -> tuple[nn.Module, LoRAInjector]:
    torch.manual_seed(seed)
    model = TinyAttention()
    injector = LoRAInjector(rank=2, alpha=2, model_arch="anima")
    injected = injector.inject(model, ["q_proj"], prefix="tiny")
    assert len(injected) == 1, f"Expected one LoRA layer, got {len(injected)}"
    return model, injector


def _assert_lora_structure(state: dict[str, torch.Tensor]) -> None:
    down_keys = [key for key in state if key.endswith(".lora_down.weight")]
    up_keys = [key for key in state if key.endswith(".lora_up.weight")]
    assert down_keys, f"Missing lora_down.weight in keys: {sorted(state)}"
    assert up_keys, f"Missing lora_up.weight in keys: {sorted(state)}"
    assert len(down_keys) == len(up_keys) == 1

    down = state[down_keys[0]]
    up = state[up_keys[0]]
    assert down.shape == (2, 6), f"Unexpected lora_down shape: {tuple(down.shape)}"
    assert up.shape == (4, 2), f"Unexpected lora_up shape: {tuple(up.shape)}"
    assert infer_rank_from_weights(state) == 2
    for key, tensor in state.items():
        assert torch.is_tensor(tensor), f"{key} is not a tensor"
        assert torch.isfinite(tensor).all(), f"{key} contains NaN/Inf"
    assert float(up.abs().sum()) > 0.0, "lora_up stayed all-zero after training"


def _train_and_export(name: str, optimizer_kind: str) -> Path:
    model, injector = _make_injected_model()
    x = torch.randn(8, 6)
    target = torch.randn(8, 4)

    params = injector.get_trainable_params()
    if optimizer_kind == "mn_lora_plus_plus":
        base_optimizer = torch.optim.AdamW(params, lr=2e-2, weight_decay=0.0)
        param_names = {id(param): pname for pname, param in model.named_parameters()}
        optimizer = MNLoRAOptimizer(
            base_optimizer,
            enable_tgwd=False,
            enable_gsp=False,
            enable_pilot=False,
            plus_plus_config={
                "enabled": True,
                "lr_up": 1.05,
                "lr_down": 0.8,
                "min_mult": 0.25,
                "max_mult": 2.0,
                "update_rms_cap": 0.01,
            },
            param_names=param_names,
        )
    elif optimizer_kind == "auto_prodigy":
        optimizer = AutoProdigy(params, lr=1.0, d0=1e-2, growth_rate=1.2)
    else:
        raise ValueError(optimizer_kind)

    initial = model(x).detach()
    for _ in range(12):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

    trained_output = model(x).detach()
    assert torch.isfinite(trained_output).all()
    assert not torch.allclose(initial, trained_output), f"{name} did not change model output"

    state = {key: value.detach().cpu().clone() for key, value in injector.get_lora_state_dict().items()}
    _assert_lora_structure(state)

    out_dir = Path.cwd() / "tmp" / "optimizer_lora_artifact_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.safetensors"
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for this artifact smoke") from exc
    save_file(
        state,
        str(out_path),
        metadata={
            "ss_network_dim": "2",
            "ss_network_alpha": "2",
            "optimizer_smoke": name,
        },
    )
    loaded = load_file(str(out_path), device="cpu")
    _assert_lora_structure(loaded)

    reloaded_model, reloaded_injector = _make_injected_model()
    loaded_count, expected_count = reloaded_injector.load_lora_state_dict(loaded)
    assert loaded_count == expected_count == 2, (loaded_count, expected_count)
    reloaded_output = reloaded_model(x).detach()
    torch.testing.assert_close(reloaded_output, trained_output, rtol=1e-5, atol=1e-6)

    delta_norm = math.sqrt(sum(float(t.float().square().sum()) for t in loaded.values()))
    assert delta_norm > 0.0
    print(f"  {name}: wrote {out_path} with {len(loaded)} tensors, delta_norm={delta_norm:.6f} -- PASS")
    return out_path


def main() -> int:
    mn_path = _train_and_export("mn_lora_plus_plus", "mn_lora_plus_plus")
    ap_path = _train_and_export("auto_prodigy", "auto_prodigy")
    print("LoRA artifact smoke passed:")
    print(f"  MN-LoRA++ artifact: {mn_path}")
    print(f"  AutoProdigy artifact: {ap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

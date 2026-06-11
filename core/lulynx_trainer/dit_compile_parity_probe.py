"""Default-off compile parity probe for DiT-shaped training blocks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DiTCompileParityProbeConfig:
    batch_size: int = 1
    token_count: int = 8
    hidden_size: int = 16
    num_heads: int = 4
    steps: int = 1
    seed: int = 20260605
    compile_backend: str = "eager"
    compile_mode: str = "default"
    max_output_abs_diff: float = 1e-5
    max_loss_delta: float = 1e-6
    max_grad_abs_diff: float = 1e-5
    device: str = "cpu"
    dtype: str = "float32"

    def normalized(self) -> "DiTCompileParityProbeConfig":
        dtype = str(self.dtype or "float32").strip().lower()
        if dtype not in {"float32", "float16", "bfloat16"}:
            dtype = "float32"
        mode = str(self.compile_mode or "default").strip()
        if mode not in {"default", "reduce-overhead", "max-autotune"}:
            mode = "default"
        return DiTCompileParityProbeConfig(
            batch_size=max(int(self.batch_size), 1),
            token_count=max(int(self.token_count), 2),
            hidden_size=max(int(self.hidden_size), 2),
            num_heads=max(int(self.num_heads), 1),
            steps=max(int(self.steps), 1),
            seed=int(self.seed),
            compile_backend=str(self.compile_backend or "eager"),
            compile_mode=mode,
            max_output_abs_diff=max(float(self.max_output_abs_diff), 0.0),
            max_loss_delta=max(float(self.max_loss_delta), 0.0),
            max_grad_abs_diff=max(float(self.max_grad_abs_diff), 0.0),
            device=str(self.device or "cpu"),
            dtype=dtype,
        )


class TinyDiTCompileParityBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = int(hidden_size // num_heads)
        self.norm1 = torch.nn.LayerNorm(hidden_size)
        self.qkv = torch.nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm2 = torch.nn.LayerNorm(hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, token_count, hidden = tokens.shape
        qkv = self.qkv(self.norm1(tokens))
        qkv = qkv.view(batch, token_count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        mixed = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], dropout_p=0.0)
        mixed = mixed.transpose(1, 2).reshape(batch, token_count, hidden)
        tokens = tokens + self.proj(mixed)
        return tokens + self.mlp(self.norm2(tokens))


def run_dit_compile_parity_probe(
    config: DiTCompileParityProbeConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    if cfg.hidden_size % cfg.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)
    eager = TinyDiTCompileParityBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    compiled_source = TinyDiTCompileParityBlock(cfg.hidden_size, cfg.num_heads).to(device=device, dtype=dtype)
    compiled_source.load_state_dict(eager.state_dict())
    tokens = torch.randn(cfg.batch_size, cfg.token_count, cfg.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125

    blockers: list[str] = []
    compile_error = ""
    compiled_runner: Any = compiled_source
    if not hasattr(torch, "compile"):
        blockers.append("torch_compile_unavailable")
    else:
        try:
            compiled_runner = torch.compile(
                compiled_source,
                backend=cfg.compile_backend,
                mode=None if cfg.compile_mode == "default" else cfg.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            blockers.append("torch_compile_failed")

    eager_sample = _run_step(eager, eager, tokens, target)
    compiled_sample = _run_step(compiled_runner, compiled_source, tokens, target) if not compile_error else {}
    output_diff = _max_abs_diff(eager_sample.get("output"), compiled_sample.get("output"))
    input_grad_diff = _max_abs_diff(eager_sample.get("input_grad"), compiled_sample.get("input_grad"))
    param_grad_diff = _max_param_grad_diff(eager_sample.get("param_grads"), compiled_sample.get("param_grads"))
    loss_delta = abs(float(eager_sample.get("loss", 0.0)) - float(compiled_sample.get("loss", 0.0)))

    checks = {
        "compiled_output_shape": tuple(compiled_sample.get("output_shape", ())) == tuple(eager_sample.get("output_shape", ())),
        "finite_loss": math.isfinite(float(eager_sample.get("loss", 0.0))) and math.isfinite(float(compiled_sample.get("loss", 0.0))),
        "output_parity": output_diff <= cfg.max_output_abs_diff,
        "loss_parity": loss_delta <= cfg.max_loss_delta,
        "input_gradient_parity": input_grad_diff <= cfg.max_grad_abs_diff,
        "parameter_gradient_parity": param_grad_diff <= cfg.max_grad_abs_diff,
    }
    blockers.extend(f"{name}_failed" for name, ok in checks.items() if not ok)
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compile_parity_probe_v0",
        "ok": ready,
        "compile_parity_ready": ready,
        "route": "dit_tiny_block",
        "target": "per_block_compile",
        "compile_backend": cfg.compile_backend,
        "compile_mode": cfg.compile_mode,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "config": _config_payload(cfg, device, dtype),
        "eager_loss": float(eager_sample.get("loss", 0.0)),
        "compiled_loss": float(compiled_sample.get("loss", 0.0)),
        "loss_delta": loss_delta,
        "max_output_abs_diff": output_diff,
        "max_input_grad_abs_diff": input_grad_diff,
        "max_param_grad_abs_diff": param_grad_diff,
        "thresholds": {
            "max_output_abs_diff": float(cfg.max_output_abs_diff),
            "max_loss_delta": float(cfg.max_loss_delta),
            "max_grad_abs_diff": float(cfg.max_grad_abs_diff),
        },
        "checks": checks,
        "compile_error": compile_error,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run Anima/Newbie real cached per-block compile loss/gradient parity A/B"
            if ready
            else "resolve compile parity blockers before compile route recommendation"
        ),
    }


def build_dit_cached_shape_compile_parity_matrix(
    *,
    compile_backend: str = "eager",
    compile_mode: str = "default",
    seed: int = 20260605,
) -> dict[str, Any]:
    rows = [
        run_anima_native_tiny_per_block_compile_parity_probe(
            {
                "compile_backend": compile_backend,
                "compile_mode": compile_mode,
                "seed": seed,
            }
        ),
        run_newbie_cached_shape_compile_parity_probe(
            {
                "compile_backend": compile_backend,
                "compile_mode": compile_mode,
                "seed": seed + 1,
            }
        ),
    ]
    blockers = {
        str(row.get("family") or row.get("route") or ""): list(row.get("blocked_reasons") or [])
        for row in rows
        if row.get("blocked_reasons")
    }
    ready_rows = [row for row in rows if bool(row.get("compile_parity_ready", row.get("ok", False)))]
    return {
        "schema_version": 1,
        "scorecard": "dit_cached_shape_compile_parity_matrix_v0",
        "ok": len(ready_rows) == len(rows),
        "compile_parity_matrix_ready": len(ready_rows) == len(rows),
        "row_count": len(rows),
        "ready_row_count": len(ready_rows),
        "rows": rows,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "blocked_by_family": blockers,
        "recommended_next_step": (
            "run real cached Anima/Newbie block compile parity with loaded checkpoints"
            if len(ready_rows) == len(rows)
            else "resolve cached-shape compile parity blockers before real block probe"
        ),
    }


def run_anima_native_tiny_per_block_compile_parity_probe(
    config: DiTCompileParityProbeConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    from .anima_native_dit import AnimaNativeDiTTinyTrainable

    kwargs = {
        "latent_channels": 16,
        "hidden_dim": 8,
        "patch_size": 2,
        "block_count": 2,
        "condition_dim": 8,
        "device": str(device),
        "dtype": dtype,
    }
    eager = AnimaNativeDiTTinyTrainable(**kwargs)
    compiled = AnimaNativeDiTTinyTrainable(**kwargs)
    compiled.load_state_dict(eager.state_dict())
    sample = torch.randn(1, 16, 4, 4, device=device, dtype=dtype)
    timestep = torch.tensor([0.25], device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 4, 8, device=device, dtype=dtype)
    target = torch.randn_like(sample) * 0.125
    compile_error = _compile_anima_blocks(compiled, cfg)

    eager_sample = _run_anima_step(eager, sample, timestep, encoder_hidden_states, target)
    compiled_sample = (
        _run_anima_step(compiled, sample, timestep, encoder_hidden_states, target)
        if not compile_error
        else {}
    )
    return _parity_report(
        cfg,
        family="anima",
        route="anima_native_tiny_trainable",
        target="per_block_compile",
        eager_sample=eager_sample,
        compiled_sample=compiled_sample,
        compile_error=compile_error,
        extra={
            "compiled_block_count": len(getattr(compiled.net, "blocks", []) or []),
            "sample_shape": list(sample.shape),
            "encoder_hidden_shape": list(encoder_hidden_states.shape),
            "native_tiny_trainable": True,
        },
    )


def run_newbie_cached_shape_compile_parity_probe(
    config: DiTCompileParityProbeConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _config(config)
    latents = torch.linspace(-1.0, 1.0, steps=1 * 4 * 3 * 4, dtype=torch.float32).reshape(1, 4, 3, 4)
    from .diffcr_cached_token_ab import build_diffcr_cached_token_replay

    replay = build_diffcr_cached_token_replay(latents, family="newbie")
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype, device)
    torch.manual_seed(cfg.seed)

    from .newbie_loader import _NextDiTWrapper

    state = _tiny_nextdit_state_dict(hidden_dim=int(replay.hidden_size), ffn_dim=int(replay.hidden_size) * 2)
    eager = _NextDiTWrapper(state).to(device=device, dtype=dtype)
    compiled = _NextDiTWrapper(state).to(device=device, dtype=dtype)
    tokens = replay.tokens.to(device=device, dtype=dtype)
    timestep = torch.tensor([0.25], device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 5, replay.hidden_size, device=device, dtype=dtype)
    text_embeds = torch.randn(1, replay.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(tokens) * 0.125
    compile_error = _compile_newbie_blocks(compiled, cfg)

    eager_sample = _run_newbie_wrapper_step(eager, tokens, timestep, encoder_hidden_states, text_embeds, target)
    compiled_sample = (
        _run_newbie_wrapper_step(compiled, tokens, timestep, encoder_hidden_states, text_embeds, target)
        if not compile_error
        else {}
    )
    return _parity_report(
        cfg,
        family="newbie",
        route="newbie_nextdit_wrapper_cached_shape",
        target="per_block_compile",
        eager_sample=eager_sample,
        compiled_sample=compiled_sample,
        compile_error=compile_error,
        extra={
            "compiled_block_count": len(getattr(compiled, "_block_modules", []) or []),
            "cache_replay": replay.as_dict(),
            "wrapper": "_NextDiTWrapper",
        },
    )


def normalize_orig_mod_state_dict_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(state_dict).items():
        name = str(key)
        while name.startswith("_orig_mod."):
            name = name[len("_orig_mod.") :]
        while "._orig_mod." in name:
            name = name.replace("._orig_mod.", ".")
        normalized[name] = value
    return normalized


def _compile_anima_blocks(module: torch.nn.Module, cfg: DiTCompileParityProbeConfig) -> str:
    if not hasattr(torch, "compile"):
        return "torch.compile unavailable"
    try:
        for index, block in enumerate(module.net.blocks):
            module.net.blocks[index] = torch.compile(
                block,
                backend=cfg.compile_backend,
                mode=None if cfg.compile_mode == "default" else cfg.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _compile_newbie_blocks(module: torch.nn.Module, cfg: DiTCompileParityProbeConfig) -> str:
    if not hasattr(torch, "compile"):
        return "torch.compile unavailable"
    try:
        blocks = list(getattr(module, "_block_modules", []) or [])
        compiled_blocks = []
        for block in blocks:
            compiled_blocks.append(
                torch.compile(
                    block,
                    backend=cfg.compile_backend,
                    mode=None if cfg.compile_mode == "default" else cfg.compile_mode,
                    fullgraph=False,
                    dynamic=False,
                )
            )
        module._block_modules = compiled_blocks
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def _run_anima_step(
    module: torch.nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    module.zero_grad(set_to_none=True)
    work = sample.detach().clone().requires_grad_(True)
    output_obj = module(work, timestep, encoder_hidden_states)
    output = _extract_tensor_output(output_obj)
    loss = F.mse_loss(output.float(), target.float())
    loss.backward()
    return {
        "output": output.detach().float(),
        "output_shape": tuple(output.shape),
        "loss": float(loss.detach().item()),
        "input_grad": work.grad.detach().float() if work.grad is not None else None,
        "param_grads": normalize_orig_mod_state_dict_keys(
            {
                name: param.grad.detach().float().clone()
                for name, param in module.named_parameters()
                if param.grad is not None
            }
        ),
    }


def _run_newbie_wrapper_step(
    module: torch.nn.Module,
    tokens: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    text_embeds: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    module.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    output_obj = module(
        sample=work,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs={"text_embeds": text_embeds},
    )
    output = _extract_tensor_output(output_obj)
    loss = F.mse_loss(output.float(), target.float())
    loss.backward()
    return {
        "output": output.detach().float(),
        "output_shape": tuple(output.shape),
        "loss": float(loss.detach().item()),
        "input_grad": work.grad.detach().float() if work.grad is not None else None,
        "param_grads": normalize_orig_mod_state_dict_keys(
            {
                name: param.grad.detach().float().clone()
                for name, param in module.named_parameters()
                if param.grad is not None
            }
        ),
    }


def _tiny_nextdit_state_dict(hidden_dim: int = 4, ffn_dim: int = 8) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for prefix in ("layers.0", "context_refiner.0"):
        state[f"{prefix}.norm1.weight"] = torch.ones(hidden_dim)
        state[f"{prefix}.norm1.bias"] = torch.zeros(hidden_dim)
        state[f"{prefix}.norm2.weight"] = torch.ones(hidden_dim)
        state[f"{prefix}.norm2.bias"] = torch.zeros(hidden_dim)
        state[f"{prefix}.attention.qkv.weight"] = torch.randn(hidden_dim * 3, hidden_dim) * 0.02
        state[f"{prefix}.attention.qkv.bias"] = torch.zeros(hidden_dim * 3)
        state[f"{prefix}.attention.out.weight"] = torch.randn(hidden_dim, hidden_dim) * 0.02
        state[f"{prefix}.attention.out.bias"] = torch.zeros(hidden_dim)
        state[f"{prefix}.feed_forward.w1.weight"] = torch.randn(ffn_dim, hidden_dim) * 0.02
        state[f"{prefix}.feed_forward.w1.bias"] = torch.zeros(ffn_dim)
        state[f"{prefix}.feed_forward.w2.weight"] = torch.randn(hidden_dim, ffn_dim) * 0.02
        state[f"{prefix}.feed_forward.w2.bias"] = torch.zeros(hidden_dim)
        state[f"{prefix}.feed_forward.w3.weight"] = torch.randn(ffn_dim, hidden_dim) * 0.02
        state[f"{prefix}.feed_forward.w3.bias"] = torch.zeros(ffn_dim)
    return state


def _run_step(
    runner: torch.nn.Module,
    grad_source: torch.nn.Module,
    tokens: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    grad_source.zero_grad(set_to_none=True)
    work = tokens.detach().clone().requires_grad_(True)
    output = runner(work)
    loss = F.mse_loss(output.float(), target.float())
    loss.backward()
    return {
        "output": output.detach().float(),
        "output_shape": tuple(output.shape),
        "loss": float(loss.detach().item()),
        "input_grad": work.grad.detach().float() if work.grad is not None else None,
        "param_grads": {
            name: param.grad.detach().float().clone()
            for name, param in grad_source.named_parameters()
            if param.grad is not None
        },
    }


def _parity_report(
    cfg: DiTCompileParityProbeConfig,
    *,
    family: str,
    route: str,
    target: str,
    eager_sample: Mapping[str, Any],
    compiled_sample: Mapping[str, Any],
    compile_error: str = "",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_diff = _max_abs_diff(eager_sample.get("output"), compiled_sample.get("output"))
    input_grad_diff = _max_abs_diff(eager_sample.get("input_grad"), compiled_sample.get("input_grad"))
    param_grad_diff = _max_param_grad_diff(eager_sample.get("param_grads"), compiled_sample.get("param_grads"))
    loss_delta = abs(float(eager_sample.get("loss", 0.0)) - float(compiled_sample.get("loss", 0.0)))
    checks = {
        "compiled_output_shape": tuple(compiled_sample.get("output_shape", ())) == tuple(eager_sample.get("output_shape", ())),
        "finite_loss": math.isfinite(float(eager_sample.get("loss", 0.0))) and math.isfinite(float(compiled_sample.get("loss", 0.0))),
        "output_parity": output_diff <= cfg.max_output_abs_diff,
        "loss_parity": loss_delta <= cfg.max_loss_delta,
        "input_gradient_parity": input_grad_diff <= cfg.max_grad_abs_diff,
        "parameter_gradient_parity": param_grad_diff <= cfg.max_grad_abs_diff,
    }
    blockers = ["torch_compile_failed"] if compile_error else []
    blockers.extend(f"{name}_failed" for name, ok in checks.items() if not ok)
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compile_parity_probe_v0",
        "ok": ready,
        "compile_parity_ready": ready,
        "family": family,
        "route": route,
        "target": target,
        "compile_backend": cfg.compile_backend,
        "compile_mode": cfg.compile_mode,
        "training_path_enabled": False,
        "trainer_wiring_allowed": False,
        "request_fields_emitted": False,
        "runtime_activation_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "eager_loss": float(eager_sample.get("loss", 0.0)),
        "compiled_loss": float(compiled_sample.get("loss", 0.0)),
        "loss_delta": loss_delta,
        "max_output_abs_diff": output_diff,
        "max_input_grad_abs_diff": input_grad_diff,
        "max_param_grad_abs_diff": param_grad_diff,
        "thresholds": {
            "max_output_abs_diff": float(cfg.max_output_abs_diff),
            "max_loss_delta": float(cfg.max_loss_delta),
            "max_grad_abs_diff": float(cfg.max_grad_abs_diff),
        },
        "checks": checks,
        "compile_error": compile_error,
        "blocked_reasons": blockers,
        **dict(extra or {}),
    }


def _extract_tensor_output(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    sample = getattr(value, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError("compiled parity target must return a tensor or an object with tensor .sample")


def _max_abs_diff(left: Any, right: Any) -> float:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return float("inf")
    if tuple(left.shape) != tuple(right.shape):
        return float("inf")
    return float((left - right).abs().max().item())


def _max_param_grad_diff(left: Any, right: Any) -> float:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return float("inf")
    keys = set(left.keys()) | set(right.keys())
    if not keys:
        return 0.0
    return max(_max_abs_diff(left.get(key), right.get(key)) for key in keys)


def _config(config: DiTCompileParityProbeConfig | Mapping[str, Any] | None) -> DiTCompileParityProbeConfig:
    if isinstance(config, DiTCompileParityProbeConfig):
        cfg = config
    elif isinstance(config, Mapping):
        cfg = DiTCompileParityProbeConfig(**dict(config))
    else:
        cfg = DiTCompileParityProbeConfig()
    return cfg.normalized()


def _resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def _config_payload(
    cfg: DiTCompileParityProbeConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    return {
        "batch_size": int(cfg.batch_size),
        "token_count": int(cfg.token_count),
        "hidden_size": int(cfg.hidden_size),
        "num_heads": int(cfg.num_heads),
        "steps": int(cfg.steps),
        "seed": int(cfg.seed),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
    }


__all__ = [
    "build_dit_cached_shape_compile_parity_matrix",
    "DiTCompileParityProbeConfig",
    "TinyDiTCompileParityBlock",
    "normalize_orig_mod_state_dict_keys",
    "run_anima_native_tiny_per_block_compile_parity_probe",
    "run_dit_compile_parity_probe",
    "run_newbie_cached_shape_compile_parity_probe",
]

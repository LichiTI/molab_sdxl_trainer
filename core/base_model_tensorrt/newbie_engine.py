from __future__ import annotations

import importlib.util
import inspect
import time
from pathlib import Path
from typing import Any, Sequence

from .family_adapters import MODELS_ROOT
from .newbie_export import (
    NewbieStaticShape,
    create_newbie_synthetic_inputs,
    default_newbie_checkpoint,
    default_newbie_config_path,
    load_newbie_selective_wrapper,
    parse_layer_indices,
    summarize_tensor,
)
from .static_engine import StaticTensorRtEngine, build_static_tensorrt_engine, compare_tensor_outputs


NEWBIE_TAP_OUTPUT_NAMES = (
    "sample_out",
    "tap_layer_input",
    "tap_modulation",
    "tap_attn_in",
    "tap_attn_out",
    "tap_after_attn",
    "tap_ffn_in",
    "tap_ffn_hidden",
    "tap_ffn_out",
    "tap_layer_output",
)


def newbie_tap_output_names() -> tuple[str, ...]:
    return NEWBIE_TAP_OUTPUT_NAMES


def default_newbie_onnx_path(
    *,
    output_dir: str | Path = "",
    shape: NewbieStaticShape | None = None,
    layer_indices: str | Sequence[int] = (0,),
    opset: int = 18,
) -> Path:
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    root = Path(output_dir) if str(output_dir or "").strip() else MODELS_ROOT / "newbie" / "tensorrt_spike"
    name = f"newbie_transformer_{_layer_label(layers)}_{shape.latent_height}x{shape.latent_width}_tok{shape.tokens}_op{int(opset)}.onnx"
    return root / name


def default_newbie_engine_path(
    *,
    output_dir: str | Path = "",
    shape: NewbieStaticShape | None = None,
    layer_indices: str | Sequence[int] = (0,),
    opset: int = 18,
    precision: str = "fp32",
) -> Path:
    onnx_path = default_newbie_onnx_path(output_dir=output_dir, shape=shape, layer_indices=layer_indices, opset=opset)
    suffix = _precision_suffix(precision, default="fp32")
    return onnx_path.with_name(f"{onnx_path.stem}_{suffix}.engine")


def _precision_suffix(value: str | None, *, default: str) -> str:
    key = str(value or default).strip().lower().replace("-", "_")
    aliases = {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}
    return aliases.get(key, key)


def create_newbie_static_export_wrapper(
    model: Any,
    shape: NewbieStaticShape,
    *,
    tap_layer_index: int | None = None,
) -> Any:
    import torch

    shape.validate()

    class _NewbieStaticTensorRtExportWrapper(torch.nn.Module):
        def __init__(self, inner: Any):
            super().__init__()
            self.inner = inner
            self.tap_layer_index = tap_layer_index

        def _patchify(self, sample: Any) -> Any:
            batch = shape.batch
            patch_h = shape.latent_height // shape.patch_size
            patch_w = shape.latent_width // shape.patch_size
            patches = sample.reshape(
                batch,
                shape.latent_channels,
                patch_h,
                shape.patch_size,
                patch_w,
                shape.patch_size,
            )
            patches = patches.permute(0, 2, 4, 1, 3, 5)
            return patches.reshape(batch, patch_h * patch_w, shape.latent_channels * shape.patch_size * shape.patch_size)

        def _unpatchify(self, patch_values: Any) -> Any:
            batch = shape.batch
            patch_h = shape.latent_height // shape.patch_size
            patch_w = shape.latent_width // shape.patch_size
            patches = patch_values.reshape(
                batch,
                patch_h,
                patch_w,
                shape.latent_channels,
                shape.patch_size,
                shape.patch_size,
            )
            patches = patches.permute(0, 3, 1, 4, 2, 5)
            return patches.reshape(batch, shape.latent_channels, shape.latent_height, shape.latent_width)

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            text_embeds: Any,
        ) -> Any:
            x_embedder = getattr(self.inner, "x_embedder", None)
            if not isinstance(x_embedder, torch.nn.Linear):
                raise RuntimeError("Newbie static export requires linear x_embedder")
            x = x_embedder(self._patchify(sample))
            x = self.inner._bounded_rms(x, max_rms=8.0, clamp=64.0)

            conditioning_dim = self.inner._detect_conditioning_dim()
            time_cond = self.inner._build_timestep_condition(
                timestep=timestep,
                batch_size=shape.batch,
                conditioning_dim=conditioning_dim,
                dtype=sample.dtype,
                device=sample.device,
            )
            text_cond = text_embeds.float().to(dtype=sample.dtype).reshape(shape.batch, -1)
            clip_text_proj = self.inner._find_linear_leaf(getattr(self.inner, "clip_text_pooled_proj", None))
            if clip_text_proj is not None and text_cond.shape[-1] == clip_text_proj.in_features:
                text_cond = clip_text_proj(text_cond)
            elif text_cond.shape[-1] != conditioning_dim:
                proj = getattr(self.inner, "_text_embed_proj", None)
                if proj is not None and isinstance(proj, torch.nn.Linear):
                    text_cond = proj(text_cond)
                else:
                    text_cond = torch.zeros_like(time_cond)

            t_emb = time_cond
            time_text_embed = self.inner._find_linear_leaf(getattr(self.inner, "time_text_embed", None))
            if time_text_embed is not None:
                time_text_in = torch.cat((time_cond, text_cond), dim=-1)
                if time_text_in.shape[-1] > time_text_embed.in_features:
                    time_text_in = time_text_in[:, : time_text_embed.in_features]
                elif time_text_in.shape[-1] < time_text_embed.in_features:
                    time_text_in = torch.nn.functional.pad(time_text_in, (0, time_text_embed.in_features - time_text_in.shape[-1]))
                t_emb = time_text_embed(time_text_in)
            else:
                t_emb = time_cond + text_cond
            t_emb = self.inner._bounded_rms(t_emb, max_rms=8.0, clamp=64.0)

            enc = encoder_hidden_states.float().to(dtype=sample.dtype)
            if enc.dim() == 3:
                enc = enc.mean(dim=1)
            if enc.shape[-1] == shape.hidden_dim:
                x = x + enc.unsqueeze(1) * 0.01
                x = self.inner._bounded_rms(x, max_rms=8.0, clamp=64.0)

            tap_values: dict[str, Any] | None = None
            for block_index, block in enumerate(self.inner._block_modules if self.inner._block_modules else []):
                if self.tap_layer_index is not None and block_index == self.tap_layer_index:
                    x, tap_values = self._run_static_block(block, x, t_emb, collect_taps=True)
                else:
                    x = self._run_static_block(block, x, t_emb)

            final_layer = getattr(self.inner, "final_layer", None)
            if final_layer is not None:
                final_mod = self.inner._find_submodule(final_layer, "adaLN_modulation")
                final_mod_linear = self.inner._find_linear_leaf(final_mod) if final_mod is not None else None
                final_norm = self.inner._find_submodule(final_layer, "norm_final") or self.inner._find_submodule(final_layer, "norm")
                if final_norm is not None:
                    x = final_norm(x)
                if final_mod_linear is not None:
                    final_cond = final_mod_linear(t_emb)
                    if final_cond.shape[-1] == 2 * x.shape[-1]:
                        shift, scale = final_cond.chunk(2, dim=-1)
                        shift = self.inner._bounded_rms(shift, max_rms=4.0, clamp=16.0)
                        scale = torch.clamp(torch.nan_to_num(scale), min=-4.0, max=4.0)
                        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                    elif final_cond.shape[-1] == x.shape[-1]:
                        final_cond = torch.clamp(torch.nan_to_num(final_cond), min=-4.0, max=4.0)
                        x = x * (1 + final_cond.unsqueeze(1))
                    x = self.inner._bounded_rms(x, max_rms=8.0, clamp=64.0)
                final_linear = getattr(final_layer, "linear", None)
                if isinstance(final_linear, torch.nn.Linear):
                    x = final_linear(x)
            x = self._unpatchify(x)
            sample_out = self.inner._bounded_rms(x, max_rms=4.0, clamp=32.0)
            if self.tap_layer_index is None:
                return sample_out
            taps = tap_values or {}
            return (sample_out, *[taps.get(name, sample_out) for name in NEWBIE_TAP_OUTPUT_NAMES[1:]])

        def _run_static_block(self, block: Any, x: Any, t_emb: Any, collect_taps: bool = False) -> Any:
            residual = x
            hidden_dim = shape.hidden_dim
            batch = shape.batch
            tokens = shape.visual_tokens
            taps: dict[str, Any] = {"tap_layer_input": x}
            scale1 = gate1 = scale2 = gate2 = None
            ada_mod = self.inner._find_submodule(block, "adaLN_modulation")
            ada_linear = self.inner._find_linear_leaf(ada_mod) if ada_mod is not None else None
            if ada_linear is not None:
                modulation = ada_linear(t_emb)
                taps["tap_modulation"] = modulation
                if modulation.shape[-1] == 4 * hidden_dim:
                    scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
                    gate1 = gate1.tanh()
                    gate2 = gate2.tanh()
                elif modulation.shape[-1] == 6 * hidden_dim:
                    _shift1, scale1, gate1, _shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
                    gate1 = gate1.tanh()
                    gate2 = gate2.tanh()

            norm1 = self.inner._find_submodule(block, "attention_norm1") or self.inner._find_submodule(block, "norm1")
            h = norm1(x) if norm1 is not None else x
            if scale1 is not None:
                h = h * (1 + torch.clamp(torch.nan_to_num(scale1), min=-4.0, max=4.0).unsqueeze(1))
                h = self.inner._bounded_rms(h, max_rms=8.0, clamp=64.0)
            taps["tap_attn_in"] = h

            attn = self.inner._find_submodule(block, "attention") or self.inner._find_submodule(block, "attn")
            if attn is not None:
                qkv = self.inner._find_submodule(attn, "qkv")
                out = self.inner._find_submodule(attn, "out") or self.inner._find_submodule(attn, "to_out")
                if qkv is not None and out is not None:
                    qkv_out = qkv(h)
                    cfg = getattr(self.inner, "config", None)
                    num_heads = int(getattr(cfg, "n_heads", 24) or 24)
                    num_kv_heads = int(getattr(cfg, "n_kv_heads", 8) or 8)
                    head_dim = hidden_dim // max(num_heads, 1)
                    q_dim = num_heads * head_dim
                    kv_dim = num_kv_heads * head_dim
                    q, k, v = torch.split(qkv_out, [q_dim, kv_dim, kv_dim], dim=-1)
                    q = q.reshape(batch, tokens, num_heads, head_dim).transpose(1, 2)
                    k = k.reshape(batch, tokens, num_kv_heads, head_dim).transpose(1, 2)
                    v = v.reshape(batch, tokens, num_kv_heads, head_dim).transpose(1, 2)
                    if num_heads != num_kv_heads:
                        repeat = num_heads // max(num_kv_heads, 1)
                        k = k.repeat_interleave(repeat, dim=1)
                        v = v.repeat_interleave(repeat, dim=1)
                    q_norm = self.inner._find_submodule(attn, "q_norm")
                    k_norm = self.inner._find_submodule(attn, "k_norm")
                    if q_norm is not None:
                        q = q_norm(q)
                    if k_norm is not None:
                        k = k_norm(k)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * (float(head_dim) ** -0.5)
                    weights = torch.softmax(scores, dim=-1)
                    attn_out = torch.matmul(weights, v)
                    attn_out = attn_out.transpose(1, 2).reshape(batch, tokens, hidden_dim)
                    attn_out = out(attn_out)
                    if gate1 is not None:
                        attn_out = attn_out * gate1.unsqueeze(1)
                    attn_out = self.inner._bounded_rms(attn_out, max_rms=8.0, clamp=64.0)
                    taps["tap_attn_out"] = attn_out
                    x = residual + attn_out
                    x = self.inner._bounded_rms(x, max_rms=8.0, clamp=64.0)
            taps["tap_after_attn"] = x

            residual2 = x
            norm2 = self.inner._find_submodule(block, "ffn_norm1") or self.inner._find_submodule(block, "norm2")
            h = norm2(x) if norm2 is not None else x
            if scale2 is not None:
                h = h * (1 + torch.clamp(torch.nan_to_num(scale2), min=-4.0, max=4.0).unsqueeze(1))
                h = self.inner._bounded_rms(h, max_rms=8.0, clamp=64.0)
            taps["tap_ffn_in"] = h
            ff = self.inner._find_submodule(block, "feed_forward") or self.inner._find_submodule(block, "ff")
            if ff is not None:
                w1 = self.inner._find_submodule(ff, "w1")
                w2 = self.inner._find_submodule(ff, "w2")
                w3 = self.inner._find_submodule(ff, "w3")
                if w1 is not None and w2 is not None and w3 is not None:
                    h = torch.nn.functional.silu(w1(h)) * w3(h)
                    taps["tap_ffn_hidden"] = self.inner._bounded_rms(h, max_rms=8.0, clamp=64.0)
                    h = w2(h)
                    if gate2 is not None:
                        h = h * gate2.unsqueeze(1)
                    h = self.inner._bounded_rms(h, max_rms=8.0, clamp=64.0)
                    taps["tap_ffn_out"] = h
                    x = residual2 + h
                    x = self.inner._bounded_rms(x, max_rms=8.0, clamp=64.0)
            taps["tap_layer_output"] = x
            if collect_taps:
                fallback = x
                for name in NEWBIE_TAP_OUTPUT_NAMES[1:]:
                    taps.setdefault(name, fallback)
                return x, taps
            return x

    return _NewbieStaticTensorRtExportWrapper(model)


def export_newbie_static_onnx(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    output_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cpu",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    external_data: bool = False,
    tap_layer_index: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    phase = "prepare"
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_newbie_checkpoint(model_root)
    cfg_path = Path(config_path) if str(config_path or "").strip() else default_newbie_config_path(model_root)
    dst = Path(output_path) if str(output_path or "").strip() else default_newbie_onnx_path(
        output_dir=output_dir,
        shape=shape,
        layer_indices=layers,
        opset=opset,
    )
    try:
        model, selected_keys, target_device, normalized_dtype = load_newbie_selective_wrapper(
            checkpoint,
            config_path=cfg_path,
            layer_indices=layers,
            device=device,
            dtype_name=dtype_name,
        )
        wrapper = create_newbie_static_export_wrapper(model, shape, tap_layer_index=tap_layer_index).eval()
        inputs = create_newbie_synthetic_inputs(shape=shape, device=target_device, dtype_name=normalized_dtype, seed=seed)
        args = (inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"], inputs["text_embeds"])

        import torch

        phase = "torch_forward"
        with torch.no_grad():
            output = wrapper(*args)
        output_names = list(NEWBIE_TAP_OUTPUT_NAMES if tap_layer_index is not None else ("sample_out",))
        output_summary = _summarize_named_outputs(output_names, output) if tap_layer_index is not None else summarize_tensor(output)

        phase = "onnx_export"
        dst.parent.mkdir(parents=True, exist_ok=True)
        export_kwargs: dict[str, Any] = {
            "model": wrapper,
            "args": args,
            "f": str(dst),
            "input_names": ["sample", "timestep", "encoder_hidden_states", "text_embeds"],
            "output_names": output_names,
            "opset_version": max(13, int(opset or 18)),
            "do_constant_folding": True,
        }
        export_params = inspect.signature(torch.onnx.export).parameters
        if "dynamo" in export_params:
            export_kwargs["dynamo"] = False
        if "external_data" in export_params:
            export_kwargs["external_data"] = bool(external_data)
        with torch.no_grad():
            torch.onnx.export(**export_kwargs)

        phase = "onnx_check"
        check = _onnx_check(dst)
        success = bool(check.get("ok", True)) if check.get("available") else True
        return {
            "schema_version": 1,
            "kind": "newbie_tensorrt_static_onnx_export",
            "success": success,
            "checkpoint_path": str(checkpoint),
            "config_path": str(cfg_path),
            "onnx_path": str(dst),
            "bytes": dst.stat().st_size if dst.exists() else 0,
            "artifact_bytes": _onnx_artifact_bytes(dst),
            "layer_indices": list(layers),
            "selected_key_count": len(selected_keys),
            "device": str(target_device),
            "dtype": normalized_dtype,
            "shape": shape.to_dict(),
            "input_signature": shape.input_signature(),
            "external_data": bool(external_data),
            "tap_layer_index": tap_layer_index,
            "output_names": output_names,
            "torch_output": output_summary,
            "onnx_check": check,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:
        if strict:
            raise
        return {
            "schema_version": 1,
            "kind": "newbie_tensorrt_static_onnx_export",
            "success": False,
            "phase": phase,
            "checkpoint_path": str(checkpoint),
            "config_path": str(cfg_path),
            "onnx_path": str(dst),
            "layer_indices": list(layers),
            "shape": shape.to_dict(),
            "tap_layer_index": tap_layer_index,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def build_newbie_tensorrt_engine(
    *,
    onnx_path: str | Path = "",
    output_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    opset: int = 18,
    precision: str = "fp32",
    workspace_mb: int = 4096,
    fp32_layer_policy: str = "none",
    fp32_layer_name_filters: Sequence[str] | None = None,
) -> dict[str, Any]:
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    src = Path(onnx_path) if str(onnx_path or "").strip() else default_newbie_onnx_path(output_dir=output_dir, shape=shape, layer_indices=layers, opset=opset)
    dst = Path(output_path) if str(output_path or "").strip() else default_newbie_engine_path(output_dir=output_dir, shape=shape, layer_indices=layers, opset=opset, precision=precision)
    result = build_static_tensorrt_engine(
        onnx_path=src,
        output_path=dst,
        precision=precision,
        workspace_mb=workspace_mb,
        fp32_layer_policy=fp32_layer_policy,
        fp32_layer_name_filters=fp32_layer_name_filters,
    )
    result.update({"model_family": "newbie", "layer_indices": list(layers), "shape": shape.to_dict(), "input_signature": shape.input_signature()})
    return result


def compare_newbie_tensorrt_parity(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    engine_path: str | Path = "",
    output_dir: str | Path = "",
    layer_indices: str | Sequence[int] = (0,),
    shape: NewbieStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    precision: str = "fp32",
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape()
    layers = parse_layer_indices(layer_indices)
    checkpoint = Path(checkpoint_path) if str(checkpoint_path or "").strip() else default_newbie_checkpoint(model_root)
    cfg_path = Path(config_path) if str(config_path or "").strip() else default_newbie_config_path(model_root)
    engine = Path(engine_path) if str(engine_path or "").strip() else default_newbie_engine_path(output_dir=output_dir, shape=shape, layer_indices=layers, opset=opset, precision=precision)
    model, selected_keys, target_device, normalized_dtype = load_newbie_selective_wrapper(
        checkpoint,
        config_path=cfg_path,
        layer_indices=layers,
        device=device,
        dtype_name=dtype_name,
    )
    wrapper = create_newbie_static_export_wrapper(model, shape).eval()
    inputs = create_newbie_synthetic_inputs(shape=shape, device=target_device, dtype_name=normalized_dtype, seed=seed)
    args = (inputs["sample"], inputs["timestep"], inputs["encoder_hidden_states"], inputs["text_embeds"])

    import torch

    with torch.no_grad():
        torch_output = wrapper(*args)

    runtime = StaticTensorRtEngine(engine)
    trt_outputs = runtime.infer({
        "sample": inputs["sample"],
        "timestep": inputs["timestep"],
        "encoder_hidden_states": inputs["encoder_hidden_states"],
        "text_embeds": inputs["text_embeds"],
    })
    if "sample_out" not in trt_outputs:
        raise RuntimeError(f"TensorRT engine did not return sample_out, got {list(trt_outputs)}")
    trt_output = trt_outputs["sample_out"]
    comparison = compare_tensor_outputs(torch_output, trt_output)
    return {
        "schema_version": 1,
        "kind": "newbie_tensorrt_parity",
        "success": bool(comparison.get("same_shape")) and bool(comparison.get("all_finite", True)),
        "checkpoint_path": str(checkpoint),
        "config_path": str(cfg_path),
        "engine_path": str(engine),
        "layer_indices": list(layers),
        "selected_key_count": len(selected_keys),
        "device": str(target_device),
        "dtype": normalized_dtype,
        "shape": shape.to_dict(),
        "input_signature": shape.input_signature(),
        "torch_output": summarize_tensor(torch_output),
        "tensorrt_output": summarize_tensor(trt_output),
        "comparison": comparison,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _summarize_named_outputs(names: Sequence[str], output: Any) -> dict[str, Any]:
    values = output if isinstance(output, tuple) else (output,)
    return {name: summarize_tensor(values[index]) for index, name in enumerate(names[: len(values)])}


def _onnx_check(path: Path) -> dict[str, Any]:
    if importlib.util.find_spec("onnx") is None:
        return {"available": False, "ok": None}
    try:
        import onnx  # type: ignore

        onnx.checker.check_model(str(path))
        return {"available": True, "ok": True}
    except Exception as exc:
        return {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _onnx_artifact_bytes(path: Path) -> int:
    total = path.stat().st_size if path.exists() else 0
    if importlib.util.find_spec("onnx") is None:
        return total
    try:
        import onnx  # type: ignore

        model = onnx.load(str(path), load_external_data=False)
        locations: set[str] = set()
        for tensor in model.graph.initializer:
            for item in tensor.external_data:
                if item.key == "location" and item.value:
                    locations.add(str(item.value))
        for location in locations:
            data_path = path.parent / location
            if data_path.is_file():
                total += data_path.stat().st_size
    except Exception:
        pass
    return total


def _layer_label(layer_indices: Sequence[int]) -> str:
    layers = parse_layer_indices(layer_indices)
    if len(layers) == 1:
        return f"l{layers[0]}"
    if layers == tuple(range(layers[0], layers[-1] + 1)):
        return f"l{layers[0]}-{layers[-1]}"
    return "l" + "_".join(str(item) for item in layers)

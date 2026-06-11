"""Shape-only contracts for Lulynx image GGUF containers.

These checks are intentionally limited to tensor descriptors. They do not read
tensor payloads for computation, instantiate models, or claim runtime loading.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def inspect_image_gguf_shape_contract(*, component: str, family: str, tensors: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_map = {str(item["name"]): item for item in tensors}
    checks: list[dict[str, Any]] = []
    derived: dict[str, Any] = {}
    _apply_common_shape_checks(tensors, checks)
    if component == "vae":
        if family == "qwen_image_vae":
            _inspect_qwen_image_vae(tensor_map, checks, derived)
        else:
            _inspect_vae(tensor_map, checks, derived)
    elif component == "clip":
        _inspect_clip(tensor_map, checks, derived)
    elif component == "t5":
        _inspect_t5(tensor_map, checks, derived)
    elif component in {"sd15_unet", "sdxl_unet"}:
        _inspect_unet(tensor_map, checks, derived)
    elif component == "anima_dit":
        _inspect_anima_dit(tensor_map, checks, derived)
    elif component == "newbie_dit":
        _inspect_newbie_dit(tensor_map, checks, derived)
    issues = [str(check["message"]) for check in checks if not check["ok"]]
    return {
        "ok": not issues,
        "schema_version": 1,
        "loader": "python_reference_shape_loader",
        "mode": "shape_only",
        "component": component,
        "family": family,
        "tensor_count": len(tensors),
        "rank_counts": dict(sorted(Counter(str(item["rank"]) for item in tensors).items())),
        "tensor_type_counts": dict(sorted(Counter(str(item["tensor_type"]) for item in tensors).items())),
        "derived": derived,
        "checks": checks[:120],
        "issues": issues,
    }


def _inspect_vae(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _VAE_RANKS)
    encoder_in = tensor_map.get("encoder.conv_in.weight") or tensor_map.get("conv1.weight")
    encoder_out = tensor_map.get("encoder.conv_out.weight")
    decoder_in = tensor_map.get("decoder.conv_in.weight")
    decoder_out = tensor_map.get("decoder.conv_out.weight") or tensor_map.get("conv2.weight")
    if _rank(encoder_in) == 4:
        derived["input_channels"] = _shape(encoder_in)[1]
    if _rank(decoder_out) == 4:
        derived["output_channels"] = _shape(decoder_out)[0]
    if _rank(decoder_in) == 4:
        derived["latent_channels"] = _shape(decoder_in)[1]
    if _rank(encoder_out) == 4:
        derived["encoder_out_channels"] = _shape(encoder_out)[0]
    if _rank(encoder_in) == 4 and _rank(decoder_out) == 4:
        _add_check(
            checks,
            "vae.pixel_channels",
            "encoder input channels match decoder output channels",
            _shape(encoder_in)[1] == _shape(decoder_out)[0],
            [_shape(encoder_in)[1], _shape(decoder_out)[0]],
        )
    if _rank(decoder_in) == 4 and _rank(encoder_out) == 4:
        latent_channels = _shape(decoder_in)[1]
        encoder_channels = _shape(encoder_out)[0]
        _add_check(
            checks,
            "vae.latent_channels",
            "encoder output channels are latent channels or mean/logvar pair",
            encoder_channels in {latent_channels, latent_channels * 2},
            [encoder_channels, latent_channels],
        )
    quant_conv = tensor_map.get("quant_conv.weight")
    post_quant_conv = tensor_map.get("post_quant_conv.weight")
    if _rank(quant_conv) == 4 and _rank(encoder_out) == 4:
        _add_check(
            checks,
            "quant_conv.weight",
            "quant conv input channels match encoder output channels",
            _shape(quant_conv)[1] == _shape(encoder_out)[0],
            _shape(quant_conv),
        )
    if _rank(post_quant_conv) == 4 and _rank(decoder_in) == 4:
        _add_check(
            checks,
            "post_quant_conv.weight",
            "post quant conv output channels match decoder latent channels",
            _shape(post_quant_conv)[0] == _shape(decoder_in)[1],
            _shape(post_quant_conv),
        )


def _inspect_qwen_image_vae(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _QWEN_IMAGE_VAE_RANKS)
    conv1 = tensor_map.get("conv1.weight")
    conv2 = tensor_map.get("conv2.weight")
    encoder_res = tensor_map.get("encoder.downsamples.0.residual.2.weight")
    decoder_res = tensor_map.get("decoder.upsamples.0.residual.2.weight")
    qkv = tensor_map.get("decoder.middle.1.to_qkv.weight")
    if _rank(conv1) == 5:
        shape = _shape(conv1)
        derived["temporal_latent_channels"] = shape[1]
        derived["conv1_out_channels"] = shape[0]
        derived["conv1_kernel"] = shape[2:]
    if _rank(conv2) == 5:
        shape = _shape(conv2)
        derived["output_channels"] = shape[0]
        derived["conv2_in_channels"] = shape[1]
        derived["conv2_kernel"] = shape[2:]
    if _rank(encoder_res) == 5:
        derived["encoder_base_channels"] = _shape(encoder_res)[0]
    if _rank(decoder_res) == 5:
        derived["decoder_base_channels"] = _shape(decoder_res)[0]
    if _rank(qkv) == 4:
        shape = _shape(qkv)
        derived["middle_attention_qkv_dim"] = shape[0]
        derived["middle_attention_hidden_dim"] = shape[1]
        _add_check(checks, "decoder.middle.1.to_qkv.weight", "packed qkv output is divisible by 3", shape[0] % 3 == 0, shape)


def _inspect_clip(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _CLIP_RANKS)
    token_embedding = tensor_map.get("text_model.embeddings.token_embedding.weight") or tensor_map.get(
        "model.embeddings.word_embeddings.weight"
    )
    hidden_dim = _shape(token_embedding)[1] if _rank(token_embedding) == 2 else 0
    if hidden_dim:
        derived["hidden_dim"] = hidden_dim
        derived["vocab_size"] = _shape(token_embedding)[0]
    position_embedding = tensor_map.get("text_model.embeddings.position_embedding.weight")
    if hidden_dim and _rank(position_embedding) == 2:
        derived["max_positions"] = _shape(position_embedding)[0]
        _add_check(
            checks,
            "text_model.embeddings.position_embedding.weight",
            "position embedding hidden dim matches token embedding",
            _shape(position_embedding)[1] == hidden_dim,
            _shape(position_embedding),
        )
    for key in (
        "text_model.final_layer_norm.weight",
        "model.emb_ln.weight",
    ):
        item = tensor_map.get(key)
        if hidden_dim and _rank(item) == 1:
            _add_check(checks, key, "normalization width matches hidden dim", _shape(item)[0] == hidden_dim, _shape(item))
    _check_square_projection_group(
        tensor_map,
        checks,
        hidden_dim=hidden_dim,
        keys=(
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            "text_model.encoder.layers.0.self_attn.k_proj.weight",
            "text_model.encoder.layers.0.self_attn.v_proj.weight",
            "text_model.encoder.layers.0.self_attn.out_proj.weight",
        ),
        label="clip.self_attn",
    )
    _check_mlp_pair(
        tensor_map,
        checks,
        hidden_dim=hidden_dim,
        up_key="text_model.encoder.layers.0.mlp.fc1.weight",
        down_key="text_model.encoder.layers.0.mlp.fc2.weight",
        label="clip.mlp",
    )
    qkv = tensor_map.get("model.encoder.layers.0.mixer.Wqkv.weight")
    if hidden_dim and _rank(qkv) == 2:
        shape = _shape(qkv)
        ok = (shape[0] == hidden_dim * 3 and shape[1] == hidden_dim) or (shape[0] == hidden_dim and shape[1] == hidden_dim * 3)
        _add_check(checks, "model.encoder.layers.0.mixer.Wqkv.weight", "packed qkv shape is 3x hidden", ok, shape)
    out_proj = tensor_map.get("model.encoder.layers.0.mixer.out_proj.weight")
    if hidden_dim and _rank(out_proj) == 2:
        _add_check(checks, "model.encoder.layers.0.mixer.out_proj.weight", "output projection is hidden x hidden", _shape(out_proj) == [hidden_dim, hidden_dim], _shape(out_proj))


def _inspect_t5(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _T5_RANKS)
    shared = tensor_map.get("shared.weight")
    hidden_dim = _shape(shared)[1] if _rank(shared) == 2 else 0
    if hidden_dim:
        derived["hidden_dim"] = hidden_dim
        derived["vocab_size"] = _shape(shared)[0]
    q = tensor_map.get("encoder.block.0.layer.0.SelfAttention.q.weight")
    k = tensor_map.get("encoder.block.0.layer.0.SelfAttention.k.weight")
    v = tensor_map.get("encoder.block.0.layer.0.SelfAttention.v.weight")
    qkv = [item for item in (q, k, v) if _rank(item) == 2]
    if hidden_dim:
        for key, item in (
            ("encoder.block.0.layer.0.SelfAttention.q.weight", q),
            ("encoder.block.0.layer.0.SelfAttention.k.weight", k),
            ("encoder.block.0.layer.0.SelfAttention.v.weight", v),
        ):
            if _rank(item) == 2:
                _add_check(checks, key, "attention input dim matches hidden dim", _shape(item)[1] == hidden_dim, _shape(item))
    if len(qkv) == 3:
        inner_dims = [_shape(item)[0] for item in qkv]
        derived["attention_inner_dim"] = inner_dims[0]
        _add_check(checks, "t5.self_attention.qkv", "q/k/v output dims match", len(set(inner_dims)) == 1, inner_dims)
    out = tensor_map.get("encoder.block.0.layer.0.SelfAttention.o.weight")
    if hidden_dim and _rank(out) == 2:
        expected_inner = derived.get("attention_inner_dim")
        _add_check(checks, "encoder.block.0.layer.0.SelfAttention.o.weight", "attention output maps back to hidden dim", _shape(out)[0] == hidden_dim, _shape(out))
        if isinstance(expected_inner, int):
            _add_check(checks, "encoder.block.0.layer.0.SelfAttention.o.weight", "attention output input dim matches qkv inner dim", _shape(out)[1] == expected_inner, _shape(out))
    wo = tensor_map.get("encoder.block.0.layer.1.DenseReluDense.wo.weight")
    if hidden_dim and _rank(wo) == 2:
        derived["feed_forward_dim"] = _shape(wo)[1]
        _add_check(checks, "encoder.block.0.layer.1.DenseReluDense.wo.weight", "feed-forward output maps back to hidden dim", _shape(wo)[0] == hidden_dim, _shape(wo))
    norm = tensor_map.get("encoder.final_layer_norm.weight")
    if hidden_dim and _rank(norm) == 1:
        _add_check(checks, "encoder.final_layer_norm.weight", "final norm width matches hidden dim", _shape(norm)[0] == hidden_dim, _shape(norm))


def _inspect_unet(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _UNET_RANKS)
    input_weight = tensor_map.get("model.diffusion_model.input_blocks.0.0.weight")
    output_weight = tensor_map.get("model.diffusion_model.out.2.weight")
    time_weight = tensor_map.get("model.diffusion_model.time_embed.0.weight")
    if _rank(input_weight) == 4:
        derived["latent_channels"] = _shape(input_weight)[1]
        derived["input_block_channels"] = _shape(input_weight)[0]
    if _rank(output_weight) == 4:
        derived["output_channels"] = _shape(output_weight)[0]
        derived["output_input_channels"] = _shape(output_weight)[1]
    if _rank(time_weight) == 2:
        derived["time_embed_dim"] = _shape(time_weight)[0]
        derived["time_embed_input_dim"] = _shape(time_weight)[1]
    if _rank(input_weight) == 4 and _rank(output_weight) == 4:
        _add_check(
            checks,
            "model.diffusion_model.out.2.weight",
            "UNet output channels match latent input channels",
            _shape(output_weight)[0] == _shape(input_weight)[1],
            _shape(output_weight),
        )
    if _rank(input_weight) == 4 and _rank(time_weight) == 2:
        _add_check(
            checks,
            "model.diffusion_model.time_embed.0.weight",
            "time embedding input dim matches first block channels",
            _shape(time_weight)[1] == _shape(input_weight)[0],
            _shape(time_weight),
        )
    _check_unet_cross_attention(tensor_map, checks)


def _inspect_anima_dit(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _ANIMA_DIT_RANKS)
    x_embedder = tensor_map.get("net.x_embedder.proj.1.weight")
    final_linear = tensor_map.get("net.final_layer.linear.weight")
    hidden_dim = _shape(x_embedder)[0] if _rank(x_embedder) == 2 else 0
    if not hidden_dim and _rank(final_linear) == 2:
        hidden_dim = _shape(final_linear)[1]
    if hidden_dim:
        derived["hidden_dim"] = hidden_dim
    _check_square_projection_group(
        tensor_map,
        checks,
        hidden_dim=hidden_dim,
        keys=(
            "net.blocks.0.self_attn.q_proj.weight",
            "net.blocks.0.self_attn.k_proj.weight",
            "net.blocks.0.self_attn.v_proj.weight",
            "net.blocks.0.self_attn.output_proj.weight",
        ),
        label="anima.self_attn",
    )
    _check_mlp_pair(
        tensor_map,
        checks,
        hidden_dim=hidden_dim,
        up_key="net.blocks.0.mlp.layer1.weight",
        down_key="net.blocks.0.mlp.layer2.weight",
        label="anima.mlp",
    )
    if hidden_dim and _rank(final_linear) == 2:
        _add_check(checks, "net.final_layer.linear.weight", "final linear input dim matches hidden dim", _shape(final_linear)[1] == hidden_dim, _shape(final_linear))
    cross_q = tensor_map.get("net.blocks.0.cross_attn.q_proj.weight")
    if hidden_dim and _rank(cross_q) == 2:
        _add_check(checks, "net.blocks.0.cross_attn.q_proj.weight", "cross-attention q projection uses hidden dim", _shape(cross_q) == [hidden_dim, hidden_dim], _shape(cross_q))


def _inspect_newbie_dit(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], derived: dict[str, Any]) -> None:
    _apply_rank_checks(tensor_map, checks, _NEWBIE_DIT_RANKS)
    x_embedder = tensor_map.get("x_embedder.weight")
    final_linear = tensor_map.get("final_layer.linear.weight")
    hidden_dim = _shape(x_embedder)[0] if _rank(x_embedder) == 4 else 0
    if not hidden_dim and _rank(final_linear) == 2:
        hidden_dim = _shape(final_linear)[1]
    if hidden_dim:
        derived["hidden_dim"] = hidden_dim
        if _rank(x_embedder) == 4:
            derived["latent_channels"] = _shape(x_embedder)[1]
            derived["patch_size"] = [_shape(x_embedder)[2], _shape(x_embedder)[3]]
    qkv = tensor_map.get("layers.0.attention.qkv.weight")
    if hidden_dim and _rank(qkv) == 2:
        shape = _shape(qkv)
        _add_check(checks, "layers.0.attention.qkv.weight", "packed qkv shape is 3x hidden", shape == [hidden_dim * 3, hidden_dim], shape)
    out = tensor_map.get("layers.0.attention.out.weight")
    if hidden_dim and _rank(out) == 2:
        _add_check(checks, "layers.0.attention.out.weight", "attention output projection is hidden x hidden", _shape(out) == [hidden_dim, hidden_dim], _shape(out))
    _check_newbie_mlp(tensor_map, checks, hidden_dim=hidden_dim)
    if hidden_dim and _rank(final_linear) == 2:
        _add_check(checks, "final_layer.linear.weight", "final linear input dim matches hidden dim", _shape(final_linear)[1] == hidden_dim, _shape(final_linear))


def _check_square_projection_group(
    tensor_map: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
    *,
    hidden_dim: int,
    keys: tuple[str, ...],
    label: str,
) -> None:
    present = [(key, tensor_map.get(key)) for key in keys if _rank(tensor_map.get(key)) == 2]
    if not hidden_dim or not present:
        return
    for key, item in present:
        _add_check(checks, key, "projection is hidden x hidden", _shape(item) == [hidden_dim, hidden_dim], _shape(item))
    if len(present) >= 3:
        shapes = [_shape(item) for _, item in present]
        _add_check(checks, label, "projection shapes match within block", len({tuple(shape) for shape in shapes}) == 1, [])


def _check_mlp_pair(
    tensor_map: dict[str, dict[str, Any]],
    checks: list[dict[str, Any]],
    *,
    hidden_dim: int,
    up_key: str,
    down_key: str,
    label: str,
) -> None:
    up = tensor_map.get(up_key)
    down = tensor_map.get(down_key)
    if hidden_dim and _rank(up) == 2:
        _add_check(checks, up_key, "MLP up projection input dim matches hidden dim", _shape(up)[1] == hidden_dim, _shape(up))
    if hidden_dim and _rank(down) == 2:
        _add_check(checks, down_key, "MLP down projection output dim matches hidden dim", _shape(down)[0] == hidden_dim, _shape(down))
    if _rank(up) == 2 and _rank(down) == 2:
        _add_check(checks, label, "MLP intermediate dims match", _shape(up)[0] == _shape(down)[1], [_shape(up)[0], _shape(down)[1]])


def _check_newbie_mlp(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], *, hidden_dim: int) -> None:
    w1 = tensor_map.get("layers.0.feed_forward.w1.weight")
    w2 = tensor_map.get("layers.0.feed_forward.w2.weight")
    w3 = tensor_map.get("layers.0.feed_forward.w3.weight")
    if hidden_dim:
        for key, item in (
            ("layers.0.feed_forward.w1.weight", w1),
            ("layers.0.feed_forward.w3.weight", w3),
        ):
            if _rank(item) == 2:
                _add_check(checks, key, "feed-forward input dim matches hidden dim", _shape(item)[1] == hidden_dim, _shape(item))
        if _rank(w2) == 2:
            _add_check(checks, "layers.0.feed_forward.w2.weight", "feed-forward output dim matches hidden dim", _shape(w2)[0] == hidden_dim, _shape(w2))
    if _rank(w1) == 2 and _rank(w2) == 2:
        _add_check(checks, "newbie.feed_forward.w1_w2", "w1/w2 intermediate dims match", _shape(w1)[0] == _shape(w2)[1], [_shape(w1)[0], _shape(w2)[1]])
    if _rank(w3) == 2 and _rank(w2) == 2:
        _add_check(checks, "newbie.feed_forward.w3_w2", "w3/w2 intermediate dims match", _shape(w3)[0] == _shape(w2)[1], [_shape(w3)[0], _shape(w2)[1]])


def _check_unet_cross_attention(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    q_suffix = "attn2.to_q.weight"
    prefixes = sorted(name[: -len(q_suffix)] for name in tensor_map if name.endswith(q_suffix))[:2]
    for prefix in prefixes:
        q = tensor_map.get(prefix + "attn2.to_q.weight")
        k = tensor_map.get(prefix + "attn2.to_k.weight")
        v = tensor_map.get(prefix + "attn2.to_v.weight")
        if not (_rank(q) == 2 and _rank(k) == 2 and _rank(v) == 2):
            continue
        q_shape = _shape(q)
        k_shape = _shape(k)
        v_shape = _shape(v)
        _add_check(checks, prefix + "attn2", "cross-attention q/k/v output dims match", len({q_shape[0], k_shape[0], v_shape[0]}) == 1, [q_shape[0], k_shape[0], v_shape[0]])
        _add_check(checks, prefix + "attn2", "cross-attention k/v context dims match", k_shape[1] == v_shape[1], [k_shape[1], v_shape[1]])


def _apply_common_shape_checks(tensors: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    for item in tensors:
        shape = _shape(item)
        ok = all(int(dim) > 0 for dim in shape) and int(item["numel"]) == _numel(shape)
        _add_check(checks, str(item["name"]), "positive dims and numel match logical_shape", ok, shape)


def _apply_rank_checks(tensor_map: dict[str, dict[str, Any]], checks: list[dict[str, Any]], rules: dict[str, set[int]]) -> None:
    for key, ranks in rules.items():
        item = tensor_map.get(key)
        if item is None:
            continue
        ok = int(item["rank"]) in ranks
        _add_check(checks, key, f"rank in {sorted(ranks)}", ok, _shape(item))


def _add_check(checks: list[dict[str, Any]], key: str, rule: str, ok: bool, shape: list[int]) -> None:
    checks.append(
        {
            "key": key,
            "rule": rule,
            "ok": bool(ok),
            "logical_shape": list(shape),
            "message": "" if ok else f"{key} failed shape rule: {rule}; got {shape}",
        }
    )


def _rank(item: dict[str, Any] | None) -> int:
    if not item:
        return 0
    try:
        return int(item.get("rank") or len(item.get("logical_shape") or []))
    except Exception:
        return 0


def _shape(item: dict[str, Any] | None) -> list[int]:
    if not item:
        return []
    try:
        return [int(dim) for dim in item.get("logical_shape") or []]
    except Exception:
        return []


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


_VAE_RANKS = {
    "encoder.conv_in.weight": {4},
    "encoder.conv_out.weight": {4},
    "decoder.conv_in.weight": {4},
    "decoder.conv_out.weight": {4},
    "decoder.mid_block.attentions.0.to_q.weight": {2},
    "conv1.weight": {4},
    "conv2.weight": {4},
}
_QWEN_IMAGE_VAE_RANKS = {
    "conv1.weight": {5},
    "conv2.weight": {5},
    "encoder.downsamples.0.residual.2.weight": {5},
    "decoder.upsamples.0.residual.2.weight": {5},
    "decoder.middle.1.to_qkv.weight": {4},
}
_CLIP_RANKS = {
    "text_model.embeddings.token_embedding.weight": {2},
    "text_model.embeddings.position_embedding.weight": {2},
    "text_model.encoder.layers.0.self_attn.q_proj.weight": {2},
    "text_model.encoder.layers.0.self_attn.k_proj.weight": {2},
    "text_model.encoder.layers.0.self_attn.v_proj.weight": {2},
    "text_model.final_layer_norm.weight": {1},
    "model.embeddings.word_embeddings.weight": {2},
    "model.embeddings.token_type_embeddings.weight": {2},
    "model.encoder.layers.0.mixer.Wqkv.weight": {2},
    "model.emb_ln.weight": {1},
}
_T5_RANKS = {
    "shared.weight": {2},
    "encoder.block.0.layer.0.SelfAttention.q.weight": {2},
    "encoder.block.0.layer.0.SelfAttention.k.weight": {2},
    "encoder.block.0.layer.0.SelfAttention.v.weight": {2},
    "encoder.block.0.layer.1.DenseReluDense.wo.weight": {2},
    "encoder.final_layer_norm.weight": {1},
}
_UNET_RANKS = {
    "model.diffusion_model.input_blocks.0.0.weight": {4},
    "model.diffusion_model.time_embed.0.weight": {2},
    "model.diffusion_model.middle_block.0.in_layers.0.weight": {1},
    "model.diffusion_model.output_blocks.0.0.in_layers.0.weight": {1},
    "model.diffusion_model.out.2.weight": {4},
}
_ANIMA_DIT_RANKS = {
    "net.x_embedder.proj.1.weight": {2},
    "net.final_layer.linear.weight": {2},
    "net.blocks.0.self_attn.q_proj.weight": {2},
    "net.blocks.0.self_attn.k_proj.weight": {2},
    "net.blocks.0.self_attn.v_proj.weight": {2},
    "net.blocks.0.self_attn.output_proj.weight": {2},
    "net.blocks.0.cross_attn.q_proj.weight": {2},
    "net.blocks.0.mlp.layer1.weight": {2},
    "net.blocks.0.mlp.layer2.weight": {2},
}
_NEWBIE_DIT_RANKS = {
    "x_embedder.weight": {4},
    "final_layer.linear.weight": {2},
    "layers.0.attention.qkv.weight": {2},
    "layers.0.attention.out.weight": {2},
    "layers.0.feed_forward.w1.weight": {2},
    "layers.0.feed_forward.w2.weight": {2},
    "layers.0.feed_forward.w3.weight": {2},
}


__all__ = ["inspect_image_gguf_shape_contract"]

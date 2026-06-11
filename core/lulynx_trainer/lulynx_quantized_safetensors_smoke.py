"""Smoke tests for Lulynx quantized safetensors round-trips."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.safetensors_loader import load_safetensors, open_safetensors  # noqa: E402
from core.lulynx_trainer.lulynx_quantized_safetensors import (
    QUANT_PREFIX,
    FORMAT_KEY,
    SCHEMA_KEY,
    SCHEMA_VERSION,
    TENSORS_KEY,
    QuantizedTensorEntry,
    parse_quantized_tensor_entries,
)  # noqa: E402
from core.tools.model_quantization_validator import validate_quantized_model_file  # noqa: E402
from core.tools.model_quantizer import quantize_model_file  # noqa: E402
from lulynx_launcher.services.toolbox_runner import run_action  # noqa: E402


def _write_source(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    state = {
        "linear.weight": torch.randn(8, 17, dtype=torch.float32),
        "q8.weight": torch.randn(2, 32, dtype=torch.float32),
        "conv.weight": torch.randn(4, 3, 3, 3, dtype=torch.float32),
        "linear.bias": torch.randn(8, dtype=torch.float32),
        "steps": torch.arange(3, dtype=torch.int64),
    }
    save_file(state, path, metadata={"trainer": "lulynx"})
    return state


def _write_hf_llama_source(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import save_file

    hidden = 32
    intermediate = 64
    vocab = 64
    state = {
        "model.embed_tokens.weight": torch.randn(vocab, hidden, dtype=torch.float32),
        "model.norm.weight": torch.randn(hidden, dtype=torch.float32),
        "lm_head.weight": torch.randn(vocab, hidden, dtype=torch.float32),
        "model.layers.0.input_layernorm.weight": torch.randn(hidden, dtype=torch.float32),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(hidden, dtype=torch.float32),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(hidden, hidden, dtype=torch.float32),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(hidden, hidden, dtype=torch.float32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(hidden, hidden, dtype=torch.float32),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden, hidden, dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(intermediate, hidden, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden, intermediate, dtype=torch.float32),
        "model.layers.0.mlp.up_proj.weight": torch.randn(intermediate, hidden, dtype=torch.float32),
    }
    save_file(state, path, metadata={"trainer": "lulynx"})
    return state


def _assert_decoded_matches_contract(path: str, reference: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
    loaded = load_safetensors(path, disable_mmap=False)
    assert set(loaded) == set(reference)
    assert loaded["linear.weight"].shape == reference["linear.weight"].shape
    assert loaded["conv.weight"].shape == reference["conv.weight"].shape
    assert loaded["linear.weight"].dtype == dtype
    assert loaded["conv.weight"].dtype == dtype
    assert loaded["linear.bias"].dtype == reference["linear.bias"].dtype
    with open_safetensors(path, disable_mmap=False) as handle:
        assert set(handle.keys()) == set(reference)
        assert handle.get_slice("linear.weight").get_shape() == list(reference["linear.weight"].shape)
        assert handle.get_tensor("linear.weight").dtype == dtype


def test_plain_fp16_quantize() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "fp16.safetensors")
        reference = _write_source(src)
        result = quantize_model_file(src, dst, "fp16")
        assert result["success"] is True
        assert result["quant_format"] == "fp16"
        assert result["validation"]["ok"] is True, result["validation"]
        assert result["validation"]["trainer_loader_compatible"] is True, result["validation"]
        loaded = load_safetensors(dst)
        assert loaded["linear.weight"].dtype == torch.float16
        assert loaded["linear.weight"].shape == reference["linear.weight"].shape
        print("PASS: fp16 quantization writes plain safetensors")


def test_plain_fp8_quantize_when_available() -> None:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        print("SKIP: torch.float8_e4m3fn is not available")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "fp8.safetensors")
        reference = _write_source(src)
        result = quantize_model_file(src, dst, "fp8_e4m3fn")
        assert result["success"] is True
        assert result["quant_format"] == "fp8_e4m3fn"
        assert result["validation"]["ok"] is True, result["validation"]
        loaded = load_safetensors(dst)
        assert loaded["linear.weight"].dtype == fp8_dtype
        assert loaded["linear.weight"].shape == reference["linear.weight"].shape
        print("PASS: fp8_e4m3fn quantization writes plain safetensors")


def test_lulynx_int8_rowwise_quantize() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "int8.safetensors")
        reference = _write_source(src)
        result = quantize_model_file(src, dst, "lulynx_int8_rowwise", decode_dtype="bf16")
        assert result["converted_tensors"] == 3
        assert "rowwise_provider" in result
        assert "native_converted_tensors" in result
        assert result["validation"]["ok"] is True, result["validation"]
        assert result["validation"]["metadata_format"] == "lulynx_int8_rowwise", result["validation"]
        assert result["validation"]["quantized_entry_count"] == 3, result["validation"]
        _assert_decoded_matches_contract(dst, reference, torch.bfloat16)
        print("PASS: Lulynx int8 rowwise decodes through safetensors loader")


def test_lulynx_uint4_rowwise_quantize() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "uint4.safetensors")
        reference = _write_source(src)
        result = quantize_model_file(src, dst, "lulynx_uint4_rowwise", decode_dtype="fp16")
        assert result["converted_tensors"] == 3
        assert "rowwise_provider" in result
        assert "native_converted_tensors" in result
        assert result["validation"]["ok"] is True, result["validation"]
        assert result["validation"]["metadata_format"] == "lulynx_uint4_rowwise", result["validation"]
        assert result["validation"]["quantized_entry_count"] == 3, result["validation"]
        from safetensors import safe_open

        with safe_open(dst, framework="pt", device="cpu") as handle:
            entries = parse_quantized_tensor_entries(dict(handle.metadata() or {}))
        assert entries and all(entry.offset_key for entry in entries), entries
        assert {entry.quantization_variant for entry in entries} == {"affine_uint4_blockwise_v2"}, entries
        _assert_decoded_matches_contract(dst, reference, torch.float16)
        print("PASS: Lulynx uint4 rowwise decodes through safetensors loader")


def test_legacy_symmetric_uint4_rowwise_still_decodes() -> None:
    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        dst = os.path.join(tmp, "legacy-uint4.safetensors")
        q = torch.tensor([[8, 9, 7, 15]], dtype=torch.uint8)
        scale = torch.tensor([[0.5]], dtype=torch.float16)
        entry = QuantizedTensorEntry(
            key="linear.weight",
            format="lulynx_uint4_rowwise",
            shape=[2, 4],
            original_dtype="float32",
            q_key=f"{QUANT_PREFIX}.0.q",
            scale_key=f"{QUANT_PREFIX}.0.scale",
            decode_dtype="fp16",
            original_cols=8,
        )
        save_file(
            {entry.q_key: q, entry.scale_key: scale},
            dst,
            metadata={
                SCHEMA_KEY: SCHEMA_VERSION,
                FORMAT_KEY: "lulynx_uint4_rowwise",
                TENSORS_KEY: __import__("json").dumps([entry.to_dict()], separators=(",", ":")),
            },
        )
        loaded = load_safetensors(dst)
        assert set(loaded) == {"linear.weight"}, loaded
        assert loaded["linear.weight"].shape == (2, 4), loaded["linear.weight"].shape
        assert loaded["linear.weight"].dtype == torch.float16
        print("PASS: legacy symmetric uint4 rowwise still decodes")


def test_quantization_validator_detects_wrong_format() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "fp16.safetensors")
        _write_source(src)
        quantize_model_file(src, dst, "fp16")
        report = validate_quantized_model_file(src, dst, "lulynx_int8_rowwise")
        assert report["ok"] is False, report
        assert "rowwise_metadata_schema" in report["failed_checks"], report
        assert report["trainer_loader_compatible"] is True, report
        print("PASS: quantization validator detects format/metadata mismatch")


def test_toolbox_runner_model_quantize_returns_validation_report() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "toolbox-int8.safetensors")
        _write_source(src)
        result = run_action(
            "model_quantize",
            {
                "input_path": src,
                "output_path": dst,
                "quant_format": "lulynx_int8_rowwise",
                "decode_dtype": "fp16",
            },
        )
        assert result["success"] is True, result
        assert result["quant_format"] == "lulynx_int8_rowwise", result
        assert Path(result["output_path"]).is_file(), result
        validation = result.get("validation")
        assert isinstance(validation, dict), result
        assert validation["ok"] is True, validation
        assert validation["trainer_loader_compatible"] is True, validation
        assert validation["metadata_format"] == "lulynx_int8_rowwise", validation
        assert validation["quantized_entry_count"] == 3, validation
        print("PASS: toolbox runner model_quantize returns validation report")


def test_gguf_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            src = os.path.join(tmp, "source.safetensors")
            dst = os.path.join(tmp, "model.gguf")
            _write_source(src)
            try:
                quantize_model_file(src, dst, "gguf_f16")
            except RuntimeError as exc:
                assert "GGUF export requires" in str(exc)
                print("PASS: GGUF export reports missing gguf dependency")
                return
            raise AssertionError("GGUF export should require gguf when the module is missing")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "model.gguf")
        _write_source(src)
        result = quantize_model_file(src, dst, "gguf_f16", gguf_arch="generic", gguf_name="lulynx-smoke")
        assert result["success"] is True
        assert result["quant_format"] == "gguf_f16"
        assert result["output_format"] == "gguf"
        assert result["converted_tensors"] == 4
        assert result["skipped_tensors"] == 1
        assert os.path.getsize(dst) > 0
        reader = gguf.GGUFReader(dst)
        tensor_names = {str(tensor.name) for tensor in reader.tensors}
        assert {"linear.weight", "q8.weight", "conv.weight", "linear.bias", "steps"}.issubset(tensor_names)
        print("PASS: GGUF f16 export writes a readable GGUF file")


def test_gguf_q8_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: gguf_q8_0 export requires gguf")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "model-q8.gguf")
        _write_source(src)
        result = quantize_model_file(src, dst, "gguf_q8_0", gguf_arch="generic", gguf_name="lulynx-smoke-q8")
        assert result["success"] is True
        assert result["quant_format"] == "gguf_q8_0"
        assert result["gguf_provider"] == "python"
        assert result["gguf_quant_type"] == "Q8_0"
        assert result["converted_tensors"] == 1
        assert result["skipped_tensors"] == 4
        reader = gguf.GGUFReader(dst)
        tensor_types = {str(tensor.name): tensor.tensor_type for tensor in reader.tensors}
        assert tensor_types["q8.weight"] == gguf.GGMLQuantizationType.Q8_0
        assert tensor_types["linear.weight"] == gguf.GGMLQuantizationType.F32
        print("PASS: GGUF q8_0 export writes true Q8_0 tensors when shape permits")


def test_hf_llama_gguf_export_or_missing_dependency() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: GGUF llama export requires gguf")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "llama.safetensors")
        dst = os.path.join(tmp, "llama.gguf")
        _write_hf_llama_source(src)
        result = quantize_model_file(
            src,
            dst,
            "gguf_f16",
            gguf_arch="llama",
            gguf_name="lulynx-llama-smoke",
            gguf_metadata={"head_count": 4, "head_count_kv": 4, "context_length": 64},
        )
        assert result["success"] is True
        assert result["gguf_arch"] == "llama"
        assert result["converted_tensors"] == 12
        reader = gguf.GGUFReader(dst)
        tensor_names = {str(tensor.name) for tensor in reader.tensors}
        assert "token_embd.weight" in tensor_names
        assert "blk.0.attn_q.weight" in tensor_names
        assert "blk.0.ffn_down.weight" in tensor_names
        assert "model.layers.0.self_attn.q_proj.weight" not in tensor_names
        assert "llama.context_length" in reader.fields
        assert "llama.attention.head_count" in reader.fields
        print("PASS: HF Llama safetensors export writes llama.cpp tensor names and metadata")


def test_external_gguf_k_quantizer_or_missing_engine() -> None:
    try:
        import gguf
    except ImportError:
        print("SKIP: GGUF K quantization requires gguf")
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src = os.path.join(tmp, "source.safetensors")
        dst = os.path.join(tmp, "model-q4.gguf")
        _write_source(src)
        try:
            quantize_model_file(src, dst, "gguf_q4_k_m")
        except ValueError as exc:
            assert "requires an existing" in str(exc)
        else:
            raise AssertionError("GGUF K quantization should reject non-GGUF input")

        exe = Path(sys.executable).resolve().parent / "tools" / "llama.cpp" / "llama-quantize.exe"
        if not exe.is_file():
            print("SKIP: llama-quantize engine is not installed")
            return

        source_gguf = Path(tmp) / "tiny-llama-f16.gguf"
        output_gguf = Path(tmp) / "tiny-llama-q4.gguf"
        _write_tiny_llama_gguf(source_gguf, gguf)
        result = quantize_model_file(str(source_gguf), str(output_gguf), "gguf_q4_k_m")
        assert result["success"] is True
        assert result["gguf_provider"] == "external"
        assert result["gguf_quant_type"] == "Q4_K_M"
        assert output_gguf.is_file() and output_gguf.stat().st_size > 0
        reader = gguf.GGUFReader(str(output_gguf))
        assert len(reader.tensors) >= 8
        print("PASS: GGUF q4_k_m external quantizer runs on a llama-compatible GGUF")


def _write_tiny_llama_gguf(path: Path, gguf) -> None:
    import numpy as np

    rng = np.random.default_rng(123)
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("tiny-llama-smoke")
    writer.add_context_length(64)
    writer.add_embedding_length(32)
    writer.add_block_count(1)
    writer.add_feed_forward_length(64)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_rope_dimension_count(8)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_file_type(gguf.GGMLQuantizationType.F16)
    tensors = {
        "token_embd.weight": (64, 32),
        "output_norm.weight": (32,),
        "output.weight": (64, 32),
        "blk.0.attn_norm.weight": (32,),
        "blk.0.ffn_norm.weight": (32,),
        "blk.0.attn_q.weight": (32, 32),
        "blk.0.attn_k.weight": (32, 32),
        "blk.0.attn_v.weight": (32, 32),
        "blk.0.attn_output.weight": (32, 32),
        "blk.0.ffn_gate.weight": (64, 32),
        "blk.0.ffn_down.weight": (32, 64),
        "blk.0.ffn_up.weight": (64, 32),
    }
    for name, shape in tensors.items():
        writer.add_tensor(name, rng.normal(0, 0.02, shape).astype(np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    close = getattr(writer, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    test_plain_fp16_quantize()
    test_plain_fp8_quantize_when_available()
    test_lulynx_int8_rowwise_quantize()
    test_lulynx_uint4_rowwise_quantize()
    test_legacy_symmetric_uint4_rowwise_still_decodes()
    test_quantization_validator_detects_wrong_format()
    test_toolbox_runner_model_quantize_returns_validation_report()
    test_gguf_export_or_missing_dependency()
    test_gguf_q8_export_or_missing_dependency()
    test_hf_llama_gguf_export_or_missing_dependency()
    test_external_gguf_k_quantizer_or_missing_engine()
    print("\nAll Lulynx quantized safetensors smoke tests passed!")

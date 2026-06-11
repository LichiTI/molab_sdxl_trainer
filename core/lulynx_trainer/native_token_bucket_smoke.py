"""Smoke tests for native DiT no-pad token bucket summaries."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .anima_cached_dataset import AnimaCachedDataset
    from .compile_contract import resolve_compile_contract
    from .newbie_cached_dataset import NewbieCacheSchema, NewbieCachedDataset
except ImportError:
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from core.lulynx_trainer.anima_cached_dataset import AnimaCachedDataset
    from core.lulynx_trainer.compile_contract import resolve_compile_contract
    from core.lulynx_trainer.newbie_cached_dataset import NewbieCacheSchema, NewbieCachedDataset


def _write_text(path: Path, text: str = "test") -> None:
    path.write_text(text, encoding="utf-8")


def test_anima_visual_token_buckets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for stem, size in (("a", 128), ("b", 96)):
            (root / f"{stem}.png").write_bytes(b"fake")
            _write_text(root / f"{stem}.txt")
            np.savez(
                root / f"{stem}_{size * 8}x{size * 8}_anima.npz",
                latents_128x128=np.zeros((16, size, size), dtype=np.float16),
            )
            np.savez(
                root / f"{stem}_anima_te.npz",
                prompt_embeds=np.zeros((8, 4), dtype=np.float16),
            )
        dataset = AnimaCachedDataset(root, enable_bucket=True)
        summary = dataset.get_token_bucket_summary()
        assert summary["mode"] == "no_pad"
        assert summary["bucket_count"] == 2
        assert "4096:128x128" in summary["buckets"]
        assert "2304:96x96" in summary["buckets"]
    print("PASS: test_anima_visual_token_buckets")


def test_newbie_visual_token_buckets() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for stem, size in (("a", 64), ("b", 32)):
            np.savez(
                root / f"{stem}_newbie.npz",
                schema_version=np.array([2], dtype=np.int64),
                latents=np.zeros((16, size, size), dtype=np.float16),
                encoder_hidden_states=np.zeros((8, 4), dtype=np.float16),
                pooled_prompt_embeds=np.zeros((4,), dtype=np.float16),
                attention_mask=np.ones((8,), dtype=np.bool_),
            )
        schema = NewbieCacheSchema(require_schema_version=False, expected_latent_channels=16, expected_hidden_size=4, expected_pooled_size=4)
        dataset = NewbieCachedDataset(root, schema=schema)
        summary = dataset.get_token_bucket_summary()
        assert summary["mode"] == "no_pad"
        assert summary["bucket_count"] == 2
        assert "4096:64x64" in summary["buckets"]
        assert "1024:32x32" in summary["buckets"]
    print("PASS: test_newbie_visual_token_buckets")


def test_anima_bucket_compile_contract() -> None:
    cfg = SimpleNamespace(
        torch_compile=True,
        torch_compile_scope="per_block",
        anima_compile_scope="per_block",
        compile_contract_strict=True,
        compile_require_cache_first=True,
        compile_static_shape_drop_last=True,
        native_token_bucket_compile=True,
        anima_cached_training=True,
        anima_fixed_text_tokens=16,
        anima_fixed_visual_tokens=0,
    )
    plan = SimpleNamespace(
        torch_compile=True,
        torch_compile_scope="per_block",
        anima_compile_scope="per_block",
    )
    decision = resolve_compile_contract(cfg, plan, model_arch="anima")
    assert decision.resolved == "per_block"
    assert decision.cache_first_required is True
    assert plan.torch_compile_scope == "per_block"
    assert any("no-pad cached visual token buckets" in reason for reason in decision.reasons)
    print("PASS: test_anima_bucket_compile_contract")


def test_newbie_bucket_compile_contract() -> None:
    cfg = SimpleNamespace(
        torch_compile=True,
        torch_compile_scope="per_block",
        anima_compile_scope="",
        compile_contract_strict=True,
        compile_require_cache_first=True,
        compile_static_shape_drop_last=True,
        native_token_bucket_compile=True,
        use_cache=True,
        newbie_fixed_text_tokens=256,
        newbie_fixed_visual_tokens=0,
    )
    plan = SimpleNamespace(
        torch_compile=True,
        torch_compile_scope="per_block",
        anima_compile_scope="",
    )
    decision = resolve_compile_contract(cfg, plan, model_arch="newbie")
    assert decision.resolved == "per_block"
    assert decision.cache_first_required is True
    assert plan.torch_compile_scope == "per_block"
    assert any("Newbie per-block compile uses no-pad cached visual token buckets" in reason for reason in decision.reasons)
    print("PASS: test_newbie_bucket_compile_contract")


if __name__ == "__main__":
    test_anima_visual_token_buckets()
    test_newbie_visual_token_buckets()
    test_anima_bucket_compile_contract()
    test_newbie_bucket_compile_contract()

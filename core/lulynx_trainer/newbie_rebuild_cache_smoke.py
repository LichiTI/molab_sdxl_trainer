# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Newbie rebuild-cache deletion scope."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.trainer import LulynxTrainer


def _write_min_cache(path: Path) -> None:
    np.savez(
        path,
        newbie_cache_schema_version=np.array(1, dtype=np.int64),
        latents=np.ones((16, 2, 2), dtype=np.float32),
        encoder_hidden_states=np.ones((2, 4), dtype=np.float32),
        pooled_prompt_embeds=np.ones((4,), dtype=np.float32),
        attention_mask=np.ones((2,), dtype=np.bool_),
    )


def main() -> int:
    root = Path("H:/tmp/lulynx_newbie_rebuild_cache_smoke")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    te_cache = root / "te_cache"
    te_cache.mkdir(parents=True)

    generated_cache = root / "sample_newbie.npz"
    _write_min_cache(generated_cache)
    te_generated = te_cache / "te_sample.npz"
    np.savez(te_generated, sentinel=np.array([1], dtype=np.int64))
    keep_file = root / "keep.txt"
    keep_file.write_text("do not delete", encoding="utf-8")
    nested_keep = te_cache / "keep.bin"
    nested_keep.write_bytes(b"keep")

    rebuilt_cache = root / "rebuilt_newbie.npz"
    rebuilt_te = te_cache / "te_rebuilt.npz"

    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        model_type="newbie",
        train_data_dir=str(root),
        newbie_rebuild_cache=True,
        newbie_force_cache_only=False,
        use_cache=True,
    )
    trainer._log = lambda _msg: None
    trainer._model_arch_value = lambda: "newbie"
    trainer._has_anima_cached_training_data = lambda: False
    trainer._has_newbie_cached_training_data = lambda: False
    def _fake_build(force: bool = False):
        _write_min_cache(rebuilt_cache)
        np.savez(rebuilt_te, sentinel=np.array([2], dtype=np.int64))
        raise RuntimeError("__newbie_rebuild_done__")
    trainer._maybe_build_newbie_cache = _fake_build

    try:
        trainer._run_training()
    except RuntimeError as exc:
        if str(exc) != "__newbie_rebuild_done__":
            raise

    assert not generated_cache.exists(), "Generated *_newbie.npz cache should be deleted before rebuild"
    assert not te_generated.exists(), "Generated te_*.npz cache should be deleted before rebuild"
    assert keep_file.exists(), "Non-cache root file must be preserved"
    assert nested_keep.exists(), "Non-cache te_cache file must be preserved"
    assert rebuilt_cache.exists(), "Rebuild hook should be able to write a fresh *_newbie.npz cache"
    assert rebuilt_te.exists(), "Rebuild hook should be able to write a fresh te_*.npz cache"

    print("Newbie rebuild cache smoke passed: deletion is scoped to generated cache artifacts only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

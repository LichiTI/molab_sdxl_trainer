# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for the faithful 512-pad text-cache auto-rebuild guard (#133/#164).

Covers three layers without loading any real model:
  1. the pure policy ``anima_text_cache_needs_padding_rebuild`` (cleanroom);
  2. the cache peek ``_peek_anima_text_seq`` / ``_anima_text_cache_seq_lengths``
     against real temp npz files;
  3. the wiring ``_ensure_anima_faithful_text_cache_512`` — that faithful pins
     both text-token trims off and force-rebuilds only a short/stale cache.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAINER_ROOT = Path(_HERE)
_CORE_ROOT = _TRAINER_ROOT.parent
_BACKEND_ROOT = _CORE_ROOT.parent
_REPO_ROOT = _BACKEND_ROOT.parent
# Both roots are needed: ``_REPO_ROOT`` so internal ``backend.core.*`` imports
# resolve, ``_BACKEND_ROOT`` so the ``core.*`` package imports resolve.
for _path in (str(_REPO_ROOT), str(_BACKEND_ROOT), str(_CORE_ROOT), str(_TRAINER_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    from core.lulynx_trainer.anima_faithful_train_context import (
        anima_text_cache_needs_padding_rebuild,
    )
    from core.lulynx_trainer.trainer import LulynxTrainer
except ImportError:  # pragma: no cover - direct script execution from package dir
    from anima_faithful_train_context import anima_text_cache_needs_padding_rebuild
    from trainer import LulynxTrainer


def _needs(qwen3_seq, t5_seq, **kwargs) -> bool:
    return anima_text_cache_needs_padding_rebuild(
        qwen3_seq=qwen3_seq, t5_seq=t5_seq, **kwargs
    )


def _write_te_npz(directory: Path, stem: str, *, qwen3_seq: int, t5_seq: int) -> Path:
    path = directory / f"{stem}_anima_te.npz"
    np.savez(
        str(path),
        prompt_embeds=np.zeros((qwen3_seq, 8), dtype=np.float32),
        t5_input_ids=np.zeros((t5_seq,), dtype=np.int64),
        schema_version=np.asarray(2, dtype=np.int32),
    )
    return path


class _StubTrainer(LulynxTrainer):
    """Bind the faithful guard methods without the heavy ``LulynxTrainer`` init.

    Only the attributes the guard touches are populated; ``_build_anima_cache_now``
    is stubbed to record the ``force`` flag instead of running a real cache build.
    """

    def __init__(self, cache_dir: Path, config: SimpleNamespace) -> None:
        self.config = config
        self._cache_dir = Path(cache_dir)
        self.rebuild_calls: list[bool] = []
        self.logs: list[str] = []

    def _anima_cached_training_dirs(self):
        return [self._cache_dir]

    def _build_anima_cache_now(self, *, force: bool = False) -> None:
        self.rebuild_calls.append(bool(force))

    def _log(self, message: str) -> None:
        self.logs.append(str(message))


def _faithful_config(**overrides) -> SimpleNamespace:
    base = dict(
        anima_faithful_forward=True,
        anima_cached_training=True,
        anima_online_cache=False,
        anima_qwen3_max_token_length=0,  # 0 -> resolves to 512
        anima_t5_max_token_length=0,
        anima_cached_text_token_limit=256,  # the load-side trim that must be pinned to 0
        anima_text_token_limit=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# 1. pure policy
# --------------------------------------------------------------------------- #

def test_pure_helper_decides_staleness() -> None:
    assert _needs(66, 74) is True           # both short -> rebuild
    assert _needs(512, 512) is False        # both full -> keep
    assert _needs(None, None) is False      # nothing peeked -> fail-open
    assert _needs(512, None) is False       # qwen3 full, t5 unknown -> don't force
    assert _needs(None, 66) is True         # t5 short -> rebuild
    # custom expected lengths honor non-default max-token config
    assert _needs(256, 256, expected_qwen3=256, expected_t5=256) is False
    assert _needs(255, 256, expected_qwen3=256, expected_t5=256) is True


# --------------------------------------------------------------------------- #
# 2. cache peek
# --------------------------------------------------------------------------- #

def test_peek_reads_seq_lengths() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        stale = _write_te_npz(root, "stale", qwen3_seq=66, t5_seq=74)
        full = _write_te_npz(root, "full", qwen3_seq=512, t5_seq=512)
        assert LulynxTrainer._peek_anima_text_seq(stale) == (66, 74)
        assert LulynxTrainer._peek_anima_text_seq(full) == (512, 512)


def test_seq_lengths_takes_min_and_fails_open_on_empty_dir() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        # empty dir -> nothing to peek -> (None, None) fail-open
        empty_stub = _StubTrainer(root, _faithful_config())
        assert empty_stub._anima_text_cache_seq_lengths() == (None, None)

        # two samples; the guard tracks the MIN seq across them
        _write_te_npz(root, "a", qwen3_seq=512, t5_seq=512)
        _write_te_npz(root, "b", qwen3_seq=66, t5_seq=74)
        stub = _StubTrainer(root, _faithful_config())
        assert stub._anima_text_cache_seq_lengths() == (66, 74)


# --------------------------------------------------------------------------- #
# 3. wiring
# --------------------------------------------------------------------------- #

def test_guard_rebuilds_stale_cache_and_pins_trims_off() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _write_te_npz(root, "stale", qwen3_seq=66, t5_seq=74)
        config = _faithful_config()
        stub = _StubTrainer(root, config)

        stub._ensure_anima_faithful_text_cache_512("model_path")

        assert config.anima_cached_text_token_limit == 0  # load-side trim pinned off
        assert config.anima_text_token_limit == 0          # build-side trim pinned off
        assert stub.rebuild_calls == [True]                # force rebuild fired


def test_guard_skips_rebuild_when_cache_already_padded() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _write_te_npz(root, "full", qwen3_seq=512, t5_seq=512)
        config = _faithful_config(anima_cached_text_token_limit=0)
        stub = _StubTrainer(root, config)

        stub._ensure_anima_faithful_text_cache_512("model_path")

        assert config.anima_cached_text_token_limit == 0
        assert stub.rebuild_calls == []  # already 512 -> no rebuild


def test_guard_is_noop_without_faithful() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _write_te_npz(root, "stale", qwen3_seq=66, t5_seq=74)
        config = _faithful_config(anima_faithful_forward=False)
        stub = _StubTrainer(root, config)

        stub._ensure_anima_faithful_text_cache_512("model_path")

        # faithful off -> guard must not touch trims or rebuild
        assert config.anima_cached_text_token_limit == 256
        assert stub.rebuild_calls == []


def test_guard_pins_trims_but_skips_rebuild_for_online_cache() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        _write_te_npz(root, "stale", qwen3_seq=66, t5_seq=74)
        config = _faithful_config(anima_online_cache=True)
        stub = _StubTrainer(root, config)

        stub._ensure_anima_faithful_text_cache_512("model_path")

        assert config.anima_cached_text_token_limit == 0  # still pin trims off
        assert stub.rebuild_calls == []  # online generates 512 live -> no prebuilt rebuild


# --------------------------------------------------------------------------- #
# 4. encoder resolution for the cache-first rebuild (#133/#164 root-cause fix)
# --------------------------------------------------------------------------- #

def test_resolve_component_paths_probes_subfolders() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        stub = _StubTrainer(root, _faithful_config())
        # empty model dir -> nothing resolves (fail-open)
        assert stub._resolve_anima_component_paths(str(root)) == {}

        # standard native-Anima layout
        (root / "vae").mkdir()
        (root / "vae" / "qwen_image_vae.safetensors").write_bytes(b"x")
        (root / "text_encoders").mkdir()
        (root / "text_encoders" / "qwen_3_06b_base.safetensors").write_bytes(b"x")
        (root / "tokenizer_t5").mkdir()
        (root / "tokenizer_t5" / "tokenizer.json").write_bytes(b"{}")
        resolved = stub._resolve_anima_component_paths(str(root))
        assert resolved.get("vae", "").endswith("qwen_image_vae.safetensors")
        assert resolved.get("qwen3", "").endswith("qwen_3_06b_base.safetensors")
        assert resolved.get("t5", "").endswith("tokenizer_t5")

        # glob fallback when the preferred filename differs
        alt = root / "alt"
        (alt / "vae").mkdir(parents=True)
        (alt / "vae" / "some_other_vae.safetensors").write_bytes(b"x")
        assert stub._resolve_anima_component_paths(str(alt)).get("vae", "").endswith(
            "some_other_vae.safetensors"
        )

        # single-file DiT path: vae/ + text_encoders/ are siblings of diffusion_models/,
        # so the resolver must walk up to the model root (not stop at diffusion_models/).
        dit = root / "diffusion_models" / "anima-base-v1.0.safetensors"
        dit.parent.mkdir(parents=True)
        dit.write_bytes(b"x")
        from_file = stub._resolve_anima_component_paths(str(dit))
        assert from_file.get("vae", "").endswith("qwen_image_vae.safetensors")
        assert from_file.get("qwen3", "").endswith("qwen_3_06b_base.safetensors")
        assert from_file.get("t5", "").endswith("tokenizer_t5")


def test_ensure_encoders_is_noop_without_a_loaded_model() -> None:
    # The CPU smoke stub never sets self.model; the encoder loader must fail-open
    # to False (no real load) so the rebuild-intent tests above stay deterministic.
    stub = _StubTrainer(Path("."), _faithful_config())
    assert stub._ensure_anima_cache_encoders_loaded("model_path") is False


def main() -> int:
    test_pure_helper_decides_staleness()
    test_peek_reads_seq_lengths()
    test_seq_lengths_takes_min_and_fails_open_on_empty_dir()
    test_guard_rebuilds_stale_cache_and_pins_trims_off()
    test_guard_skips_rebuild_when_cache_already_padded()
    test_guard_is_noop_without_faithful()
    test_guard_pins_trims_but_skips_rebuild_for_online_cache()
    test_resolve_component_paths_probes_subfolders()
    test_ensure_encoders_is_noop_without_a_loaded_model()
    print("anima_faithful_text_cache_guard_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

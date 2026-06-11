# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke test for Anima Raw Online Training (#124).

Verifies:
  1. native_cache_mode='online' no longer raises RuntimeError
  2. native_cache_mode='rebuild_cache' no longer raises and routes to the builder
  3. native_cache_mode='cache_first' still toggles anima_cached_training=True

We construct a minimal trainer-like stub instead of loading a full model,
so the test runs anywhere without GPU / model weights.
"""

from __future__ import annotations

import os
import sys
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_apply_native_cache_mode():
    """Import only the trainer module's _apply_native_cache_mode source.

    We can't import trainer.py directly without diffusers, so we extract
    the relevant method as text and exec it onto a stub class.
    """
    import textwrap
    src = Path(_HERE, "trainer.py").read_text(encoding="utf-8")
    start = src.index("    def _apply_native_cache_mode")
    end = src.index("    def _apply_runtime_env_hints", start)
    method_src = textwrap.dedent(src[start:end])
    return method_src


def _make_stub_trainer():
    """Build a minimal stub with the methods _apply_native_cache_mode needs."""
    log_lines = []

    class _StubTrainer:
        def __init__(self, mode: str, model_arch: str = "anima"):
            self.config = SimpleNamespace(
                native_cache_mode=mode,
                anima_cached_training=False,
                use_cache=False,
                train_data_dir="",
                masked_loss=False,
                latent_cache_disk_format="npz",
                latent_cache_disk_dtype="float16",
                anima_vae_chunk_size=0,
                anima_text_token_limit=0,
            )
            self.model = None
            self._model_arch = model_arch
            self._anima_cache_pending = False
            self.device = "cpu"
            import torch
            self.dtype = torch.float32

        def _model_arch_value(self):
            return self._model_arch

        def _log(self, msg):
            log_lines.append(msg)

        def _build_anima_cache_now(self):
            log_lines.append("BUILDER_INVOKED")

        def _log_cache_policy_report(self, _label):
            return None

        def _log_newbie_cache_parity_report(self, _label):
            return None

    return _StubTrainer, log_lines


def test_online_mode_does_not_raise():
    Stub, logs = _make_stub_trainer()
    t = Stub(mode="online")
    # Bind the method by executing the source
    method_src = _load_apply_native_cache_mode()
    namespace = {}
    exec(method_src, namespace)
    method = namespace["_apply_native_cache_mode"]
    method(t)  # should not raise
    assert t.config.anima_cached_training is False
    assert any("online" in m.lower() for m in logs)
    print("PASS: native_cache_mode='online' no longer raises")


def test_rebuild_cache_mode_routes_to_builder():
    Stub, logs = _make_stub_trainer()
    t = Stub(mode="rebuild_cache")
    method_src = _load_apply_native_cache_mode()
    namespace = {}
    exec(method_src, namespace)
    method = namespace["_apply_native_cache_mode"]
    method(t)  # should not raise
    assert "BUILDER_INVOKED" in logs
    assert t.config.anima_cached_training is True
    print("PASS: native_cache_mode='rebuild_cache' invokes builder + enables cached training")


def test_cache_first_mode_still_works():
    Stub, logs = _make_stub_trainer()
    t = Stub(mode="cache_first")
    method_src = _load_apply_native_cache_mode()
    namespace = {}
    exec(method_src, namespace)
    method = namespace["_apply_native_cache_mode"]
    method(t)
    assert t.config.anima_cached_training is True
    print("PASS: native_cache_mode='cache_first' still toggles anima_cached_training")


def test_force_cache_only_mode_still_works():
    Stub, logs = _make_stub_trainer()
    t = Stub(mode="force_cache_only")
    method_src = _load_apply_native_cache_mode()
    namespace = {}
    exec(method_src, namespace)
    method = namespace["_apply_native_cache_mode"]
    method(t)
    assert t.config.anima_cached_training is True
    print("PASS: native_cache_mode='force_cache_only' still toggles anima_cached_training")


def test_newbie_arch_unaffected():
    Stub, logs = _make_stub_trainer()
    t = Stub(mode="rebuild_cache", model_arch="newbie")
    t.config.newbie_rebuild_cache = False
    method_src = _load_apply_native_cache_mode()
    namespace = {}
    exec(method_src, namespace)
    method = namespace["_apply_native_cache_mode"]
    method(t)
    # Newbie path takes a different branch and sets newbie_rebuild_cache
    assert t.config.newbie_rebuild_cache is True
    print("PASS: newbie arch routes to its own rebuild_cache branch")


if __name__ == "__main__":
    test_online_mode_does_not_raise()
    test_rebuild_cache_mode_routes_to_builder()
    test_cache_first_mode_still_works()
    test_force_cache_only_mode_still_works()
    test_newbie_arch_unaffected()
    print("\nAll Anima raw online training smoke tests passed!")

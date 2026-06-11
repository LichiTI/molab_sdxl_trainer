# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test: Newbie native loader propagates trust_remote_code to Transformers calls.

Patches AutoModel/AutoConfig/AutoTokenizer at the import site inside
``newbie_loader`` so no real model weights or network access is needed.
Proves the flag reaches every ``from_pretrained`` / ``from_config`` call
on the native Gemma and CLIP paths.

NOTE: This test pre-mocks transitive heavy dependencies (diffusers, xformers)
that are unavailable in the test runner's Python environment.  It imports
only the three target functions from ``newbie_loader`` via importlib with
the problematic parent modules stubbed out.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# ── Pre-mock heavy transitive deps so newbie_loader can import ────────

_MOCKED_MODULES = [
    # diffusers and submodules
    "diffusers",
    "diffusers.loaders",
    "diffusers.loaders.ip_adapter",
    "diffusers.pipelines",
    "diffusers.pipelines.stable_diffusion_xl",
    "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl",
    "diffusers.models",
    "diffusers.models.attention_processor",
    "diffusers.utils",
    "diffusers.utils.import_utils",
    # xformers
    "xformers",
    "xformers.ops",
    "xformers.ops.fmha",
    "xformers.ops.fmha.flash3",
    "xformers.flash_attn_3",
]

_orig_modules: dict[str, object] = {}
for _mod_name in _MOCKED_MODULES:
    _orig_modules[_mod_name] = sys.modules.get(_mod_name)
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = SimpleNamespace()

# Stub single_file_loader which imports from diffusers
if "core.lulynx_trainer.single_file_loader" not in sys.modules:
    sys.modules["core.lulynx_trainer.single_file_loader"] = SimpleNamespace(
        SDXLSingleFileLoader=type("SDXLSingleFileLoader", (), {}),
    )

# Ensure the package path is available
if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

# Now safe to import the target functions
from core.lulynx_trainer.newbie_loader import (  # noqa: E402
    _load_auto_model_with_local_safetensors,
    _load_clip_native,
    _load_gemma_native,
)
import torch  # noqa: E402


# ── Call-recording stubs ─────────────────────────────────────────────

_calls: list[tuple[str, dict]] = []


def _reset_calls() -> None:
    _calls.clear()


class _StubAutoModel:
    """Records kwargs passed to from_pretrained / from_config."""

    @staticmethod
    def from_pretrained(*args, **kwargs):
        _calls.append(("AutoModel.from_pretrained", kwargs))
        return SimpleNamespace(parameters=lambda: iter([]))

    @staticmethod
    def from_config(*args, **kwargs):
        _calls.append(("AutoModel.from_config", kwargs))
        return SimpleNamespace(parameters=lambda: iter([]))


class _StubAutoConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        _calls.append(("AutoConfig.from_pretrained", kwargs))
        return SimpleNamespace()


class _StubAutoTokenizer:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        _calls.append(("AutoTokenizer.from_pretrained", kwargs))
        return SimpleNamespace(model_max_length=512)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_fake_dir(base: Path, name: str) -> Path:
    """Create a directory with a dummy file so Path.exists() passes."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text("{}")
    return d


# ── Tests ────────────────────────────────────────────────────────────

def test_auto_model_helper_propagates_trust_true() -> None:
    """_load_auto_model_with_local_safetensors passes trust_remote_code=True."""
    tmpdir = Path(tempfile.mkdtemp(prefix="trc_auto_true_"))
    try:
        model_dir = _make_fake_dir(tmpdir, "model")
        with (
            patch("transformers.AutoModel", _StubAutoModel),
            patch("transformers.AutoConfig", _StubAutoConfig),
        ):
            _reset_calls()
            _load_auto_model_with_local_safetensors(
                model_dir, torch.float32, trust_remote_code=True,
            )
        assert any(
            kw.get("trust_remote_code") is True
            for _, kw in _calls
        ), f"trust_remote_code=True not found in calls: {_calls}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_auto_model_helper_omits_trust_when_false() -> None:
    """_load_auto_model_with_local_safetensors omits trust_remote_code when False."""
    tmpdir = Path(tempfile.mkdtemp(prefix="trc_auto_false_"))
    try:
        model_dir = _make_fake_dir(tmpdir, "model")
        with (
            patch("transformers.AutoModel", _StubAutoModel),
            patch("transformers.AutoConfig", _StubAutoConfig),
        ):
            _reset_calls()
            _load_auto_model_with_local_safetensors(
                model_dir, torch.float32, trust_remote_code=False,
            )
        for name, kw in _calls:
            assert "trust_remote_code" not in kw, (
                f"{name} should NOT have trust_remote_code when flag=False, "
                f"but got trust_remote_code={kw['trust_remote_code']!r}"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_gemma_native_propagates_trust_true() -> None:
    """_load_gemma_native passes trust_remote_code=True to model AND tokenizer."""
    tmpdir = Path(tempfile.mkdtemp(prefix="trc_gemma_"))
    try:
        _make_fake_dir(tmpdir, "text_encoder")
        _make_fake_dir(tmpdir, "tokenizer")
        with (
            patch("transformers.AutoModel", _StubAutoModel),
            patch("transformers.AutoConfig", _StubAutoConfig),
            patch("transformers.AutoTokenizer", _StubAutoTokenizer),
        ):
            _reset_calls()
            _load_gemma_native(str(tmpdir), torch.float32, 512, trust_remote_code=True)

        model_calls = [(n, kw) for n, kw in _calls if "AutoModel" in n or "AutoConfig" in n]
        assert model_calls, "No AutoModel/AutoConfig calls recorded for Gemma"
        for name, kw in model_calls:
            assert kw.get("trust_remote_code") is True, (
                f"{name} missing trust_remote_code=True (got {kw})"
            )

        tok_calls = [(n, kw) for n, kw in _calls if "Tokenizer" in n]
        assert tok_calls, "No AutoTokenizer calls recorded for Gemma"
        for name, kw in tok_calls:
            assert kw.get("trust_remote_code") is True, (
                f"{name} missing trust_remote_code=True (got {kw})"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_clip_native_propagates_trust_true() -> None:
    """_load_clip_native passes trust_remote_code=True to model AND tokenizer."""
    tmpdir = Path(tempfile.mkdtemp(prefix="trc_clip_"))
    try:
        _make_fake_dir(tmpdir, "clip_model")
        _make_fake_dir(tmpdir, "tokenizer")
        with (
            patch("transformers.AutoModel", _StubAutoModel),
            patch("transformers.AutoConfig", _StubAutoConfig),
            patch("transformers.AutoTokenizer", _StubAutoTokenizer),
        ):
            _reset_calls()
            _load_clip_native(str(tmpdir), torch.float32, 2048, trust_remote_code=True)

        model_calls = [(n, kw) for n, kw in _calls if "AutoModel" in n or "AutoConfig" in n]
        assert model_calls, "No AutoModel/AutoConfig calls recorded for CLIP"
        for name, kw in model_calls:
            assert kw.get("trust_remote_code") is True, (
                f"{name} missing trust_remote_code=True (got {kw})"
            )

        tok_calls = [(n, kw) for n, kw in _calls if "Tokenizer" in n]
        assert tok_calls, "No AutoTokenizer calls recorded for CLIP"
        for name, kw in tok_calls:
            assert kw.get("trust_remote_code") is True, (
                f"{name} missing trust_remote_code=True (got {kw})"
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_config_round_trip_preserves_flag() -> None:
    """UnifiedTrainingConfig trust_remote_code survives config_adapter normalization."""
    from core.lulynx_trainer.config_adapter import ConfigAdapter

    raw = {
        "trust_remote_code": True,
        "pretrained_model_name_or_path": "dummy",
        "output_dir": "/tmp/dummy",
    }
    cfg = ConfigAdapter.from_frontend_dict(raw)
    assert getattr(cfg, "trust_remote_code", False) is True, (
        f"ConfigAdapter.from_frontend_dict dropped trust_remote_code; "
        f"got {getattr(cfg, 'trust_remote_code', 'MISSING')}"
    )


# ── Main ─────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        ("auto_model_helper_propagates_trust_true", test_auto_model_helper_propagates_trust_true),
        ("auto_model_helper_omits_trust_when_false", test_auto_model_helper_omits_trust_when_false),
        ("gemma_native_propagates_trust_true", test_gemma_native_propagates_trust_true),
        ("clip_native_propagates_trust_true", test_clip_native_propagates_trust_true),
        ("config_round_trip_preserves_flag", test_config_round_trip_preserves_flag),
    ]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        print("FAIL — trust_remote_code propagation NOT fully proved")
        return 1
    print("PASS — trust_remote_code propagates from config to all Newbie HF load calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

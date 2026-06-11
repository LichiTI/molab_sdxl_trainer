# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test: Newbie CLIP loader recovers missing support files via runtime overlay."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.newbie_loader import _load_clip_native


class _StubAutoTokenizer:
    seen_paths: list[str] = []

    @staticmethod
    def from_pretrained(*args, **kwargs):
        if args:
            _StubAutoTokenizer.seen_paths.append(str(args[0]))
        return type("StubTokenizer", (), {"model_max_length": 2048})()


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="newbie_clip_overlay_"))
    try:
        _StubAutoTokenizer.seen_paths.clear()
        support_dir = tmpdir / "support_cache"
        support_dir.mkdir(parents=True, exist_ok=True)
        for name in ("__init__.py", "configuration_clip.py", "eva_model.py", "hf_model.py", "modeling_clip.py", "rope_embeddings.py", "transform.py"):
            (support_dir / name).write_text(f"# {name}\n", encoding="utf-8")
        (support_dir / "config.json").write_text(
            '{"auto_map": {"AutoConfig": "configuration_clip.JinaCLIPConfig", "AutoModel": "modeling_clip.JinaCLIPModel"}, "projection_dim": 1024, "pad_token_id": 0}',
            encoding="utf-8",
        )

        clip_dir = tmpdir / "clip_model"
        clip_dir.mkdir(parents=True, exist_ok=True)
        (clip_dir / "config.json").write_text(
            '{"auto_map": {"AutoModel": "modeling_clip.JinaCLIPModel"}}',
            encoding="utf-8",
        )
        (clip_dir / "modeling_clip.py").write_text(
            "from .eva_model import EVAVisionTransformer\n"
            "from .hf_model import HFTextEncoder\n",
            encoding="utf-8",
        )

        captured_paths: list[Path] = []

        def _fake_load_auto_model(
            model_dir: Path,
            dtype: torch.dtype,
            *,
            trust_remote_code: bool = False,
            disable_mmap: bool = False,
        ):
            captured_paths.append(Path(model_dir))
            return object()

        with (
            patch("core.lulynx_trainer.newbie_loader._find_jina_clip_support_cache", lambda: support_dir),
            patch("core.lulynx_trainer.newbie_loader._load_auto_model_with_local_safetensors", _fake_load_auto_model),
            patch("transformers.AutoTokenizer", _StubAutoTokenizer),
        ):
            model, tokenizer, notes = _load_clip_native(
                str(clip_dir),
                torch.float32,
                max_token_length=2048,
                trust_remote_code=True,
            )

        assert model is not None, f"CLIP model loader unexpectedly returned None; notes={notes}"
        assert tokenizer is not None, f"CLIP tokenizer unexpectedly returned None; notes={notes}"
        assert captured_paths, "Loader helper was never invoked"
        overlay_dir = captured_paths[-1]
        assert overlay_dir != clip_dir, "Expected runtime overlay path when support files are missing"
        assert (overlay_dir / "eva_model.py").is_file(), "Runtime overlay missing eva_model.py"
        assert (overlay_dir / "hf_model.py").is_file(), "Runtime overlay missing hf_model.py"
        assert any("runtime overlay" in note.lower() for note in notes), (
            f"Expected overlay recovery note, got: {notes}"
        )
        print(f"PASS — Newbie CLIP runtime overlay recovery staged at {overlay_dir}")

        unreadable_dir = tmpdir / "clip_model_unreadable"
        unreadable_dir.mkdir(parents=True, exist_ok=True)
        (unreadable_dir / "config.json").write_text(
            '{"auto_map": {"AutoConfig": "configuration_clip.JinaCLIPConfig", "AutoModel": "modeling_clip.JinaCLIPModel"}}',
            encoding="utf-8",
        )
        (unreadable_dir / "modeling_clip.py").write_text("class JinaCLIPModel: pass\n", encoding="utf-8")
        (unreadable_dir / "configuration_clip.py").write_text("class JinaCLIPConfig: pass\n", encoding="utf-8")
        (unreadable_dir / "tokenizer_config.json").write_text('{"pad_token_id": 0}', encoding="utf-8")
        (unreadable_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        (unreadable_dir / "jina-clip-v2.safetensors").write_bytes(b"stub")

        captured_paths.clear()
        _StubAutoTokenizer.seen_paths.clear()

        def _fake_file_readable(path: Path) -> bool:
            if Path(path) == unreadable_dir / "config.json":
                return False
            return True

        with (
            patch("core.lulynx_trainer.newbie_loader._load_local_jina_text_pooler", lambda model_dir: (object(), ["Recovered via stub local text pooler"])),
            patch("core.lulynx_trainer.newbie_loader._find_jina_clip_support_cache", lambda: support_dir),
            patch("core.lulynx_trainer.newbie_loader._file_readable", _fake_file_readable),
            patch("transformers.AutoTokenizer", _StubAutoTokenizer),
        ):
            model, tokenizer, notes = _load_clip_native(
                str(unreadable_dir),
                torch.float32,
                max_token_length=2048,
                trust_remote_code=True,
            )

        assert model is not None, f"Unreadable-config recovery unexpectedly returned no model; notes={notes}"
        assert tokenizer is not None, f"Unreadable-config recovery unexpectedly returned no tokenizer; notes={notes}"
        assert _StubAutoTokenizer.seen_paths, "Tokenizer loader was never invoked for unreadable-config case"
        overlay_tok_dir = Path(_StubAutoTokenizer.seen_paths[-1])
        assert overlay_tok_dir != unreadable_dir, "Expected tokenizer to use overlay directory when config is unreadable"
        assert (overlay_tok_dir / "config.json").is_file(), "Overlay did not restore config.json"
        assert any("unreadable" in note.lower() or "config.json" in note.lower() for note in notes), (
            f"Expected unreadable-config recovery note, got: {notes}"
        )
        print(f"PASS — Newbie CLIP unreadable config recovery staged at {overlay_tok_dir}")
        return 0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

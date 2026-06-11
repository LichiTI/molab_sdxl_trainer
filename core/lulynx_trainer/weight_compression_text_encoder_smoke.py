"""Real text-encoder weight-compression smoke for local Newbie Gemma.

This is a targeted probe for the high-value text-encoder path: load the local
Newbie Gemma text encoder, apply frozen int8 compression, then run a short
prompt forward. It skips cleanly when the local model directory is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.weight_compression import apply_weight_compression


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = repo_root / "models" / "newbie" / "text_encoder"
    if not model_dir.exists():
        print(f"SKIP: Newbie text encoder directory not found: {model_dir}")
        return 0

    from transformers import AutoModel, AutoTokenizer

    weight_file = model_dir / "gemma3-4b-it.safetensors"
    load_dir = model_dir
    if weight_file.is_file() and not (model_dir / "model.safetensors").is_file():
        load_dir = repo_root / ".tmp" / "newbie_text_encoder_overlay"
        load_dir.mkdir(parents=True, exist_ok=True)
        for src in model_dir.iterdir():
            dst = load_dir / ("model.safetensors" if src.name == "gemma3-4b-it.safetensors" else src.name)
            if dst.exists():
                continue
            try:
                import os
                os.link(src, dst)
            except OSError:
                if src.stat().st_size > 512 * 1024 * 1024:
                    try:
                        dst.symlink_to(src)
                    except OSError as exc:
                        print(f"SKIP: cannot link large Newbie text-encoder weight into overlay: {exc}")
                        return 0
                else:
                    import shutil
                    shutil.copy2(src, dst)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(load_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(load_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    for param in model.parameters():
        param.requires_grad_(False)

    compression = apply_weight_compression(
        type("Bundle", (), {"text_encoder_1": model})(),
        enabled=True,
        target="text_encoder",
        format="torchao_int8",
        train_text_encoder=False,
    )
    assert compression.enabled, compression
    assert compression.compressed_count > 0, compression.as_dict()

    model.to(device)
    tokens = tokenizer("newest, safe, 1girl", return_tensors="pt", truncation=True, max_length=32).to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hidden = outputs.hidden_states[-1] if getattr(outputs, "hidden_states", None) else getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise AssertionError("text encoder output did not include hidden states")
    assert torch.isfinite(hidden.float()).all()
    print(
        "Newbie text-encoder torchao compression smoke passed: "
        f"device={device}, compressed_params={compression.compressed_count}, "
        f"estimated_saved_mb={compression.estimated_saved_mb:.1f}, hidden_shape={tuple(hidden.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





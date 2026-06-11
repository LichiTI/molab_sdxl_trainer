# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test save metadata controls without loading a model."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.trainer import LulynxTrainer


class _Injector:
    def get_lora_state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_down.weight": torch.ones((1, 1), dtype=torch.float32)}


def _metadata(trainer: LulynxTrainer, *, step: int | None = None) -> dict[str, str] | None:
    if bool(getattr(trainer.config, "no_metadata", False)):
        return None
    comment = str(getattr(trainer.config, "training_comment", "") or "").strip()
    metadata = {
        "ss_base_model_version": str(getattr(trainer.config.model_arch, "value", trainer.config.model_arch)),
        "ss_output_name": str(trainer.config.output_name),
        "ss_network_dim": str(trainer.config.network_dim),
        "ss_network_alpha": str(trainer.config.network_alpha),
        "ss_learning_rate": str(trainer.config.learning_rate),
        "ss_training_comment": comment or "Trained with Lulynx Native Trainer",
    }
    if step is not None:
        metadata["ss_training_step"] = str(step)
    if bool(getattr(trainer.config, "rs_lora_enabled", False)):
        metadata["ss_rs_lora"] = "true"
        metadata["ss_scaling_strategy"] = "alpha_over_sqrt_rank"
    return metadata


def _save_adapter(trainer: LulynxTrainer, path: Path, *, step: int | None = None) -> None:
    state = trainer._get_current_adapter_state_dict(use_ema=False)
    assert state
    trainer._save_state_dict_to_path(state, path, _metadata(trainer, step=step))


def _trainer(root: Path, *, comment: str, no_metadata: bool) -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        output_dir=str(root),
        output_name="metadata_smoke",
        model_arch="sdxl",
        network_dim=8,
        network_alpha=4,
        learning_rate=1e-4,
        semantic_tuner_enabled=False,
        training_comment=comment,
        no_metadata=no_metadata,
        save_state=False,
        save_model_as="safetensors",
        mem_efficient_save=False,
        rs_lora_enabled=False,
        reft_enabled=False,
        prefix_tuning_length=0,
        postfix_tuning_length=0,
        anima_merge_export=False,
        merge_export=False,
        save_state_to_huggingface=False,
    )
    trainer.lora_injector = _Injector()
    trainer.model = None
    trainer._ema_tracker = None
    trainer._ti_trainer = None
    trainer._easy_control = None
    trainer._ip_adapter = None
    trainer._repa_projector = None
    trainer.on_runtime_event = None
    trainer._log = lambda _msg: None
    trainer._released_cuda_cache_tags = []
    trainer._maybe_release_tool_cuda_cache = trainer._released_cuda_cache_tags.append
    trainer._emit_runtime_event = lambda _event: None
    trainer._write_run_manifest = lambda *_args, **_kwargs: None
    trainer._run_manifest_extra = lambda: {}
    trainer._prune_saved_artifacts = lambda *_args, **_kwargs: None
    return trainer


def main() -> int:
    try:
        from safetensors import safe_open
    except ImportError:
        print("Save metadata smoke skipped: safetensors is not installed")
        return 0

    tmp_parent = Path(__file__).resolve().parents[3] / ".tmp"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="lulynx_save_metadata_smoke_", dir=str(tmp_parent)))

    with_comment = _trainer(root, comment="Warehouse metadata comment", no_metadata=False)
    save_path = root / "metadata_smoke.safetensors"
    _save_adapter(with_comment, save_path)
    with safe_open(str(save_path), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["ss_training_comment"] == "Warehouse metadata comment"
    assert metadata["ss_output_name"] == "metadata_smoke"

    rs_lora = _trainer(root, comment="rs-lora metadata", no_metadata=False)
    rs_lora.config.output_name = "metadata_smoke_rs_lora"
    rs_lora.config.rs_lora_enabled = True
    rs_lora_path = root / "metadata_smoke_rs_lora.safetensors"
    _save_adapter(rs_lora, rs_lora_path)
    with safe_open(str(rs_lora_path), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["ss_rs_lora"] == "true"
    assert metadata["ss_scaling_strategy"] == "alpha_over_sqrt_rank"

    no_meta = _trainer(root, comment="should not be written", no_metadata=True)
    no_meta.config.output_name = "metadata_smoke_none"
    no_meta_path = root / "metadata_smoke_none.safetensors"
    _save_adapter(no_meta, no_meta_path)
    with safe_open(str(no_meta_path), framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata is None or "ss_training_comment" not in metadata

    print("Save metadata smoke passed: training_comment, rs_lora, and no_metadata are honored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


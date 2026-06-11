# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test SDXL preview/sampling config wiring."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config_adapter import ConfigAdapter


def main() -> int:
    # 1. Route alias: sample_prompts -> sample_prompts (direct)
    parsed = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "enable_preview": True,
        "sample_prompts": "a cat\na dog",
        "sample_width": 768,
        "sample_height": 768,
        "sample_sampler": "dpm++_2m_sde",
        "sample_scheduler": "karras",
        "sample_every_n_epochs": 2,
        "sample_at_first": True,
        "sample_cfg": 8.5,
        "sample_steps": 30,
        "sample_seed": 42,
    })

    assert parsed.sample_prompts == "a cat\na dog", f"Got: {parsed.sample_prompts}"
    assert parsed.sample_width == 768, f"Got: {parsed.sample_width}"
    assert parsed.sample_height == 768, f"Got: {parsed.sample_height}"
    assert parsed.sample_sampler == "dpm++_2m_sde", f"Got: {parsed.sample_sampler}"
    assert parsed.sample_every_n_epochs == 2, f"Got: {parsed.sample_every_n_epochs}"
    assert parsed.sample_at_first is True, f"Got: {parsed.sample_at_first}"
    assert parsed.sample_cfg == 8.5, f"Got: {parsed.sample_cfg}"
    assert parsed.sample_steps == 30, f"Got: {parsed.sample_steps}"
    assert parsed.sample_seed == 42, f"Got: {parsed.sample_seed}"

    # 2. Route alias: positive_prompts -> sample_prompts
    parsed2 = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "positive_prompts": "landscape\nportrait",
        "negative_prompts": "blurry",
        "sample_cfg_scale": 7.0,
        "sample_every_n_steps": 500,
    })
    assert parsed2.sample_prompts == "landscape\nportrait", f"Got: {parsed2.sample_prompts}"
    assert parsed2.sample_negative == "blurry", f"Got: {parsed2.sample_negative}"
    assert parsed2.sample_cfg == 7.0, f"Got: {parsed2.sample_cfg}"
    assert parsed2.sample_every == 500, f"Got: {parsed2.sample_every}"

    # 3. Verify trainer _get_sample_prompts_list parses the config
    from core.lulynx_trainer.trainer import LulynxTrainer
    from core.lulynx_trainer.config import LulynxConfig

    config = LulynxConfig(sample_prompts="cat\ndog\nbird")
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = config
    prompts = trainer._get_sample_prompts_list()
    assert prompts == ["cat", "dog", "bird"], f"Got: {prompts}"

    # Empty case
    config2 = LulynxConfig(sample_prompts="")
    trainer2 = LulynxTrainer.__new__(LulynxTrainer)
    trainer2.config = config2
    assert trainer2._get_sample_prompts_list() == []

    # 4. Multi-preview groups support per-group delayed activation.
    preview_groups = [
        {"name": "fit", "prompt": "same-domain check"},
        {"name": "generalization", "prompt": "out-of-domain check", "start_epoch": 3},
        {"name": "late", "prompt": "late check", "start_after_epochs": 4},
    ]
    config3 = LulynxConfig(preview_groups=preview_groups, sample_prompts="fallback")
    trainer3 = LulynxTrainer.__new__(LulynxTrainer)
    trainer3.config = config3
    groups = trainer3._get_preview_groups()
    assert [group["name"] for group in groups] == ["fit", "generalization", "late"], groups
    assert [group["name"] for group in trainer3._filter_preview_groups_for_epoch(groups, 1)] == ["fit"]
    assert [group["name"] for group in trainer3._filter_preview_groups_for_epoch(groups, 3)] == ["fit", "generalization"]
    assert [group["name"] for group in trainer3._filter_preview_groups_for_epoch(groups, 5)] == ["fit", "generalization", "late"]

    parsed3 = ConfigAdapter.from_frontend_dict({
        "schema_id": "sdxl-lora",
        "enable_preview": True,
        "preview_groups": preview_groups,
        "sample_every_n_epochs": 1,
    })
    assert parsed3.preview_groups == preview_groups, f"Got: {parsed3.preview_groups}"

    print("SDXL preview smoke passed: preview/sampling config fields survive route and are consumed by trainer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

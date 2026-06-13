"""Smoke checks for TurboCore update checkpoint contract helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_flat_adamw_state import PersistentFlatAdamW  # noqa: E402
from core.turbocore_update_checkpoint_contract import (  # noqa: E402
    build_flat_adamw_checkpoint_contract,
    sync_flat_owner_state_from_optimizer,
)


def test_checkpoint_contract_roundtrip() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-2, weight_decay=0.0)
    loss = (param * param).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    owner = PersistentFlatAdamW([param], {"lr": 1e-2, "weight_decay": 0.0})
    sync = sync_flat_owner_state_from_optimizer(owner, optimizer, [param])
    assert sync["synced"] is True, sync
    assert sync["missing_state_tensors"] == 0, sync

    contract = build_flat_adamw_checkpoint_contract(owner, optimizer=optimizer, params=[param], run_roundtrip=True)
    assert contract["state_dict_available"] is True, contract
    assert contract["load_state_dict_available"] is True, contract
    assert contract["roundtrip_checked"] is True, contract
    assert contract["roundtrip_ok"] is True, contract
    assert contract["training_path_enabled"] is False, contract
    assert "trainer_checkpoint_integration_missing" in contract["blocked_reasons"], contract
    assert "trainer_state_save_sync_guard_missing" in contract["blocked_reasons"], contract
    assert "trainer_resume_owner_state_guard_missing" in contract["blocked_reasons"], contract

    verified = build_flat_adamw_checkpoint_contract(
        owner,
        optimizer=optimizer,
        params=[param],
        run_roundtrip=True,
        trainer_state_metadata_integrated=True,
        trainer_state_save_sync_verified=True,
        resume_owner_state_guard_verified=True,
    )
    assert verified["trainer_checkpoint_integration"] is True, verified
    assert verified["trainer_state_metadata_integrated"] is True, verified
    assert verified["trainer_state_save_sync_verified"] is True, verified
    assert verified["resume_owner_state_guard_verified"] is True, verified
    assert "trainer_checkpoint_integration_missing" not in verified["blocked_reasons"], verified
    assert "trainer_state_save_sync_guard_missing" not in verified["blocked_reasons"], verified
    assert "trainer_resume_owner_state_guard_missing" not in verified["blocked_reasons"], verified


def main() -> int:
    test_checkpoint_contract_roundtrip()
    print("turbocore_update_checkpoint_contract_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

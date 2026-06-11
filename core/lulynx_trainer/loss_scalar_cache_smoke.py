"""Smoke-test loss scalar caching and hot-path source guards."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.lulynx_trainer.training_loop import _LossScalarCache


class _FakeLoss:
    def __init__(self, value: float) -> None:
        self.value = value
        self.item_calls = 0

    def detach(self):
        return self

    def float(self):
        return self

    def item(self) -> float:
        self.item_calls += 1
        return self.value


def test_loss_scalar_cache_reuses_same_tensor_value() -> None:
    cache = _LossScalarCache()
    loss = _FakeLoss(1.25)

    assert cache.get(loss) == 1.25
    assert cache.get(loss) == 1.25
    assert loss.item_calls == 1

    other = _FakeLoss(1.25)
    assert cache.get(other) == 1.25
    assert other.item_calls == 1


def test_training_loop_source_guards() -> None:
    source = Path(__file__).with_name("training_loop.py").read_text(encoding="utf-8")
    assert "return raw_loss.item()" not in source
    assert "torch.stack([t.detach().float() for t in pending_loss_tensors]).mean().item()" not in source
    assert "float(loss.detach())" not in source
    assert "pending_loss_total" in source


def main() -> None:
    test_loss_scalar_cache_reuses_same_tensor_value()
    test_training_loop_source_guards()
    print("loss_scalar_cache_smoke: ok")


if __name__ == "__main__":
    main()

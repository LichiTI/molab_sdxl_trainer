from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch

TRAINER_ROOT = Path(__file__).resolve().parent
CORE_ROOT = TRAINER_ROOT.parent
BACKEND_ROOT = CORE_ROOT.parent
for _path in (BACKEND_ROOT, CORE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = type(sys)(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("core", CORE_ROOT)
_ensure_package("core.lulynx_trainer", TRAINER_ROOT)
_ensure_package("core.training_components", CORE_ROOT / "training_components")
_runtime = _import_from_file("core.lulynx_trainer.b_tier_runtime", TRAINER_ROOT / "b_tier_runtime.py")
BTierRuntime = _runtime.BTierRuntime
run_hutchinson_auto_freeze = _runtime.run_hutchinson_auto_freeze
_ghost = _import_from_file("core.training_components.ghost_replay", CORE_ROOT / "training_components" / "ghost_replay.py")
GhostRecorder = _ghost.GhostRecorder
GhostReplayer = _ghost.GhostReplayer
inspect_ghost_fingerprint = _ghost.inspect_ghost_fingerprint


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mid_block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh())
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.mid_block(x))


def test_hutchinson_auto_freeze() -> None:
    model = TinyBlock()
    with tempfile.TemporaryDirectory() as tmp:
        report = run_hutchinson_auto_freeze(model, output_dir=tmp, num_probes=2, freeze_ratio=0.5, device="cpu")
        assert report["scanned_layers"] > 0
        assert report["frozen_tensors"] > 0
        assert Path(report["report_path"]).exists()


def test_manifold_runtime() -> None:
    model = TinyBlock()
    runtime = BTierRuntime(
        model,
        device="cpu",
        manifold_enabled=True,
        manifold_weight=0.01,
        proj_dim=2,
        sparse_freq=1,
        anchor_layers="mid_block",
    )
    x = torch.randn(2, 4)
    _ = model(x)
    first_loss, first_state = runtime.compute_loss(step=0, timesteps=torch.tensor([1]), loss_device=torch.device("cpu"))
    assert first_loss is None
    assert first_state["manifold_baseline_captured"] >= 1
    runtime.clear()
    _ = model(x + 0.1)
    second_loss, second_state = runtime.compute_loss(step=1, timesteps=torch.tensor([1]), loss_device=torch.device("cpu"))
    assert second_state["captured_layers"] >= 1
    if second_loss is not None:
        assert second_state["manifold_loss"] >= 0.0
    runtime.close()


def test_ghost_runtime() -> None:
    model = TinyBlock()
    x = torch.randn(2, 4)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tiny.lulynx"
        recorder = GhostRecorder(proj_dim=4, target_layers=["mid_block"], device="cpu")
        stats = recorder.record(
            model,
            sample_inputs=[{"x": x}],
            timesteps=[7],
            forward_fn=lambda model, sample, **_kwargs: model(sample["x"]),
            metadata={"model_arch": "tiny"},
            strict=True,
        )
        assert stats["layers"] >= 1
        recorder.save(str(path))

        fingerprint = inspect_ghost_fingerprint(path)
        assert fingerprint["status"] == "ok"
        assert fingerprint["recorded_layer_count"] >= 1

        replayer = GhostReplayer.load(str(path))
        compatibility = replayer.validate_against_model(model, model_arch="tiny", anchor_layers=["mid_block"])
        assert compatibility["usable"] is True
        assert compatibility["matched_layer_count"] >= 1

        runtime = BTierRuntime(
            model,
            device="cpu",
            model_arch="tiny",
            ghost_enabled=True,
            ghost_path=str(path),
            ghost_interval=1,
            ghost_weight=0.1,
            anchor_layers="mid_block",
        )
        _ = model(x + 0.2)
        loss, state = runtime.compute_loss(step=0, timesteps=torch.tensor([7]), loss_device=torch.device("cpu"))
        assert loss is not None
        assert state["ghost_loss"] >= 0.0
        assert state["ghost_replay"]["compatibility_status"] == "ok"
        assert state["ghost_replay"]["attempts"] >= 1
        runtime.close()


if __name__ == "__main__":
    test_hutchinson_auto_freeze()
    test_manifold_runtime()
    test_ghost_runtime()
    print("b_tier_runtime_smoke: ok")

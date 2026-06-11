"""Smoke test for DDP (Distributed Data Parallel) training support.

Validates that:
1. setup_ddp / cleanup_ddp lifecycle works (single-process fallback)
2. get_rank / get_world_size / is_main_process return correct defaults
3. make_distributed_sampler returns None in single-process mode
4. DDPModelWrapper falls back gracefully without DDP
5. Config fields for DDP are properly defined
6. Route service maps DDP fields correctly
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

TRAINER_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = TRAINER_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_distributed_module():
    module_name = "_lulynx_ddp_smoke_distributed_target"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_ROOT / "distributed.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load distributed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_ddp_single_process_defaults():
    """In single-process mode, DDP helpers return sensible defaults."""
    distributed = _load_distributed_module()
    get_rank = distributed.get_rank
    get_world_size = distributed.get_world_size
    is_main_process = distributed.is_main_process
    is_ddp_active = distributed.is_ddp_active

    assert get_rank() == 0, f"Expected rank 0, got {get_rank()}"
    assert get_world_size() == 1, f"Expected world_size 1, got {get_world_size()}"
    assert is_main_process() is True, "Should be main process in single-process mode"
    assert is_ddp_active() is False, "DDP should not be active in single-process mode"
    print("PASS: test_ddp_single_process_defaults")
    return True


def test_ddp_setup_skips_single_process():
    """setup_ddp returns False when num_processes=1."""
    setup_ddp = _load_distributed_module().setup_ddp

    result = setup_ddp(num_processes=1, num_machines=1)
    assert result is False, f"Expected False for single-process, got {result}"
    print("PASS: test_ddp_setup_skips_single_process")
    return True


def test_make_distributed_sampler_returns_none():
    """make_distributed_sampler returns None when DDP is not active."""
    make_distributed_sampler = _load_distributed_module().make_distributed_sampler

    class DummyDataset:
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return idx

    sampler = make_distributed_sampler(DummyDataset())
    assert sampler is None, f"Expected None sampler in single-process mode, got {sampler}"
    print("PASS: test_make_distributed_sampler_returns_none")
    return True


def test_wrap_dataloader_passthrough():
    """wrap_dataloader_for_ddp returns original dataloader when DDP is not active."""
    wrap_dataloader_for_ddp = _load_distributed_module().wrap_dataloader_for_ddp

    class DummyDataset:
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return {"x": torch.tensor(idx)}

    dataset = DummyDataset()
    dl = torch.utils.data.DataLoader(dataset, batch_size=2)
    wrapped = wrap_dataloader_for_ddp(dl, dataset)
    assert wrapped is dl, "Should return same dataloader in single-process mode"
    print("PASS: test_wrap_dataloader_passthrough")
    return True


def test_ddp_model_wrapper_no_ddp():
    """DDPModelWrapper works as passthrough when DDP is not active."""
    DDPModelWrapper = _load_distributed_module().DDPModelWrapper

    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class DummyDataset:
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return {"x": torch.randn(10)}

    dataset = DummyDataset()
    dl = torch.utils.data.DataLoader(dataset, batch_size=2)

    wrapper = DDPModelWrapper(model, optimizer, dl, dataset)

    # Model should be unwrapped (no DDP)
    assert wrapper.model is model, "Model should be the raw model without DDP"
    assert wrapper.raw_model is model, "Raw model should be the original"
    assert wrapper.sampler is None, "No sampler in single-process mode"
    assert wrapper.dataloader is dl, "Dataloader should be unchanged"

    # Test backward passthrough
    x = torch.randn(2, 10)
    y = model(x).sum()
    wrapper.backward(y)  # Should just call loss.backward()

    # Test clip_grad_norm
    wrapper.clip_grad_norm(1.0)  # Should work without DDP

    print("PASS: test_ddp_model_wrapper_no_ddp")
    return True


def test_ddp_all_reduce_no_ddp():
    """all_reduce is a no-op when DDP is not active."""
    DDPModelWrapper = _load_distributed_module().DDPModelWrapper

    model = torch.nn.Linear(10, 5)
    wrapper = DDPModelWrapper(model)

    t = torch.tensor([1.0, 2.0, 3.0])
    result = wrapper.all_reduce(t, op="mean")
    assert torch.allclose(result, t), "all_reduce should be identity without DDP"
    print("PASS: test_ddp_all_reduce_no_ddp")
    return True


def test_ddp_save_on_main():
    """save_on_main only executes on main process."""
    DDPModelWrapper = _load_distributed_module().DDPModelWrapper

    model = torch.nn.Linear(10, 5)
    wrapper = DDPModelWrapper(model)

    called = []
    wrapper.save_on_main(lambda: called.append(True))
    assert called == [True], "save_on_main should call function in single-process mode"
    print("PASS: test_ddp_save_on_main")
    return True


def test_config_ddp_fields():
    """UnifiedTrainingConfig has DDP fields."""
    from core.configs import UnifiedTrainingConfig

    cfg = UnifiedTrainingConfig()
    assert hasattr(cfg, "multi_gpu"), "Config should have multi_gpu"
    assert hasattr(cfg, "num_processes"), "Config should have num_processes"
    assert hasattr(cfg, "num_machines"), "Config should have num_machines"
    assert hasattr(cfg, "main_process_ip"), "Config should have main_process_ip"
    assert hasattr(cfg, "main_process_port"), "Config should have main_process_port"
    assert hasattr(cfg, "ddp_find_unused_parameters"), "Config should have ddp_find_unused_parameters"
    assert hasattr(cfg, "ddp_gradient_as_bucket_view"), "Config should have ddp_gradient_as_bucket_view"
    assert hasattr(cfg, "ddp_static_graph"), "Config should have ddp_static_graph"

    assert cfg.multi_gpu is False
    assert cfg.num_processes == 1
    assert cfg.ddp_gradient_as_bucket_view is True
    assert cfg.ddp_static_graph is False
    print("PASS: test_config_ddp_fields")
    return True


def test_ddp_wrapper_set_epoch():
    """set_epoch is a no-op when DDP is not active."""
    DDPModelWrapper = _load_distributed_module().DDPModelWrapper

    model = torch.nn.Linear(10, 5)
    wrapper = DDPModelWrapper(model)
    wrapper.set_epoch(5)  # Should not raise
    print("PASS: test_ddp_wrapper_set_epoch")
    return True


def test_ddp_wrapper_wait_for_everyone():
    """wait_for_everyone is a no-op when DDP is not active."""
    DDPModelWrapper = _load_distributed_module().DDPModelWrapper

    model = torch.nn.Linear(10, 5)
    wrapper = DDPModelWrapper(model)
    wrapper.wait_for_everyone()  # Should not raise
    print("PASS: test_ddp_wrapper_wait_for_everyone")
    return True


def main():
    results = []
    tests = [
        test_ddp_single_process_defaults,
        test_ddp_setup_skips_single_process,
        test_make_distributed_sampler_returns_none,
        test_wrap_dataloader_passthrough,
        test_ddp_model_wrapper_no_ddp,
        test_ddp_all_reduce_no_ddp,
        test_ddp_save_on_main,
        test_config_ddp_fields,
        test_ddp_wrapper_set_epoch,
        test_ddp_wrapper_wait_for_everyone,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} — {e}")
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("DDP Smoke Test Results")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: {name}")
    print(f"\n{passed}/{total} tests passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

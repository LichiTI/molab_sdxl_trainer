# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for Port 5 VRAM optimizations.

Covers: Gradient Release, KahanAdamW8bit, Offloaded Checkpointing,
Dynamic Resolution Batch, Pipeline Parallel.
"""

from __future__ import annotations

import sys
import os
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


class _TinyModel(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestGradientRelease(unittest.TestCase):
    def test_post_step_release(self):
        from backend.core.lulynx_trainer.gradient_release import GradientReleaseManager

        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr = GradientReleaseManager(mode="post_step")
        count = mgr.register_parameters(model.parameters(), optimizer)
        self.assertGreater(count, 0)
        self.assertTrue(mgr.needs_external_optimizer_step)

        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        released = mgr.release_gradients_after_step()
        self.assertGreater(released, 0)

        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_during_backward_availability(self):
        from backend.core.lulynx_trainer.gradient_release import is_gradient_release_available
        available = is_gradient_release_available()
        self.assertIsInstance(available, bool)

    def test_during_backward_mode(self):
        from backend.core.lulynx_trainer.gradient_release import (
            GradientReleaseManager, is_gradient_release_available,
        )
        if not is_gradient_release_available():
            self.skipTest("register_post_accumulate_grad_hook not available")

        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mgr = GradientReleaseManager(mode="during_backward")
        mgr.register_parameters(model.parameters(), optimizer)
        self.assertFalse(mgr.needs_external_optimizer_step)

        x = torch.randn(4, 32)
        with mgr.step_context(is_accumulation_boundary=True):
            loss = model(x).sum()
            loss.backward()

        self.assertGreater(mgr.stats["released_count"], 0)


class TestKahanAdamW8bit(unittest.TestCase):
    def test_basic_step(self):
        from backend.core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit

        model = _TinyModel()
        optimizer = KahanAdamW8bit(model.parameters(), lr=1e-3)
        x = torch.randn(4, 32)

        initial_param = model.fc1.weight.clone()

        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        self.assertFalse(torch.equal(model.fc1.weight, initial_param))

    def test_kahan_compensation(self):
        from backend.core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit

        model = _TinyModel()
        optimizer = KahanAdamW8bit(model.parameters(), lr=1e-3)

        for _ in range(5):
            x = torch.randn(4, 32)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for p in model.parameters():
            state = optimizer.state[p]
            self.assertIn("kahan_comp", state)
            self.assertIn("exp_avg_q", state)

    def test_state_memory_estimate(self):
        from backend.core.lulynx_trainer.kahan_adamw8bit import KahanAdamW8bit

        model = _TinyModel()
        optimizer = KahanAdamW8bit(model.parameters(), lr=1e-3)
        x = torch.randn(4, 32)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        mb = optimizer.estimate_state_memory_mb()
        self.assertGreater(mb, 0)


class TestOffloadedCheckpointing(unittest.TestCase):
    def test_pool_allocate_reset(self):
        from backend.core.lulynx_trainer.offloaded_checkpointing import PinnedMemoryPool

        pool = PinnedMemoryPool(pool_gb=0.001)
        buf = pool.allocate(1024)
        self.assertEqual(buf.numel(), 1024)
        stats = pool.stats
        self.assertGreater(stats["alloc_count"], 0)
        pool.reset()
        pool.cleanup()

    def test_context_manager(self):
        from backend.core.lulynx_trainer.offloaded_checkpointing import OffloadedCheckpointContext

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        ctx = OffloadedCheckpointContext(pool_gb=0.01, device="cuda")
        model = _TinyModel().cuda()
        x = torch.randn(4, 32, device="cuda")

        with ctx.offload_context():
            out = model(x)
        out.sum().backward()

        stats = ctx.stats
        self.assertGreater(stats["offloaded_count"], 0)
        ctx.cleanup()


class TestDynamicResolutionBatch(unittest.TestCase):
    def test_same_resolution_no_adjustment(self):
        from backend.core.lulynx_trainer.dynamic_resolution_batch import (
            DynamicMicroBatchScheduler, ResolutionBatchConfig,
        )
        scheduler = DynamicMicroBatchScheduler(
            config=ResolutionBatchConfig(base_resolution=1024, base_accumulation_steps=4),
        )
        steps = scheduler.compute_accumulation_steps(1024)
        self.assertEqual(steps, 4)

    def test_lower_resolution_more_steps(self):
        from backend.core.lulynx_trainer.dynamic_resolution_batch import (
            DynamicMicroBatchScheduler, ResolutionBatchConfig,
        )
        scheduler = DynamicMicroBatchScheduler(
            config=ResolutionBatchConfig(
                base_resolution=1024,
                base_accumulation_steps=4,
                max_factor=4.0,
            ),
        )
        steps = scheduler.compute_accumulation_steps(512)
        self.assertGreater(steps, 4)

    def test_higher_resolution_fewer_steps(self):
        from backend.core.lulynx_trainer.dynamic_resolution_batch import (
            DynamicMicroBatchScheduler, ResolutionBatchConfig,
        )
        scheduler = DynamicMicroBatchScheduler(
            config=ResolutionBatchConfig(
                base_resolution=1024,
                base_accumulation_steps=4,
                min_factor=0.25,
            ),
        )
        steps = scheduler.compute_accumulation_steps(2048)
        self.assertLess(steps, 4)

    def test_batch_resolution_extraction(self):
        from backend.core.lulynx_trainer.dynamic_resolution_batch import (
            DynamicMicroBatchScheduler, ResolutionBatchConfig,
        )
        scheduler = DynamicMicroBatchScheduler(
            config=ResolutionBatchConfig(base_resolution=1024),
        )
        batch = {"images": torch.randn(1, 3, 768, 512)}
        res = scheduler.get_batch_resolution(batch)
        self.assertEqual(res, 768)

        batch_latent = {"latents": torch.randn(1, 4, 96, 64)}
        res_latent = scheduler.get_batch_resolution(batch_latent)
        self.assertEqual(res_latent, 96 * 8)


class TestPipelineParallel(unittest.TestCase):
    def test_availability_check(self):
        from backend.core.lulynx_trainer.pipeline_parallel import is_pipeline_parallel_available
        result = is_pipeline_parallel_available()
        self.assertIsInstance(result, bool)

    def test_config_validation(self):
        from backend.core.lulynx_trainer.pipeline_parallel import PipelineConfig
        config = PipelineConfig(num_chunks=2)
        config.validate()

        with self.assertRaises(ValueError):
            PipelineConfig(num_chunks=0).validate()

    def test_balance_by_layers(self):
        from backend.core.lulynx_trainer.pipeline_parallel import PipelineParallelManager, PipelineConfig

        mgr = PipelineParallelManager(PipelineConfig())
        blocks = [nn.Linear(32, 32) for _ in range(8)]
        assignments = mgr._balance_by_layers(blocks, 2)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(sum(len(a) for a in assignments), 8)

    def test_balance_by_params(self):
        from backend.core.lulynx_trainer.pipeline_parallel import PipelineParallelManager, PipelineConfig

        mgr = PipelineParallelManager(PipelineConfig())
        blocks = [nn.Linear(32, 32) for _ in range(8)]
        assignments = mgr._balance_by_params(blocks, 2)
        self.assertEqual(len(assignments), 2)
        all_indices = []
        for a in assignments:
            all_indices.extend(a)
        self.assertEqual(sorted(all_indices), list(range(8)))

    def test_stats_inactive(self):
        from backend.core.lulynx_trainer.pipeline_parallel import PipelineParallelManager, PipelineConfig
        mgr = PipelineParallelManager(PipelineConfig())
        self.assertFalse(mgr.is_active)
        self.assertFalse(mgr.stats["active"])


class TestConfigFields(unittest.TestCase):
    def test_port5_fields_exist(self):
        from backend.core.configs import UnifiedTrainingConfig, OptimizerType

        cfg = UnifiedTrainingConfig()
        self.assertFalse(cfg.gradient_release_enabled)
        self.assertEqual(cfg.gradient_release_mode, "post_step")
        self.assertEqual(cfg.cpu_offload_checkpointing_mode, "standard")
        self.assertAlmostEqual(cfg.cpu_offload_checkpointing_pool_gb, 1.0)
        self.assertFalse(cfg.resolution_aware_batch_enabled)
        self.assertEqual(cfg.resolution_aware_batch_base_resolution, 1024)
        self.assertFalse(cfg.pipeline_parallel_enabled)
        self.assertEqual(OptimizerType.KAHAN_ADAMW_8BIT.value, "KahanAdamW8bit")


if __name__ == "__main__":
    unittest.main()

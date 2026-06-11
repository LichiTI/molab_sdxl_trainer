# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for the four simple monitoring features."""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure backend.core modules are importable
# ---------------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"
AUDITOR_ROOT = CORE_ROOT / "auditor"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import importlib.util
from types import ModuleType


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Namespace setup
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_ensure_namespace("core.auditor", AUDITOR_ROOT)

# Load constants first (dependency of watchdog)
_load_module("core.constants", CORE_ROOT / "constants.py")
# Load types (dependency of auditor, watchdog, icu_health)
types_mod = _load_module("core.auditor.types", AUDITOR_ROOT / "types.py")
AuditMetrics = types_mod.AuditMetrics
AuditMode = types_mod.AuditMode

# Load icu_health
icu_mod = _load_module("core.auditor.icu_health", AUDITOR_ROOT / "icu_health.py")
compute_icu_score = icu_mod.compute_icu_score

# Load watchdog
watchdog_mod = _load_module("core.auditor.watchdog", AUDITOR_ROOT / "watchdog.py")
HardwareWatchdog = watchdog_mod.HardwareWatchdog


# ── Test 1: Peak VRAM capture ──────────────────────────────────────────────

def test_peak_vram_capture() -> None:
    """Verify torch.cuda peak memory APIs work (or skip on CPU)."""
    import torch
    if not torch.cuda.is_available():
        print("  SKIP: test_peak_vram_capture (no CUDA)")
        return

    torch.cuda.reset_peak_memory_stats()
    _ = torch.randn(1024, 1024, device="cuda")
    peak_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
    assert peak_mb > 0, f"Expected peak > 0 MB, got {peak_mb}"

    torch.cuda.reset_peak_memory_stats()
    peak_after_reset = torch.cuda.max_memory_reserved() / (1024 * 1024)
    assert peak_after_reset <= peak_mb, "Peak should not grow after reset without new alloc"
    print("  PASS: test_peak_vram_capture")


# ── Test 2: Attention runtime stats ────────────────────────────────────────

def test_attention_stats() -> None:
    """Verify attention call counters increment correctly."""
    # Load the anima_attention module
    # We need amd_runtime as a dependency
    _load_module("core.lulynx_trainer.amd_runtime", TRAINER_ROOT / "amd_runtime.py")
    attn_mod = _load_module(
        "core.lulynx_trainer.anima_attention",
        TRAINER_ROOT / "anima_attention.py",
    )

    attn_mod.reset_attention_stats()
    stats_before = attn_mod.snapshot_attention_stats()
    assert stats_before["sdpa_calls"] == 0, "Should start at 0 after reset"

    import torch
    q = torch.randn(1, 4, 16, 64)
    k = torch.randn(1, 4, 16, 64)
    v = torch.randn(1, 4, 16, 64)

    attn_mod.dit_attention(q, k, v, backend="sdpa")
    attn_mod.dit_attention(q, k, v, backend="sdpa")
    attn_mod.dit_attention(q, k, v, backend="torch")

    stats = attn_mod.snapshot_attention_stats()
    assert stats["sdpa_calls"] == 2, f"Expected 2 sdpa_calls, got {stats['sdpa_calls']}"
    assert stats["torch_calls"] == 1, f"Expected 1 torch_calls, got {stats['torch_calls']}"

    attn_mod.reset_attention_stats()
    stats_reset = attn_mod.snapshot_attention_stats()
    assert stats_reset["sdpa_calls"] == 0, "Should be 0 after reset"
    print("  PASS: test_attention_stats")


# ── Test 3: ICU health score ───────────────────────────────────────────────

def test_icu_health_score() -> None:
    """Verify ICU score computation from AuditMetrics."""
    # Perfect health
    metrics_good = AuditMetrics(
        stable_rank=8.0,
        svd_entropy=0.8,
        dead_neuron_rate=0.01,
        spectral_smoothness=0.2,
        gsnr=5.0,
    )
    score = compute_icu_score(metrics_good, ema_loss=0.05, prev_ema_loss=0.06)
    assert 90 <= score <= 100, f"Healthy metrics should give 90-100, got {score}"

    # Unhealthy: high dead neuron rate, collapsing rank, rising loss
    metrics_bad = AuditMetrics(
        stable_rank=0.5,
        svd_entropy=0.1,
        dead_neuron_rate=0.3,
        spectral_smoothness=1.5,
        gsnr=0.1,
    )
    score_bad = compute_icu_score(metrics_bad, ema_loss=0.10, prev_ema_loss=0.05)
    assert 0 <= score_bad <= 30, f"Unhealthy metrics should give 0-30, got {score_bad}"

    # All None metrics should give 100 (no penalties)
    metrics_empty = AuditMetrics()
    score_empty = compute_icu_score(metrics_empty)
    assert score_empty == 100, f"Empty metrics should give 100, got {score_empty}"

    # Clamp test
    assert 0 <= score <= 100
    assert 0 <= score_bad <= 100
    print("  PASS: test_icu_health_score")


# ── Test 4: Audit mode override ───────────────────────────────────────────

def test_audit_mode_override() -> None:
    """Verify HardwareWatchdog respects mode_override."""
    wd_lite = HardwareWatchdog(mode_override="lite")
    assert wd_lite.check_policy() == AuditMode.LITE, "Override to LITE should work"

    wd_pro = HardwareWatchdog(mode_override="PRO")
    assert wd_pro.check_policy() == AuditMode.PRO, "Override to PRO should work"

    wd_stop = HardwareWatchdog(mode_override="stop")
    assert wd_stop.check_policy() == AuditMode.STOP, "Override to STOP should work"

    # SUSPEND override should NOT work (not in the valid override set)
    wd_suspend = HardwareWatchdog(mode_override="suspend")
    result = wd_suspend.check_policy()
    assert result != AuditMode.SUSPEND, "SUSPEND should not be overridable (it's sampling-only)"
    print("  PASS: test_audit_mode_override")


# ── Test 5: Audit mode auto (no override) ─────────────────────────────────

def test_audit_mode_auto() -> None:
    """Without override, watchdog should return PRO/LITE/STOP based on VRAM."""
    wd = HardwareWatchdog(mode_override="")
    mode = wd.check_policy()
    assert mode in (AuditMode.PRO, AuditMode.LITE, AuditMode.STOP), \
        f"Auto mode should return PRO/LITE/STOP, got {mode}"

    # Sampling flag should return SUSPEND regardless
    mode_sampling = wd.check_policy(is_sampling=True)
    assert mode_sampling == AuditMode.SUSPEND, "Sampling should always return SUSPEND"
    print("  PASS: test_audit_mode_auto")


# ── Test 6: Attention entropy probe ──────────────────────────────────────────

def test_attn_entropy_probe() -> None:
    """Verify attention entropy sparse-sampling probe works correctly."""
    ent_mod = _load_module("core.lulynx_trainer.attn_entropy", TRAINER_ROOT / "attn_entropy.py")

    # State machine: not armed → should_probe returns False
    ent_mod.disarm_probe()
    assert not ent_mod.should_probe(), "Should not probe when disarmed"

    # Arm → one call should hit, rest should miss
    ent_mod.arm_probe(total_attention_calls=5)
    hits = sum(1 for _ in range(5) if ent_mod.should_probe())
    assert hits == 1, f"Expected exactly 1 hit in 5 calls, got {hits}"

    # probe_from_qk: compute entropy from Q/K tensors
    ent_mod.reset_entropy_stats()
    ent_mod.arm_probe(total_attention_calls=1)
    assert ent_mod.should_probe()

    import torch
    q = torch.randn(1, 4, 64, 32)
    k = torch.randn(1, 4, 64, 32)
    ent_mod.probe_from_qk(q, k)

    val = ent_mod.collect_probe()
    assert val is not None, "Probe should have a result"
    assert 0.0 < val < 20.0, f"Entropy should be in a reasonable range, got {val}"

    # probe_materialized: compute from existing attention weights
    ent_mod.arm_probe(total_attention_calls=1)
    assert ent_mod.should_probe()

    weights = torch.zeros(1, 4, 64, 64)
    weights[:, :, :, 0] = 1.0  # all attention on first token → low entropy
    ent_mod.probe_materialized(weights)

    val_peaked = ent_mod.collect_probe()
    assert val_peaked is not None
    assert val_peaked < 0.01, f"Peaked weights should give near-zero entropy, got {val_peaked}"

    # Stats accumulation
    stats = ent_mod.snapshot_entropy_stats()
    assert stats["count"] == 2, f"Expected 2 samples, got {stats['count']}"
    assert stats["min"] <= stats["max"]

    ent_mod.reset_entropy_stats()
    stats_reset = ent_mod.snapshot_entropy_stats()
    assert stats_reset["count"] == 0, "Should be 0 after reset"

    print("  PASS: test_attn_entropy_probe")


def test_loss_tracker() -> None:
    """Verify LossTracker records, snapshots, and clears correctly."""
    lt_mod = _load_module("core.lulynx_trainer.loss_tracker", TRAINER_ROOT / "loss_tracker.py")
    tracker = lt_mod.LossTracker()

    # Disabled by default — record should be no-op
    tracker.record("test", 1.0, 0.9)
    snap = tracker.snapshot()
    assert snap is None, "Should return None when disabled"

    # Enable and record
    tracker.enable()
    tracker.record("snr_gamma", 1.0, 0.8, scale=0.8)
    tracker.record("wavelet", 0.8, 0.85, bias=0.05)

    snap = tracker.snapshot()
    assert snap is not None, "Should have records after enable"
    assert len(snap) == 2, f"Expected 2 records, got {len(snap)}"
    assert snap[0]["stage"] == "snr_gamma"
    assert snap[1]["stage"] == "wavelet"

    # Snapshot clears
    snap2 = tracker.snapshot()
    assert snap2 is None, "Should be empty after snapshot"

    # Summary
    tracker.record("prior", 0.85, 0.90, bias=0.05)
    summary = tracker.summary()
    assert summary["active_modifiers"] == 1

    print("  PASS: test_loss_tracker")


def test_dataset_analyzer() -> None:
    """Verify DatasetAnalyzer produces a valid report."""
    da_mod = _load_module("core.lulynx_trainer.dataset_analyzer", TRAINER_ROOT / "dataset_analyzer.py")

    # Create mock dataset with samples
    class MockSample:
        def __init__(self, w, h, tw, th, caption_path=None, image_path="img.png"):
            self.original_size = (w, h)
            self.target_size = (tw, th)
            self.caption_path = caption_path
            self.image_path = image_path

    class MockDataset:
        def __init__(self):
            self.samples = [
                MockSample(1024, 1024, 1024, 1024, caption_path=None),
                MockSample(512, 768, 512, 768, caption_path=None),
            ]

    dataset = MockDataset()
    report = da_mod.DatasetAnalyzer(dataset).analyze()
    assert report.total_images == 2
    assert report.captioned_images == 0
    assert report.caption_coverage == 0.0
    assert "1024x1024" in report.resolution_distribution

    d = report.as_dict()
    assert "total_images" in d

    lines = report.summary_lines()
    assert len(lines) >= 2

    print("  PASS: test_dataset_analyzer")


def test_vram_estimator() -> None:
    """Verify VRAM breakdown estimation with a toy model."""
    ve_mod = _load_module("core.lulynx_trainer.vram_estimator", TRAINER_ROOT / "vram_estimator.py")
    import torch.nn as nn

    class MockModel:
        def __init__(self):
            self.unet = nn.Linear(10, 10)
            self.vae = nn.Linear(10, 10)
            self.text_encoder_1 = nn.Linear(10, 10)
            self.text_encoder_2 = None

    class MockConfig:
        optimizer_type = "adamw"
        resolution = 512
        train_batch_size = 1
        gradient_checkpointing = False

    breakdown = ve_mod.estimate_vram_breakdown(MockModel(), MockConfig())
    assert breakdown.total_estimate_mb > 0
    assert breakdown.safety_rating in ("safe", "watch", "tight", "danger")
    assert len(breakdown.components) >= 3

    d = breakdown.as_dict()
    assert "total_estimate_mb" in d

    lines = breakdown.summary_lines()
    assert len(lines) >= 3

    assert ve_mod._optimizer_state_multiplier("Prodigy") == 4.0
    assert ve_mod._optimizer_state_multiplier("prodigyopt.Prodigy") == 4.0
    assert ve_mod._optimizer_state_multiplier("ProdigyScheduleFree") == 1.25
    assert ve_mod._optimizer_state_multiplier("prodigyplus.ProdigyPlusScheduleFree") == 1.25
    assert ve_mod._optimizer_state_multiplier("Adafactor") == 1.0
    assert ve_mod._optimizer_state_multiplier("SGD") == 0.0

    print("  PASS: test_vram_estimator")


def test_act_drift_tracker() -> None:
    """Verify ActivationDriftTracker installs hooks and computes drift."""
    ad_mod = _load_module("core.lulynx_trainer.act_drift", TRAINER_ROOT / "act_drift.py")
    import torch
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mid_block = nn.Linear(8, 8)

        def forward(self, x):
            return self.mid_block(x)

    model = TinyModel()
    tracker = ad_mod.ActivationDriftTracker(model)
    tracker.install()
    assert len(tracker._hooks) >= 1, "Should have at least 1 hook on mid_block"

    # Forward pass to capture features
    x = torch.randn(2, 8)
    _ = model(x)
    assert len(tracker._features) >= 1, "Should have captured features"

    # Capture baseline
    tracker.capture_baseline()
    assert tracker.has_baseline

    # Forward again with different input → should show drift
    _ = model(torch.randn(2, 8) * 10)
    drift = tracker.compute_drift()
    assert len(drift) >= 1, "Should report drift for tracked layers"
    for layer_name, vals in drift.items():
        assert "mean_drift" in vals
        assert "std_drift" in vals

    tracker.clear_features()
    assert len(tracker._features) == 0

    tracker.remove_hooks()
    assert len(tracker._hooks) == 0

    print("  PASS: test_act_drift_tracker")


def test_lr_finder() -> None:
    """Verify LRFinder runs and produces a result."""
    lrf_mod = _load_module("core.lulynx_trainer.lr_finder", TRAINER_ROOT / "lr_finder.py")
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

    call_count = [0]
    def mock_step():
        call_count[0] += 1
        x = torch.randn(2, 4)
        y = model(x)
        loss = y.pow(2).mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    finder = lrf_mod.LRFinder(
        model=model, optimizer=optimizer, step_fn=mock_step,
        start_lr=1e-5, end_lr=1e-1, num_steps=10,
    )
    result = finder.run()
    assert result.num_steps_run > 0, "Should have run at least 1 step"
    assert result.suggested_lr > 0, f"Suggested LR should be positive, got {result.suggested_lr}"
    assert len(result.lr_values) == result.num_steps_run

    d = result.as_dict()
    assert "suggested_lr" in d

    lines = result.summary_lines()
    assert len(lines) >= 2

    print("  PASS: test_lr_finder")


def test_grad_tracker() -> None:
    """Verify GradientCovarianceTracker EMA, variance, cosine, and Fisher."""
    gt_mod = _load_module("core.lulynx_trainer.grad_tracker", TRAINER_ROOT / "grad_tracker.py")
    import torch
    import torch.nn as nn

    tracker = gt_mod.GradientCovarianceTracker(smooth_factor=0.5)

    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    y = model(x)
    y.sum().backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    snap = tracker.update(float(norm), model.parameters())
    assert snap.norm > 0, f"Norm should be > 0, got {snap.norm}"
    assert snap.norm_ema > 0, "EMA should be > 0 after first update"
    assert snap.norm_var == 0.0, "Variance should be 0 after one sample"
    assert snap.cosine_sim is None, "Cosine should be None without enable_cosine"
    assert snap.fisher_diag > 0, f"Fisher diag should be > 0, got {snap.fisher_diag}"

    # Second update — variance should become nonzero
    model.zero_grad()
    y2 = model(torch.randn(2, 4))
    y2.sum().backward()
    norm2 = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    snap2 = tracker.update(float(norm2), model.parameters())
    assert snap2.norm_var >= 0, "Variance should be >= 0"

    # Enable cosine
    tracker.enable_cosine()
    model.zero_grad()
    y3 = model(torch.randn(2, 4))
    y3.sum().backward()
    norm3 = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    snap3 = tracker.update(float(norm3), model.parameters())
    assert snap3.cosine_sim is None, "First cosine after enable should be None (no prev)"

    model.zero_grad()
    y4 = model(torch.randn(2, 4))
    y4.sum().backward()
    norm4 = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    snap4 = tracker.update(float(norm4), model.parameters())
    assert snap4.cosine_sim is not None, "Should have cosine after second step"
    assert -1.0 <= snap4.cosine_sim <= 1.0, f"Cosine should be in [-1, 1], got {snap4.cosine_sim}"

    d = snap4.as_dict()
    assert "norm" in d and "cosine_sim" in d

    tracker.reset()
    assert tracker.last_snapshot is None

    print("  PASS: test_grad_tracker")


def test_hessian_trace() -> None:
    """Verify HessianTraceEstimator produces a positive trace for a convex loss."""
    ht_mod = _load_module("core.lulynx_trainer.hessian_trace", TRAINER_ROOT / "hessian_trace.py")
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 4)
    x = torch.randn(8, 4)
    target = torch.randn(8, 4)
    loss = ((model(x) - target) ** 2).mean()

    estimator = ht_mod.HessianTraceEstimator(num_vectors=3)
    params = [p for p in model.parameters() if p.requires_grad]
    snap = estimator.estimate(loss, params)

    assert snap.trace > 0, f"Trace of MSE Hessian should be > 0, got {snap.trace}"
    assert snap.num_params > 0
    assert snap.num_vectors == 3

    d = snap.as_dict()
    assert "trace" in d

    # Verify loss.backward() still works after estimate
    loss.backward()

    print("  PASS: test_hessian_trace")


def test_forgetting_probe() -> None:
    """Verify ForgettingProbe capture, baseline, and ratio tracking."""
    fp_mod = _load_module("core.lulynx_trainer.forgetting_probe", TRAINER_ROOT / "forgetting_probe.py")
    import torch

    probe = fp_mod.ForgettingProbe(num_anchors=2, warning_ratio=1.3, critical_ratio=1.8)
    assert not probe.has_anchors, "Should have no anchors before capture"

    fake_batches = [{"pixel_values": torch.randn(2, 3, 8, 8), "input_ids": torch.randint(0, 100, (2, 4))} for _ in range(4)]

    class FakeLoader:
        def __iter__(self):
            return iter(fake_batches)

    probe.capture_anchors(FakeLoader())
    assert probe.has_anchors, "Should have anchors after capture"
    assert len(probe._anchor_batches) == 2, f"Expected 2 anchors, got {len(probe._anchor_batches)}"

    call_count = [0]
    def fake_val_step(batch):
        call_count[0] += 1
        return 0.05

    snap = probe.probe(fake_val_step, step=10)
    assert snap is not None, "First probe should set baseline and return snapshot"
    assert snap.baseline_loss > 0
    assert snap.ratio == 1.0 or abs(snap.ratio - 1.0) < 0.01
    assert snap.trend == "stable"
    assert snap.score >= 99

    def drifting_val_step(batch):
        return 0.08

    snap2 = probe.probe(drifting_val_step, step=20)
    assert snap2.ratio > 1.0, f"Ratio should be >1 for drifting loss, got {snap2.ratio}"

    d = snap2.as_dict()
    assert "ratio" in d and "trend" in d and "score" in d

    print("  PASS: test_forgetting_probe")


def test_manifold_tracker() -> None:
    """Verify ManifoldTracker snapshot capture and PCA computation."""
    mt_mod = _load_module("core.lulynx_trainer.manifold_tracker", TRAINER_ROOT / "manifold_tracker.py")
    import torch
    import torch.nn as nn

    tracker = mt_mod.ManifoldTracker()
    model = nn.Linear(8, 4)
    params = [p for p in model.parameters() if p.requires_grad]

    for step in range(6):
        with torch.no_grad():
            for p in params:
                p.add_(torch.randn_like(p) * 0.01)
        tracker.snapshot(step, params, loss=0.1 - step * 0.01)

    assert tracker.num_snapshots == 6

    result = tracker.compute_pca()
    assert result is not None, "PCA should work with >=4 snapshots"
    assert len(result.points) == 6
    assert len(result.variance_explained) == 3
    assert result.num_params > 0

    for pt in result.points:
        assert hasattr(pt, "x") and hasattr(pt, "y") and hasattr(pt, "z")

    total_var = sum(result.variance_explained)
    assert 0 < total_var <= 1.01, f"Variance explained sum should be <=1, got {total_var}"

    d = result.as_dict()
    assert "points" in d and "variance_explained" in d

    print("  PASS: test_manifold_tracker")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    print("Monitoring Features Smoke Tests")
    print("=" * 40)
    test_peak_vram_capture()
    test_attention_stats()
    test_icu_health_score()
    test_audit_mode_override()
    test_audit_mode_auto()
    test_attn_entropy_probe()
    test_loss_tracker()
    test_dataset_analyzer()
    test_vram_estimator()
    test_act_drift_tracker()
    test_lr_finder()
    test_grad_tracker()
    test_hessian_trace()
    test_forgetting_probe()
    test_manifold_tracker()
    print("=" * 40)
    print("All monitoring smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

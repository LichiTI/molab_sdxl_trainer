# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Local wiring + parity smoke for the SRA2/HASTE alignment auxiliary loss.

This proves the SRA2 VAE self-representation alignment loss (gated by the HASTE
schedule guard) is wired into the *real* training loop as a default-off,
strategy-selectable auxiliary loss, and that the default keeps the loss path
bitwise-identical to legacy behavior.

It exercises the REAL seams (no copies):
  * ``TrainingLoop._compute_sra2_haste_loss`` -- bound to a light owner so the
    actual gating logic runs (off -> None, no capture -> None, no target ->
    None, schedule-gated -> None, active -> finite loss).
  * ``training_step_loss_execution_handler._apply_auxiliary_losses`` -- the real
    fold-in: when the method returns ``None`` the total loss is bitwise the base
    loss; when it returns a tensor the total loss is ``base + sra2``.
  * ``REPAFeatureCapture`` -- the same generic forward-hook capture REPA uses.
  * ``TrainingConfig`` field defaults + ``TrainingLoop.__init__`` signature --
    the config -> trainer -> loop contract carrying the new strategy.

Checks (CPU, seconds, no pytest -- run as a script):

  1. CONFIG defaults:    all 13 ``sra2_haste_*`` fields exist with default-off
     values (enabled=False, schedule neutral).
  2. LOOP signature:     ``TrainingLoop.__init__`` accepts ``sra2_haste_enabled``
     / ``sra2_haste_capture_layers`` / ``sra2_haste_policy``.
  3. METHOD off:         disabled -> ``None`` (parity: contributes nothing).
  4. METHOD no capture:  enabled but no capture -> ``None``.
  5. METHOD no target:   enabled + 3D capture but no latent/VAE batch key -> None.
  6. METHOD schedule:    enabled + valid inputs but ``start_step`` in the future
     -> ``None`` (HASTE guard wired).
  7. METHOD active:      enabled + 3D capture + latent target + in-window ->
     a finite, positive, scalar loss tensor.
  8. HANDLER parity:     fold-in with method -> None leaves the loss bitwise equal.
  9. HANDLER additive:   fold-in with method -> 0.5 yields ``base + 0.5``.

Run:
  backend/env/python-flashattention/python.exe \
    backend/core/lulynx_trainer/sra2_haste_loss_wiring_smoke.py
"""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    here = Path(__file__).resolve()
    backend_root = here.parents[2]          # .../backend  -> exposes `core`
    repo_root = here.parents[3]             # repo root     -> exposes `backend`
    for path in (str(repo_root), str(backend_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.repa import REPAFeatureCapture
from core.lulynx_trainer.training_loop import TrainingLoop
from core.lulynx_trainer.training_step_loss_execution_handler import _apply_auxiliary_losses

MODEL_SEED = 0
DIM = 8
BATCH = 2
TOKENS = 4

EXPECTED_CONFIG_DEFAULTS = {
    "sra2_haste_enabled": False,
    "sra2_haste_capture_layers": "",
    "sra2_haste_loss_type": "cosine",
    "sra2_haste_base_weight": 1.0,
    "sra2_haste_start_step": 0,
    "sra2_haste_stop_step": -1,
    "sra2_haste_decay_start_step": -1,
    "sra2_haste_decay_end_step": -1,
    "sra2_haste_min_weight": 0.0,
    "sra2_haste_plateau_patience": 0,
    "sra2_haste_min_relative_improvement": 0.0,
    "sra2_haste_normalize_targets": True,
    "sra2_haste_stop_grad_target": True,
}


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.proj(x)


class _Tiny(nn.Module):
    """Minimal model whose ``block.proj`` output is a 3D [B, tokens, hidden]."""

    def __init__(self, dim: int = DIM):
        super().__init__()
        self.block = _Block(dim)

    def forward(self, x):
        return self.block(x)


class _MethodOwner:
    """Light stand-in carrying exactly the attributes the method reads."""

    def __init__(self, *, enabled, capture, policy, global_step=0, total_steps=100):
        self.sra2_haste_enabled = enabled
        self.sra2_haste_capture = capture
        self.sra2_haste_policy = dict(policy or {})
        self._sra2_haste_warned = False
        self.global_step = global_step
        self.total_steps = total_steps


class _HandlerOwner:
    """Light stand-in for the auxiliary-loss handler (all other aux paths off)."""

    def __init__(self, sra2_value):
        self._loss_tracker = None
        self.lulynx_wrapper = None
        self.dop = None
        self.b_tier_runtime = None
        self.global_step = 0
        self._sra2_value = sra2_value

    def _compute_repa_loss(self, batch, prompt_embeds):
        return None

    def _compute_sra2_haste_loss(self, batch, prompt_embeds):
        return self._sra2_value


def _bind_method(owner):
    return types.MethodType(TrainingLoop._compute_sra2_haste_loss, owner)


def _capture_3d_feature():
    """Install REPAFeatureCapture and forward once to capture a 3D feature."""
    torch.manual_seed(MODEL_SEED)
    model = _Tiny()
    capture = REPAFeatureCapture(model, ["block.proj"]).install()
    model(torch.randn(BATCH, TOKENS, DIM))
    return capture


def _config_field_defaults():
    fields = getattr(UnifiedTrainingConfig, "model_fields", None)
    if fields is None:  # pydantic v1 fallback
        fields = UnifiedTrainingConfig.__fields__
    return {name: getattr(field, "default", None) for name, field in fields.items()}


def run() -> dict:
    results: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": bool(ok), "detail": detail})

    # --- 1. CONFIG defaults --------------------------------------------------
    defaults = _config_field_defaults()
    missing = [k for k in EXPECTED_CONFIG_DEFAULTS if k not in defaults]
    check("config_fields_present", not missing, f"missing={missing}")
    bad = {k: defaults.get(k) for k, v in EXPECTED_CONFIG_DEFAULTS.items() if defaults.get(k) != v}
    check("config_defaults_off", not bad, f"unexpected={bad}")

    # --- 2. LOOP signature ---------------------------------------------------
    params = set(inspect.signature(TrainingLoop.__init__).parameters)
    needed = {"sra2_haste_enabled", "sra2_haste_capture_layers", "sra2_haste_policy"}
    check("loop_signature_accepts_kwargs", needed <= params, f"missing={sorted(needed - params)}")

    latent_batch = {"latents": torch.randn(BATCH, 4, 2, 2)}

    # --- 3. METHOD off (parity: no contribution) -----------------------------
    off_owner = _MethodOwner(enabled=False, capture=_capture_3d_feature(), policy={})
    check("method_off_returns_none", _bind_method(off_owner)(latent_batch, {}) is None)

    # --- 4. METHOD enabled but no capture ------------------------------------
    nocap_owner = _MethodOwner(enabled=True, capture=None, policy={})
    check("method_no_capture_returns_none", _bind_method(nocap_owner)(latent_batch, {}) is None)

    # --- 5. METHOD enabled + capture but no latent/VAE target ----------------
    notgt_owner = _MethodOwner(enabled=True, capture=_capture_3d_feature(), policy={})
    notgt_loss = _bind_method(notgt_owner)({}, {})
    check("method_no_target_returns_none", notgt_loss is None and notgt_owner._sra2_haste_warned is True)

    # --- 6. METHOD schedule-gated (start_step in the future) -----------------
    sched_owner = _MethodOwner(
        enabled=True,
        capture=_capture_3d_feature(),
        policy={"start_step": 100},
        global_step=0,
    )
    check("method_schedule_gated_returns_none", _bind_method(sched_owner)(latent_batch, {}) is None)

    # --- 7. METHOD active -> finite positive scalar loss ---------------------
    active_owner = _MethodOwner(enabled=True, capture=_capture_3d_feature(), policy={}, global_step=0)
    active_loss = _bind_method(active_owner)(latent_batch, {})
    active_scalar = None if active_loss is None else float(active_loss.detach())
    ok_active = (
        isinstance(active_loss, torch.Tensor)
        and active_loss.ndim == 0
        and bool(torch.isfinite(active_loss))
        and active_scalar > 0.0
    )
    check("method_active_returns_loss", ok_active, f"loss={'none' if active_scalar is None else f'{active_scalar:.4f}'}")

    # --- 8. HANDLER fold-in parity (method -> None) --------------------------
    base = torch.tensor(1.0)
    off_loss, _ = _apply_auxiliary_losses(
        owner=_HandlerOwner(None),
        batch={},
        prompt_embeds={},
        noise_pred=torch.zeros(1),
        noisy_latents=torch.zeros(1),
        timesteps=torch.zeros(1),
        loss=base,
        do_backward=False,
        tracker_value=None,
        loss_scalars=None,
    )
    check("handler_off_parity_bitwise", torch.equal(off_loss, base), f"loss={float(off_loss)}")

    # --- 9. HANDLER fold-in additive (method -> 0.5) -------------------------
    on_loss, _ = _apply_auxiliary_losses(
        owner=_HandlerOwner(torch.tensor(0.5)),
        batch={},
        prompt_embeds={},
        noise_pred=torch.zeros(1),
        noisy_latents=torch.zeros(1),
        timesteps=torch.zeros(1),
        loss=torch.tensor(1.0),
        do_backward=False,
        tracker_value=None,
        loss_scalars=None,
    )
    check("handler_on_additive", abs(float(on_loss) - 1.5) < 1e-6, f"loss={float(on_loss)}")

    passed = sum(1 for r in results if r["ok"])
    return {
        "smoke": "sra2_haste_loss_wiring_smoke",
        "passed": passed,
        "total": len(results),
        "ok": passed == len(results),
        "results": results,
    }


def main() -> int:
    report = run()
    for r in report["results"]:
        status = "PASS" if r["ok"] else "FAIL"
        line = f"  [{status}] {r['check']}"
        if r["detail"]:
            line += f"  ({r['detail']})"
        print(line)
    print(f"\n[sra2_haste_loss_wiring_smoke] {report['passed']}/{report['total']} checks passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

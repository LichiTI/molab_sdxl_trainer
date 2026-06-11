# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Fair-arena convergence benchmark for the MN-LoRA optimizer suite.

The LoRA / 40-step benchmark is the *worst* case for MN-LoRA's heavy machinery
(GSP / trust-region / Fisher): LoRA is already low-rank, and 40 steps never lets
a second-order-ish method amortize its setup cost. This benchmark instead puts
the suite in the regime it was *designed* for:

  * **full-parameter** training (every weight matrix trainable -> GSP/TG-WD/
    trust-region actually engage on real 2D weights),
  * a **genuinely ill-conditioned** problem -- inputs are scaled per-dim across
    three orders of magnitude *and rotated* by a fixed orthogonal matrix, so the
    parameter Hessian is ill-conditioned AND non-axis-aligned. Adam's diagonal
    preconditioner cannot undo the rotation, which is exactly the gap an SVD /
    subspace method could close,
  * a **long horizon** (hundreds of steps) so curvature methods can pay back
    their early cost.

It calls the REAL ``wrap_optimizer`` and the REAL ``mn_presets`` -- nothing about
the suite is re-implemented here, so the verdict is honest. Only the optimizer
"arm" changes; model init, data, and batch order are identical across arms.

This is a CPU, seconds-per-arm, local-runnable test (small model on purpose).
KFAC-lite and effective-delta are LoRA-module-coupled and stay no-ops here with
no injected modules -- that itself is reported as a finding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.training_components.mn_lora.hijacker import wrap_optimizer
from core.training_components.mn_lora.mn_presets import (
    select_mnlora_preset,
    split_mnlora_preset,
)


# Each arm = the set of MN-LoRA components turned ON (everything else off).
# ``adamw`` is special: no wrap at all (the reference line).
# ``mn_bare`` wraps with everything off -> must reproduce adamw (transparency
# sanity). ``mn_convergence`` is the fair convergence stack (NO Fisher, which is
# an anti-forgetting tool with no job in single-task training). ``mn_full`` is
# the production-style bundle including Fisher, shown to quantify its drag.
ARM_COMPONENTS = {
    "adamw": None,
    "mn_bare": set(),
    "gsp_only": {"gsp"},
    "tgwd_only": {"tgwd"},
    "trust_region_only": {"trust_region"},
    "plus_plus_only": {"plus_plus"},
    "fisher_only": {"fisher"},
    "mn_convergence": {"gsp", "tgwd", "trust_region", "plus_plus", "pilot"},
    "mn_full": {"gsp", "tgwd", "trust_region", "plus_plus", "pilot", "fisher", "effective_delta"},
}
ARMS = tuple(ARM_COMPONENTS.keys())


@dataclass
class ArmResult:
    arm: str
    success: bool
    failed_reason: str = ""
    optimizer_type: str = ""
    final_train_loss: float = 0.0
    min_train_loss: float = 0.0
    final_eval_loss: float = 0.0
    late_train_std: float = 0.0
    mean_step_ms: float = 0.0
    train_curve: list = field(default_factory=list)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, width: int, hidden_layers: int, out_dim: int):
        super().__init__()
        dims = [in_dim] + [width] * hidden_layers + [out_dim]
        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_model(in_dim, width, hidden_layers, out_dim, seed) -> _MLP:
    torch.manual_seed(seed)
    return _MLP(in_dim, width, hidden_layers, out_dim).double()


def _make_dataset(*, in_dim, out_dim, width, hidden_layers, cond_exp, n_train, n_eval, seed):
    """Realizable regression with rotated, ill-conditioned inputs (float64)."""
    g = torch.Generator().manual_seed(seed)
    scales = torch.logspace(0.0, float(cond_exp), in_dim, dtype=torch.float64)
    rot, _ = torch.linalg.qr(torch.randn(in_dim, in_dim, generator=g, dtype=torch.float64))

    def transform(x):
        return (x * scales) @ rot

    teacher = _build_model(in_dim, width, hidden_layers, out_dim, seed + 7)
    for p in teacher.parameters():
        p.requires_grad_(False)

    x_train = transform(torch.randn(n_train, in_dim, generator=g, dtype=torch.float64))
    x_eval = transform(torch.randn(n_eval, in_dim, generator=g, dtype=torch.float64))
    with torch.no_grad():
        y_train = teacher(x_train)
        y_eval = teacher(x_eval)
    cond_number = float((scales.max() / scales.min()) ** 2)
    return x_train, y_train, x_eval, y_eval, cond_number


def _mn_kwargs(components: set, param_names: dict) -> dict:
    """Faithfully mirror trainer.py's wrap_optimizer call, toggling components."""
    preset = split_mnlora_preset(select_mnlora_preset("slim", ""))
    gsp_config = dict(preset["gsp_config"])
    gsp_config.update({
        "k_ratio": 0.5, "update_interval": 20, "adaptive_k": True, "lazy_update": True,
        "residual_threshold": 0.3, "min_k_ratio": 0.2, "max_k_ratio": 0.8,
        "lazy_threshold": 0.5, "precondition_mode": "grad_ema", "svd_precond_beta": 0.5,
        "precond_min_scale": 0.25, "precond_max_scale": 4.0, "coord_curv_beta": 0.95,
        "precond_clip": 3.0, "precond_eps": 1e-6, "adaptive_sparse_enabled": True,
        "adaptive_sparse_warmup_steps": 10, "adaptive_sparse_refresh_interval": 20,
        "adaptive_sparse_hot_ratio": 0.20, "adaptive_sparse_warm_ratio": 0.0,
        "adaptive_sparse_warm_interval": 4, "adaptive_sparse_cold_interval": 16,
        "adaptive_sparse_min_hot_layers": 16, "adaptive_sparse_zero_cold_after": 3,
    })
    tgwd_config = dict(preset["tgwd_config"])
    tgwd_config.update({"alpha": 1.0, "n_probes": 1, "finite_diff_eps": 1e-3})
    pilot_config = dict(preset["pilot_config"])
    return dict(
        enable_gsp="gsp" in components,
        enable_tgwd="tgwd" in components,
        enable_pilot="pilot" in components,
        gsp_config=gsp_config,
        tgwd_config=tgwd_config,
        pilot_config=pilot_config,
        plus_plus_config={
            "enabled": "plus_plus" in components,
            "lr_up": 1.03, "lr_down": 0.90, "min_mult": 0.25, "max_mult": 2.5,
            "lora_up_max_mult": 1.5, "protected_max_mult": 1.0, "update_rms_cap": 0.02,
            "rank_adapt": True, "module_adapt": True,
        },
        kfac_lite_config={"enabled": "kfac" in components},
        trust_region_config={
            "enabled": "trust_region" in components,
            "max_update_rms_ratio": 0.01, "max_update_norm_ratio": 0.10,
            "hotspot_only": False,
        },
        effective_delta_config={
            "enabled": "effective_delta" in components,
            "clip_enabled": True, "max_norm_ratio": 0.25, "max_rms_ratio": 0.05,
            "fisher_weighted": True, "fisher_beta": 0.95, "fisher_strength": 1.0,
            "fisher_max_weight": 4.0,
        },
        fisher_ewc_config={
            "enabled": "fisher" in components,
            "lambda_ewc": 1e-4, "fisher_beta": 0.95, "start_step": 1,
            "update_interval": 5, "max_penalty_norm_ratio": 0.25,
        },
        gradient_conflict_config={"enabled": False, "reduction": "sum"},
        lora_modules={},
        param_names=param_names,
    )


def _run_arm(arm, *, data, steps, batch, lr, model_dims, model_seed, batch_index) -> ArmResult:
    x_train, y_train, x_eval, y_eval = data
    in_dim, width, hidden_layers, out_dim = model_dims
    components = ARM_COMPONENTS[arm]
    try:
        model = _build_model(in_dim, width, hidden_layers, out_dim, model_seed)
        base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        if components is None:
            optimizer = base
        else:
            param_names = {id(p): n for n, p in model.named_parameters()}
            optimizer = wrap_optimizer(base, **_mn_kwargs(components, param_names))

        loss_fn = nn.MSELoss()
        curve: list = []
        step_ms: list = []
        for s in range(steps):
            idx = batch_index[s]
            xb, yb = x_train[idx], y_train[idx]
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            step_ms.append((time.perf_counter() - t0) * 1000.0)
            curve.append(float(loss.detach()))

        with torch.no_grad():
            final_eval = float(loss_fn(model(x_eval), y_eval))
        late = curve[-50:] if len(curve) >= 50 else curve
        late_std = float(torch.tensor(late).std())
        return ArmResult(
            arm=arm, success=True, optimizer_type=type(optimizer).__name__,
            final_train_loss=curve[-1], min_train_loss=min(curve),
            final_eval_loss=final_eval, late_train_std=late_std,
            mean_step_ms=sum(step_ms) / len(step_ms),
            train_curve=[round(c, 6) for c in curve],
        )
    except Exception as exc:  # a component failing outside LoRA is itself a finding
        import traceback
        return ArmResult(arm=arm, success=False, failed_reason=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def run_benchmark(args) -> dict:
    torch.use_deterministic_algorithms(True, warn_only=True)
    in_dim, width, hidden_layers, out_dim = args.in_dim, args.width, args.hidden_layers, args.out_dim
    x_train, y_train, x_eval, y_eval, cond_number = _make_dataset(
        in_dim=in_dim, out_dim=out_dim, width=width, hidden_layers=hidden_layers,
        cond_exp=args.cond_exp, n_train=args.n_train, n_eval=args.n_eval, seed=args.seed,
    )
    # Fixed batch sequence -> identical data order for every arm.
    g = torch.Generator().manual_seed(args.seed + 101)
    batch_index = [torch.randint(0, args.n_train, (args.batch,), generator=g) for _ in range(args.steps)]

    arms = args.arms or list(ARMS)
    results: list = []
    for arm in arms:
        print(f"[fullft-fair] arm={arm} steps={args.steps} -> running", flush=True)
        res = _run_arm(
            arm, data=(x_train, y_train, x_eval, y_eval), steps=args.steps,
            batch=args.batch, lr=args.lr, model_dims=(in_dim, width, hidden_layers, out_dim),
            model_seed=args.seed + 1, batch_index=batch_index,
        )
        results.append(res)
        if res.success:
            print(f"[fullft-fair] {arm}: OK eval={res.final_eval_loss:.6f} "
                  f"min_train={res.min_train_loss:.6f} late_std={res.late_train_std:.6f} "
                  f"step_ms={res.mean_step_ms:.3f} opt={res.optimizer_type}", flush=True)
        else:
            print(f"[fullft-fair] {arm}: FAIL {res.failed_reason.splitlines()[0]}", flush=True)

    by_arm = {r.arm: r for r in results}
    base = by_arm.get("adamw")
    comparison = {}
    if base is not None and base.success:
        for r in results:
            if r.arm == "adamw" or not r.success:
                continue
            comparison[r.arm] = {
                "eval_minus_adamw": round(r.final_eval_loss - base.final_eval_loss, 6),
                "min_train_minus_adamw": round(r.min_train_loss - base.min_train_loss, 6),
                "late_std_ratio": round(r.late_train_std / base.late_train_std, 4) if base.late_train_std else None,
                "step_ms_ratio": round(r.mean_step_ms / base.mean_step_ms, 4) if base.mean_step_ms else None,
            }
    sanity = None
    if base is not None and base.success and by_arm.get("mn_bare") and by_arm["mn_bare"].success:
        delta = abs(by_arm["mn_bare"].final_eval_loss - base.final_eval_loss)
        sanity = {"mn_bare_vs_adamw_eval_abs_delta": round(delta, 8), "transparent": delta < 1e-4}

    report = {
        "benchmark": "mn_lora_fullft_fair_benchmark",
        "regime": "full-parameter / ill-conditioned (rotated) / long-horizon",
        "model": {"in_dim": in_dim, "width": width, "hidden_layers": hidden_layers, "out_dim": out_dim},
        "problem": {"condition_number": round(cond_number, 1), "cond_exp": args.cond_exp,
                    "n_train": args.n_train, "n_eval": args.n_eval},
        "steps": args.steps, "batch": args.batch, "lr": args.lr, "seed": args.seed,
        "dtype": "float64",
        "sanity": sanity,
        "comparison_vs_adamw": comparison,
        "results": [asdict(r) for r in results],
    }
    return report


def main():
    p = argparse.ArgumentParser(description="MN-LoRA fair-arena (full-param, ill-conditioned, long-horizon) benchmark.")
    p.add_argument("--arms", nargs="*", choices=list(ARMS), default=None)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--in-dim", dest="in_dim", type=int, default=48)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--hidden-layers", dest="hidden_layers", type=int, default=2)
    p.add_argument("--out-dim", dest="out_dim", type=int, default=4)
    p.add_argument("--cond-exp", dest="cond_exp", type=float, default=3.0)
    p.add_argument("--n-train", dest="n_train", type=int, default=2048)
    p.add_argument("--n-eval", dest="n_eval", type=int, default=1024)
    p.add_argument("--json", type=str, default="temp/mn_lora_fullft_fair.json")
    args = p.parse_args()

    report = run_benchmark(args)

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[fullft-fair] report -> {args.json}")
    print(json.dumps(report["comparison_vs_adamw"], indent=2))
    if report["sanity"]:
        print(f"[fullft-fair] transparency sanity: {report['sanity']}")


if __name__ == "__main__":
    main()

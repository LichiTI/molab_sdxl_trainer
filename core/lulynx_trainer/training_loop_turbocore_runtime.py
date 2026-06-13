# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Turbocore native-update runtime cluster extracted verbatim from ``training_loop.py``
as a mixin.

These methods (native-update readiness/runtime-profile refresh, runtime context,
training-executor config + build for native-update / simple / quantized /
kahan-adamw8bit optimizers, direct-grad executor preparation, executor
sync/close, and checkpoint-state get/load) run as bound methods of the
``TrainingLoop`` instance — identical ``self`` semantics, identical call sites
(including the public ``get_turbocore_*`` / ``load_turbocore_*`` surface, kept
reachable via MRO). Behaviour is unchanged; this split only carves the single
largest cohesive cluster out of ``training_loop.py`` to keep it navigable.

The three large methods here (``_turbocore_native_update_training_executor_config``,
``_get_turbocore_native_update_training_executor``, ``_normalize_turbocore_quantized_optimizer_kind``)
are essential-complexity executor builders — moved verbatim, not restructured.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import torch

from .training_step_orchestrator_handlers import (
    run_lulynx_turbocore_native_update_runtime_profile_stage_handler,
)
from .turbocore_native_update_readiness_adapter import (
    build_native_update_runtime_context,
    build_training_loop_native_update_readiness,
)
from core.turbocore_kahan_adamw8bit_training_executor import build_kahan_adamw8bit_training_executor
from core.turbocore_native_update_probe_cache import can_retain_native_update_probe_evidence
from core.turbocore_native_update_training_executor import build_native_update_training_executor
from core.turbocore_simple_optimizer_training_executor import build_simple_optimizer_training_executor
from core.turbocore_v5_stream_lifetime_lease_evidence import build_single_step_lifetime_lease_request

logger = logging.getLogger(__name__)


class TrainingLoopTurbocoreRuntimeMixin:
    def _refresh_turbocore_native_update_readiness(
        self,
        shadow_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            self._turbocore_native_update_readiness = build_training_loop_native_update_readiness(
                optimizer=self.optimizer,
                params=self._get_trainable_params(),
                mode=self._turbocore_native_update_gate.config.mode,
                runtime_context=self._turbocore_native_update_runtime_context(),
                shadow_config=self._turbocore_update_shadow.config,
                save_owner_state=bool(self._turbocore_update_shadow_save_owner_state),
                shadow_report=shadow_report,
            )
        except Exception as exc:
            self._turbocore_native_update_readiness = {
                "schema_version": 1,
                "report": "turbocore_native_update_readiness_v0",
                "ok": False,
                "training_path_enabled": False,
                "native_kernel_present": False,
                "performance_test_ready": False,
                "stream_lifetime_bound": False,
                "error": f"{type(exc).__name__}: {exc}",
                "blocked_reasons": ["readiness_error"],
            }
            logger.debug("TurboCore native update readiness probe skipped: %s", exc)
        return dict(self._turbocore_native_update_readiness)

    def _refresh_turbocore_native_update_runtime_profile(
        self,
        *,
        shadow_report: Optional[Dict[str, Any]] = None,
        gate_report: Optional[Dict[str, Any]] = None,
        dispatch_arming: Optional[Dict[str, Any]] = None,
        dispatch_runtime_report: Optional[Dict[str, Any]] = None,
        dispatch_recovery: Optional[Dict[str, Any]] = None,
        diagnostic_replay: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        execution = run_lulynx_turbocore_native_update_runtime_profile_stage_handler(
            shadow=self._turbocore_update_shadow,
            gate=self._turbocore_native_update_gate,
            readiness=self._turbocore_native_update_readiness,
            runtime_context=runtime_context or self._turbocore_native_update_runtime_context(),
            dispatch_runtime=self._turbocore_native_update_dispatch_runtime,
            dispatch_armer=self._turbocore_native_update_dispatch_armer,
            shadow_report=shadow_report,
            gate_report=gate_report,
            dispatch_arming=dispatch_arming,
            dispatch_runtime_report=dispatch_runtime_report,
            dispatch_recovery=dispatch_recovery,
            diagnostic_replay=diagnostic_replay,
            step=step,
            memory_optimization_state=self.memory_optimization_state,
        )
        self._turbocore_native_update_runtime_profile = dict(execution.profile or {})
        return dict(self._turbocore_native_update_runtime_profile)

    def get_turbocore_native_update_runtime_profile(self) -> Dict[str, Any]:
        if not getattr(self, "_turbocore_native_update_runtime_profile", None):
            self._refresh_turbocore_native_update_runtime_profile()
        return dict(getattr(self, "_turbocore_native_update_runtime_profile", {}) or {})

    def _turbocore_native_update_runtime_context(self) -> Dict[str, Any]:
        context = build_native_update_runtime_context(
            multi_gpu=self.multi_gpu,
            num_processes=self.num_processes,
            num_machines=self.num_machines,
            deepspeed=self.deepspeed,
            gradient_release_active=self._gradient_release_manager is not None,
        )
        explicit_training = bool(
            self._turbocore_native_update_gate.config.dispatch_enabled
            and self._turbocore_native_update_training_path_enabled
            and self._turbocore_native_update_require_native_cuda
        )
        context.update(
            {
                "training_path_enabled": explicit_training,
                "native_update_training_dispatch_enabled": explicit_training,
                "native_update_runtime_dispatch_available": explicit_training,
                "native_update_executor_present": explicit_training,
                "native_update_runtime_execution_guard_enabled": explicit_training,
                "native_update_training_mutation_guard_enabled": explicit_training,
                "native_update_allow_short_training_dispatch_evidence": bool(
                    explicit_training and self._turbocore_native_update_gate.config.allow_missing_native_kernel
                ),
                "native_update_owner_gradient_sync_guard_enabled": explicit_training,
                "native_update_owner_gradient_sync_bound": explicit_training,
                "native_update_flat_owner_training_guard_enabled": explicit_training,
                "native_update_flat_owner_bound": explicit_training,
                "native_update_training_dispatch_kernel_guard_enabled": explicit_training,
                "native_update_training_dispatch_kernel_bound": explicit_training,
                "native_update_stream_lifetime_ownership_guard_enabled": explicit_training,
                "native_update_stream_lifetime_ownership_bound": explicit_training,
                "native_update_direct_gradient_write_guard_enabled": bool(
                    explicit_training and self._turbocore_update_shadow.config.direct_grad
                ),
                "native_update_direct_gradient_write_bound": bool(
                    explicit_training and self._turbocore_update_shadow.config.direct_grad
                ),
                "native_update_training_executor_config": self._turbocore_native_update_training_executor_config(),
            }
        )
        return context

    def _turbocore_native_update_training_executor_config(self) -> Dict[str, Any]:
        group = self.optimizer.param_groups[0] if self.optimizer and self.optimizer.param_groups else {}
        betas = group.get("betas", (0.9, 0.999))
        if not isinstance(betas, (tuple, list)):
            betas = (0.9, 0.999)

        def _optional_beta(index: int, default: float | None) -> float | None:
            value = betas[index] if len(betas) > index else default
            return None if value is None else float(value)

        lr = group.get("lr")
        if lr is None:
            lr = getattr(self, "learning_rate", 0.0)
        eps_value = group.get("eps", 1e-8)
        if isinstance(eps_value, (tuple, list)):
            eps_value = eps_value[0] if eps_value else 1e-8
        config = {
            "optimizer_kind": self._turbocore_native_update_quantized_optimizer_kind,
            "lr": float(lr or 0.0),
            "betas": [_optional_beta(0, 0.9), _optional_beta(1, 0.999)],
            "eps": float(eps_value),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "max_grad_norm": float(self.max_grad_norm or 0.0),
            "prefer_native_cuda": True,
            "require_native_cuda": bool(self._turbocore_native_update_require_native_cuda),
            "prefer_triton": False,
            "sync_optimizer_state_each_step": not bool(self._turbocore_native_update_defer_state_sync),
            "sync_params_from_optimizer_each_step": not bool(self._turbocore_native_update_defer_state_sync),
            "sync_pytorch_optimizer_state_each_step": not bool(self._turbocore_native_update_defer_state_sync),
            "direct_grad": bool(self._turbocore_native_update_direct_grad_executor_enabled()),
            "native_runtime_synchronization_policy": self._turbocore_native_update_runtime_synchronization_policy,
            "native_runtime_stream_lifetime_lease_evidence": self._turbocore_native_update_stream_lifetime_lease_request(),
        }
        if self._turbocore_native_update_quantized_optimizer_kind == "adamg":
            beta3 = betas[2] if len(betas) >= 3 else group.get("beta3", 0.95)
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["p"] = float(group.get("p", getattr(self.optimizer, "p", 0.2)))
            config["q"] = float(group.get("q", getattr(self.optimizer, "q", 0.24)))
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(group.get("maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "schedulefree_sgd":
            config["momentum"] = float(group.get("momentum", 0.9))
            config["warmup_steps"] = int(group.get("warmup_steps", 0) or 0)
            config["r"] = float(group.get("r", 0.0))
            config["weight_lr_power"] = float(group.get("weight_lr_power", 2.0))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "schedulefree_radam":
            config["silent_sgd_phase"] = bool(group.get("silent_sgd_phase", True))
            config["r"] = float(group.get("r", 0.0))
            config["weight_lr_power"] = float(group.get("weight_lr_power", 2.0))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "radam":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["n_sma_threshold"] = int(getattr(self.optimizer, "n_sma_threshold", 5) or 5)
            config["degenerated_to_sgd"] = bool(getattr(self.optimizer, "degenerated_to_sgd", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind in {"muon", "distributedmuon"}:
            config["momentum"] = float(group.get("momentum", 0.95))
            config["ns_steps"] = int(group.get("ns_steps", 5) or 5)
            config["nesterov"] = bool(group.get("nesterov", True))
        elif self._turbocore_native_update_quantized_optimizer_kind == "sgdsai":
            config["momentum"] = float(group.get("momentum", 0.9))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "sm3":
            config["momentum"] = float(group.get("momentum", 0.0))
            config["beta"] = float(group.get("beta", 0.0))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "spam":
            config["density"] = float(getattr(self.optimizer, "density", 1.0))
            config["threshold"] = int(getattr(self.optimizer, "threshold", 0) or 0)
            config["grad_accu_steps"] = int(getattr(self.optimizer, "grad_accu_steps", 20) or 20)
            config["update_proj_gap"] = int(getattr(self.optimizer, "update_proj_gap", 500) or 500)
            config["warmup_epoch"] = int(getattr(self.optimizer, "warmup_epoch", 1) or 1)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "stablespam":
            config["gamma1"] = float(getattr(self.optimizer, "gamma1", 0.7))
            config["gamma2"] = float(getattr(self.optimizer, "gamma2", 0.9))
            config["theta"] = float(getattr(self.optimizer, "theta", 0.999))
            config["t_max"] = getattr(self.optimizer, "t_max", None)
            config["update_proj_gap"] = int(getattr(self.optimizer, "update_proj_gap", 1000) or 1000)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adadelta":
            config["rho"] = float(group.get("rho", 0.9))
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ftrl":
            config["lr_power"] = float(group.get("lr_power", -0.5))
            config["beta"] = float(group.get("beta", 0.0))
            config["lambda_1"] = float(group.get("lambda_1", 0.0))
            config["lambda_2"] = float(group.get("lambda_2", 0.0))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "diffgrad":
            config["rectify"] = bool(group.get("rectify", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adabelief":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["rectify"] = bool(group.get("rectify", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adabound":
            base_lrs = getattr(self.optimizer, "base_lrs", None)
            config["final_lr"] = float(group.get("final_lr", 1.0e-1))
            config["gamma"] = float(group.get("gamma", 1.0e-3))
            config["base_lr"] = float(base_lrs[0] if isinstance(base_lrs, list) and base_lrs else group.get("lr", lr or 1.0e-3))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "laprop":
            config["centered"] = bool(group.get("centered", False))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["cautious"] = bool(group.get("cautious", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adai":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["stable_weight_decay"] = bool(group.get("stable_weight_decay", False))
            config["dampening"] = float(group.get("dampening", 1.0))
            config["use_gc"] = bool(group.get("use_gc", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adopt":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["cautious"] = bool(group.get("cautious", False))
            config["stable_adamw"] = bool(group.get("stable_adamw", False))
            config["clip_enabled"] = getattr(self.optimizer, "clip_lambda", None) is not None
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "msvag":
            config["beta"] = float(group.get("beta", 0.9))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ademamix":
            beta3 = betas[2] if len(betas) >= 3 else 0.9999
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["alpha"] = float(group.get("alpha", 5.0))
            config["t_alpha_beta3"] = group.get("t_alpha_beta3", None)
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["cautious"] = bool(group.get("cautious", False))
            config["stable_adamw"] = bool(group.get("stable_adamw", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "simplifiedademamix":
            config["betas"] = [float(betas[0]), float(betas[1])]
            config["alpha"] = float(group.get("alpha", 0.0))
            config["beta1_warmup"] = group.get("beta1_warmup", None)
            config["min_beta1"] = float(group.get("min_beta1", 0.9))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "a2grad":
            config["beta"] = float(group.get("beta", 10.0))
            config["lips"] = float(group.get("lips", 10.0))
            config["rho"] = float(group.get("rho", 0.5))
            config["variant"] = str(getattr(self.optimizer, "variant", "uni") or "uni")
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "avagrad":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adanorm":
            config["r"] = float(group.get("r", 0.95))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "bcos":
            config["beta"] = float(group.get("beta", 0.9))
            beta2 = group.get("beta2", None)
            config["beta2"] = None if beta2 is None else float(beta2)
            config["mode"] = str(getattr(self.optimizer, "mode", "g") or "g")
            config["simple_cond"] = bool(getattr(self.optimizer, "simple_cond", False))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adagc":
            config["beta"] = float(group.get("beta", 0.98))
            config["lambda_abs"] = float(group.get("lambda_abs", 100.0))
            config["lambda_rel"] = float(group.get("lambda_rel", 1.05))
            config["warmup_steps"] = int(group.get("warmup_steps", 100) or 100)
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adasmooth":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adapnm":
            beta3 = betas[2] if len(betas) >= 3 else 1.0
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adan":
            beta3 = betas[2] if len(betas) >= 3 else 0.99
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["max_grad_norm"] = float(group.get("max_grad_norm", 0.0) or 0.0)
            config["use_gc"] = bool(group.get("use_gc", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ano":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["logarithmic_schedule"] = bool(getattr(self.optimizer, "logarithmic_schedule", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "amos":
            config["beta"] = float(group.get("beta", 0.999))
            config["momentum"] = float(group.get("momentum", 0.0) or 0.0)
            config["extra_l2"] = float(group.get("extra_l2", 0.0) or 0.0)
            config["c_coef"] = float(getattr(self.optimizer, "c_coef", 0.25))
            config["d_coef"] = float(getattr(self.optimizer, "d_coef", 0.25))
            config["foreach"] = group.get("foreach", False)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "apollo":
            config["scale_type"] = str(group.get("scale_type", "tensor") or "tensor")
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["correct_bias"] = bool(group.get("correct_bias", True))
            config["rank"] = group.get("rank", None)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "galore":
            config["rank"] = group.get("rank", None)
            config["update_proj_gap"] = int(group.get("update_proj_gap", 200) or 200)
            config["scale"] = float(group.get("scale", 1.0))
            config["projection_type"] = str(group.get("projection_type", "std") or "std")
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "fira":
            config["rank"] = group.get("rank", None)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "focus":
            config["gamma"] = float(group.get("gamma", 0.1))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "alig":
            config["max_lr"] = group.get("max_lr", None)
            config["momentum"] = float(group.get("momentum", 0.0) or 0.0)
            config["adjusted_momentum"] = bool(group.get("adjusted_momentum", False))
            config["maximize"] = bool(getattr(getattr(self.optimizer, "_base", self.optimizer), "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "alice":
            config["rank"] = int(group.get("rank", 2) or 2)
            config["leading_basis"] = int(group.get("leading_basis", 1) or 1)
            config["alpha"] = float(group.get("alpha", 0.3) or 0.3)
            config["alpha_c"] = float(group.get("alpha_c", 0.4) or 0.4)
            config["update_interval"] = int(group.get("update_interval", 2) or 2)
            config["gamma"] = float(group.get("gamma", 1.01) or 1.01)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adahessian":
            config["hessian_power"] = float(group.get("hessian_power", 1.0) or 1.0)
            config["update_period"] = int(getattr(self.optimizer, "update_period", 1) or 1)
            config["num_samples"] = int(getattr(self.optimizer, "num_samples", 1) or 1)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "kron":
            config["momentum"] = float(group.get("momentum", 0.0) or 0.0)
            config["memory_save_mode"] = group.get("memory_save_mode", "all_diag")
            config["balance_prob"] = float(getattr(self.optimizer, "balance_prob", 0.0) or 0.0)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "lorarite":
            config["relative_epsilon"] = bool(group.get("relative_epsilon", False))
            config["clip_unmagnified_grad"] = float(group.get("clip_unmagnified_grad", 0.0) or 0.0)
            config["update_capping"] = float(group.get("update_capping", 0.0) or 0.0)
            config["update_skipping"] = float(group.get("update_skipping", 0.0) or 0.0)
            config["apply_escape"] = bool(group.get("apply_escape", False))
            config["balance_param"] = bool(group.get("balance_param", False))
            config["lora_l_dim"] = int(group.get("lora_l_dim", 0) or 0)
            config["lora_r_dim"] = int(group.get("lora_r_dim", -1) or -1)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind in {"shampoo", "scalableshampoo", "soap"}:
            config["matrix_eps"] = float(group.get("matrix_eps", 1.0e-6) or 1.0e-6)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
            config["correct_bias"] = bool(group.get("correct_bias", True))
        elif self._turbocore_native_update_quantized_optimizer_kind == "conda":
            config["update_proj_gap"] = int(group.get("update_proj_gap", 2000) or 2000)
            config["scale"] = float(group.get("scale", 1.0))
            config["projection_type"] = str(group.get("projection_type", "std") or "std")
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "grams":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "srmm":
            config["beta"] = float(group.get("beta", 0.5))
            config["memory_length"] = group.get("memory_length", 100)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "splus":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ema_rate"] = float(group.get("ema_rate", 0.999))
            config["inverse_steps"] = int(group.get("inverse_steps", 100) or 100)
            config["nonstandard_constant"] = float(group.get("nonstandard_constant", 1.0e-3))
            config["max_dim"] = int(group.get("max_dim", 10000) or 10000)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "tam":
            config["momentum"] = float(group.get("momentum", 0.9))
            config["decay_rate"] = float(group.get("decay_rate", 0.9))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "swats":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["nesterov"] = bool(group.get("nesterov", False))
            config["phase"] = str(group.get("phase", "adam") or "adam")
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "sophiah":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["p"] = float(group.get("p", 1.0e-2))
            config["update_period"] = int(getattr(self.optimizer, "update_period", 10) or 10)
            config["num_samples"] = int(getattr(self.optimizer, "num_samples", 1) or 1)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "racs":
            config["beta"] = float(group.get("beta", 0.9))
            config["alpha"] = float(group.get("alpha", 0.05))
            config["gamma"] = float(group.get("gamma", 1.01))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "kate":
            config["delta"] = float(group.get("delta", 0.0))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "rose":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["centralize"] = bool(group.get("centralize", True))
            config["stabilize"] = bool(group.get("stabilize", True))
            config["bf16_sr"] = bool(group.get("bf16_sr", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "mars":
            config["lr_1d"] = float(group.get("lr_1d", group.get("lr", 3.0e-3)) or 3.0e-3)
            config["lr_1d_factor"] = float(group.get("lr_1d_factor", 1.0) or 1.0)
            config["betas_1d"] = group.get("betas_1d", (0.9, 0.95))
            config["mars_type"] = str(group.get("mars_type", "adamw") or "adamw")
            config["optimize_1d"] = bool(group.get("optimize_1d", False))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["cautious"] = bool(group.get("cautious", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "aida":
            config["k"] = int(group.get("k", 0) or 0)
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["rectify"] = bool(group.get("rectify", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adatam":
            config["decay_rate"] = float(group.get("decay_rate", 0.9))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adashift":
            config["keep_num"] = int(group.get("keep_num", 1) or 1)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adammini":
            config["model_sharding"] = bool(getattr(self.optimizer, "model_sharding", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "came":
            betas = group.get("betas", (0.9, 0.999, 0.9999))
            if not isinstance(betas, (tuple, list)):
                betas = (0.9, 0.999, 0.9999)
            config["betas"] = [
                float(betas[0] if len(betas) > 0 else 0.9),
                float(betas[1] if len(betas) > 1 else 0.999),
                float(betas[2] if len(betas) > 2 else 0.9999),
            ]
            config["clip_threshold"] = float(group.get("clip_threshold", 1.0))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adalite":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["g_norm_min"] = float(group.get("g_norm_min", 1.0e-10))
            config["ratio_min"] = float(group.get("ratio_min", 1.0e-4))
            config["tau"] = float(group.get("tau", 1.0))
            config["eps1"] = float(group.get("eps1", 1.0e-6))
            config["eps2"] = float(group.get("eps2", 1.0e-10))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "apollodqn":
            config["init_lr"] = float(group.get("init_lr", 1.0e-5))
            config["beta"] = float(group.get("beta", 0.9))
            config["rebound"] = str(group.get("rebound", "constant") or "constant")
            config["weight_decay_type"] = str(group.get("weight_decay_type", "l2") or "l2")
            config["warmup_steps"] = int(group.get("warmup_steps", 500) or 500)
            config["eps"] = float(group.get("eps", 1.0e-4))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind in {"emolynx", "emonavi", "emofact"}:
            default_betas = (
                (0.9, 0.99)
                if self._turbocore_native_update_quantized_optimizer_kind == "emolynx"
                else (0.9, 0.999)
            )
            betas = group.get("betas", default_betas)
            if not isinstance(betas, (tuple, list)):
                betas = default_betas
            config["betas"] = [
                float(betas[0] if len(betas) > 0 else default_betas[0]),
                float(betas[1] if len(betas) > 1 else default_betas[1]),
            ]
            config["use_shadow"] = bool(group.get("use_shadow", False))
            config["shadow_weight"] = float(group.get("shadow_weight", 0.05))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["eps"] = float(group.get("eps", 1.0e-8))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind in {"scion", "scionlight"}:
            config["momentum"] = float(group.get("momentum", 0.1))
            config["constraint"] = bool(group.get("constraint", False))
            config["norm_type"] = int(group.get("norm_type", 0) or 0)
            config["norm_kwargs"] = dict(group.get("norm_kwargs", {}) or {})
            config["scale"] = float(group.get("scale", 1.0))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["foreach"] = bool(group.get("foreach", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "pnm":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "padam":
            config["partial"] = float(group.get("partial", 0.25))
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "yogi":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "dualadam":
            config["switch_rate"] = float(group.get("switch_rate", 1e-2))
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "exadam":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "qhadam":
            nus = group.get("nus", (1.0, 1.0))
            config["nus"] = [float(nus[0]), float(nus[1])]
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "nadam":
            config["momentum_decay"] = float(group.get("momentum_decay", 0.004))
            config["decoupled_weight_decay"] = bool(group.get("decoupled_weight_decay", False))
            config["maximize"] = bool(group.get("maximize", getattr(self.optimizer, "maximize", False)))
        elif self._turbocore_native_update_quantized_optimizer_kind == "grokfastadamw":
            config["grokfast_alpha"] = float(group.get("grokfast_alpha", 0.98))
            config["grokfast_lamb"] = float(group.get("grokfast_lamb", 2.0))
            config["grokfast_after_step"] = int(group.get("grokfast_after_step", 0) or 0)
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ranger":
            config["alpha"] = float(group.get("alpha", 0.5))
            config["k"] = int(group.get("k", 6) or 6)
            config["n_sma_threshold"] = int(getattr(self.optimizer, "n_sma_threshold", 5) or 5)
            config["degenerated_to_sgd"] = bool(getattr(self.optimizer, "degenerated_to_sgd", False))
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ranger21":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["agc_eps"] = float(getattr(self.optimizer, "agc_eps", 1e-3))
            config["agc_clip"] = float(getattr(self.optimizer, "agc_clipping_value", 1e-2))
            config["norm_loss_factor"] = float(getattr(self.optimizer, "norm_loss_factor", 1e-4))
            config["use_softplus"] = bool(getattr(self.optimizer, "use_softplus", True))
            config["beta_softplus"] = float(getattr(self.optimizer, "beta_softplus", 50.0))
            config["lookahead_merge_time"] = int(getattr(self.optimizer, "lookahead_merge_time", 5) or 5)
            config["lookahead_blending_alpha"] = float(getattr(self.optimizer, "lookahead_blending_alpha", 0.5))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "ranger25":
            beta3 = betas[2] if len(betas) >= 3 else 0.9999
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["alpha"] = float(group.get("alpha", 5.0))
            config["cautious"] = bool(getattr(self.optimizer, "cautious", True))
            config["stable_adamw"] = bool(getattr(self.optimizer, "stable_adamw", True))
            config["orthograd"] = bool(getattr(self.optimizer, "orthograd", True))
            config["lookahead_merge_time"] = int(getattr(self.optimizer, "lookahead_merge_time", 5) or 5)
            config["lookahead_blending_alpha"] = float(getattr(self.optimizer, "lookahead_blending_alpha", 0.5))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "novograd":
            config["weight_decouple"] = bool(group.get("weight_decouple", False))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["grad_averaging"] = bool(group.get("grad_averaging", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "stableadamw":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["kahan_sum"] = bool(group.get("kahan_sum", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adamwsn":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["sn"] = bool(group.get("sn", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adams":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["ams_bound"] = bool(group.get("ams_bound", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "lamb":
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["rectify"] = bool(group.get("rectify", False))
            config["pre_norm"] = bool(getattr(self.optimizer, "pre_norm", False))
            config["adanorm"] = bool(group.get("adanorm", False))
            config["grad_averaging"] = bool(group.get("grad_averaging", True))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["adam"] = bool(group.get("adam", False))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "fadam":
            config["weight_decay"] = float(group.get("weight_decay", 0.1))
            config["clip"] = float(group.get("clip", 1.0))
            config["p"] = float(group.get("p", 0.5))
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "flashadamw":
            config["decouple_lr"] = bool(group.get("decouple_lr", False))
            config["quantize"] = bool(group.get("quantize", False))
            config["master_bytewidth"] = int(group.get("master_bytewidth", 0) or 0)
            config["initial_lr"] = float(group.get("initial_lr", group.get("lr", lr or 0.0)) or 0.0)
            config["maximize"] = bool(getattr(self.optimizer, "maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adamod":
            beta3 = betas[2] if len(betas) >= 3 else group.get("beta3", 0.9999)
            config["betas"] = [float(betas[0]), float(betas[1]), float(beta3)]
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["maximize"] = bool(group.get("maximize", False))
        elif self._turbocore_native_update_quantized_optimizer_kind == "adamp":
            config["betas"] = [float(betas[0]), float(betas[1])]
            config["weight_decouple"] = bool(group.get("weight_decouple", True))
            config["fixed_decay"] = bool(group.get("fixed_decay", False))
            config["delta"] = float(group.get("delta", 0.1))
            config["wd_ratio"] = float(group.get("wd_ratio", 0.1))
            config["nesterov"] = bool(group.get("nesterov", False))
            config["adam_debias"] = bool(group.get("adam_debias", False))
            config["maximize"] = bool(group.get("maximize", getattr(self.optimizer, "maximize", False)))
        elif self._turbocore_native_update_quantized_optimizer_kind in {"lion8bit", "paged_lion8bit"}:
            config["betas"] = [float(betas[0]), float(betas[1] if len(betas) >= 2 else 0.99)]
        elif self._turbocore_native_update_quantized_optimizer_kind == "sgd_nesterov8bit":
            config["momentum"] = float(group.get("momentum", 0.9))
        return config

    @staticmethod
    def _normalize_turbocore_simple_optimizer_kind(value: Any) -> str:
        kind = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if kind in {"sgd", "plain_sgd", "torch_sgd"}:
            return "sgd"
        if kind in {"sgdnesterov", "sgd_nesterov"}:
            return "sgd_nesterov"
        if kind in {"signmomentum", "sign_momentum", "signsgd", "tiger"}:
            return "sign_momentum"
        if kind == "qhm":
            return "qhm"
        if kind in {"accsgd", "acc_sgd"}:
            return "accsgd"
        if kind == "fromage":
            return "fromage"
        if kind == "rmsprop":
            return "rmsprop"
        if kind == "lars":
            return "lars"
        if kind == "pid":
            return "pid"
        if kind == "sgdp":
            return "sgdp"
        if kind == "gravity":
            return "gravity"
        if kind == "aggmo":
            return "aggmo"
        if kind == "asgd":
            return "asgd"
        if kind == "madgrad":
            return "madgrad"
        if kind == "nero":
            return "nero"
        if kind == "vsgd":
            return "vsgd"
        if kind == "lion":
            return "lion"
        return ""

    @staticmethod
    def _normalize_turbocore_quantized_optimizer_kind(value: Any) -> str:
        kind = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if kind in {"adamw8bit", "adamw_8bit"}:
            return "adamw8bit"
        if kind in {"kahanadamw8bit", "kahan_adamw8bit"}:
            return "kahan_adamw8bit"
        if kind in {"pagedadamw8bit", "paged_adamw8bit"}:
            return "paged_adamw8bit"
        if kind in {"pagedadamw", "paged_adamw"}:
            return "paged_adamw"
        if kind in {"pagedadamw32bit", "paged_adamw32bit"}:
            return "paged_adamw32bit"
        if kind in {"lion8bit", "lion_8bit"}:
            return "lion8bit"
        if kind in {"pagedlion8bit", "paged_lion8bit", "paged_lion_8bit"}:
            return "paged_lion8bit"
        if kind in {"sgdnesterov8bit", "sgd_nesterov8bit", "sgd_nesterov_8bit", "sgd8bit"}:
            return "sgd_nesterov8bit"
        if kind in {"automagicpp", "automagic_plus_plus", "automagic++"}:
            return "automagicpp"
        if kind in {"animafactoredadamw", "anima_factored_adamw", "anima_factored"}:
            return "anima_factored_adamw"
        if kind in {"adafactor", "ada_factor"}:
            return "adafactor"
        if kind in {"adamwschedulefree", "adamw_schedule_free", "schedulefreeadamw", "schedulefree_adamw"}:
            return "adamw_schedule_free"
        if kind in {"autoprodigy", "auto_prodigy", "prodigy", "prodigyplusschedulefree", "prodigy_plus_schedule_free"}:
            return "prodigy"
        if kind in {"muon", "builtin_muon"}:
            return "muon"
        if kind in {"distributedmuon", "distributed_muon"}:
            return "distributedmuon"
        if kind == "adamuon":
            return "adamuon"
        if kind == "adago":
            return "adago"
        if kind in {"sgdsai", "sgd_sai"}:
            return "sgdsai"
        if kind == "sm3":
            return "sm3"
        if kind == "spam":
            return "spam"
        if kind in {"stablespam", "stable_spam"}:
            return "stablespam"
        if kind == "adadelta":
            return "adadelta"
        if kind == "ftrl":
            return "ftrl"
        if kind == "diffgrad":
            return "diffgrad"
        if kind == "adabelief":
            return "adabelief"
        if kind == "adabound":
            return "adabound"
        if kind == "laprop":
            return "laprop"
        if kind == "adai":
            return "adai"
        if kind == "adopt":
            return "adopt"
        if kind == "msvag":
            return "msvag"
        if kind == "ademamix":
            return "ademamix"
        if kind == "simplifiedademamix":
            return "simplifiedademamix"
        if kind == "a2grad":
            return "a2grad"
        if kind == "avagrad":
            return "avagrad"
        if kind == "adanorm":
            return "adanorm"
        if kind == "bcos":
            return "bcos"
        if kind == "adagc":
            return "adagc"
        if kind == "adasmooth":
            return "adasmooth"
        if kind == "adapnm":
            return "adapnm"
        if kind == "adan":
            return "adan"
        if kind == "ano":
            return "ano"
        if kind == "amos":
            return "amos"
        if kind == "apollo":
            return "apollo"
        if kind == "galore":
            return "galore"
        if kind == "fira":
            return "fira"
        if kind == "focus":
            return "focus"
        if kind == "conda":
            return "conda"
        if kind == "grams":
            return "grams"
        if kind == "srmm":
            return "srmm"
        if kind == "splus":
            return "splus"
        if kind == "tam":
            return "tam"
        if kind == "swats":
            return "swats"
        if kind in {"sophiah", "sophia_h"}:
            return "sophiah"
        if kind == "racs":
            return "racs"
        if kind == "kate":
            return "kate"
        if kind == "rose":
            return "rose"
        if kind == "mars":
            return "mars"
        if kind == "aida":
            return "aida"
        if kind == "adatam":
            return "adatam"
        if kind == "adashift":
            return "adashift"
        if kind == "adammini":
            return "adammini"
        if kind == "came":
            return "came"
        if kind == "adalite":
            return "adalite"
        if kind == "apollodqn":
            return "apollodqn"
        if kind == "emonavi":
            return "emonavi"
        if kind == "emofact":
            return "emofact"
        if kind == "emolynx":
            return "emolynx"
        if kind == "scion":
            return "scion"
        if kind == "scionlight":
            return "scionlight"
        if kind in {"spectralsphere", "spectral_sphere"}:
            return "spectralsphere"
        if kind == "demo":
            return "demo"
        if kind == "alig":
            return "alig"
        if kind == "alice":
            return "alice"
        if kind == "adahessian":
            return "adahessian"
        if kind == "kron":
            return "kron"
        if kind == "lorarite":
            return "lorarite"
        if kind in {"scalableshampoo", "scalable_shampoo"}:
            return "scalableshampoo"
        if kind == "shampoo":
            return "shampoo"
        if kind == "soap":
            return "soap"
        if kind == "pnm":
            return "pnm"
        if kind in {
            "dadapt",
            "dadaptation",
            "dadaptadampreprint",
            "dadapt_adam_preprint",
            "dadaptadagrad",
            "dadapt_adagrad",
            "dadaptadam",
            "dadapt_adam",
            "dadaptadan",
            "dadapt_adan",
            "dadaptadanip",
            "dadapt_adan_ip",
            "dadaptlion",
            "dadapt_lion",
            "dadaptsgd",
            "dadapt_sgd",
        }:
            return "dadapt"
        if kind in {"schedulefreesgd", "schedulefree_sgd"}:
            return "schedulefree_sgd"
        if kind in {"schedulefreeradam", "schedulefree_radam"}:
            return "schedulefree_radam"
        if kind == "radam":
            return "radam"
        if kind == "padam":
            return "padam"
        if kind == "yogi":
            return "yogi"
        if kind == "dualadam":
            return "dualadam"
        if kind == "exadam":
            return "exadam"
        if kind == "qhadam":
            return "qhadam"
        if kind == "nadam":
            return "nadam"
        if kind in {"grokfastadamw", "grokfast_adamw"}:
            return "grokfastadamw"
        if kind == "ranger":
            return "ranger"
        if kind == "ranger21":
            return "ranger21"
        if kind == "ranger25":
            return "ranger25"
        if kind == "novograd":
            return "novograd"
        if kind in {"stableadamw", "stable_adamw"}:
            return "stableadamw"
        if kind in {"adamwsn", "adamw_sn"}:
            return "adamwsn"
        if kind in {"adams", "adam_s"}:
            return "adams"
        if kind == "lamb":
            return "lamb"
        if kind == "fadam":
            return "fadam"
        if kind in {"flashadamw", "flash_adamw"}:
            return "flashadamw"
        if kind == "adam":
            return "adam"
        if kind == "adamax":
            return "adamax"
        if kind == "adamc":
            return "adamc"
        if kind == "adamg":
            return "adamg"
        if kind == "adamod":
            return "adamod"
        if kind == "adamp":
            return "adamp"
        return ""

    def _turbocore_simple_optimizer_training_executor_config(self) -> Dict[str, Any]:
        group = self.optimizer.param_groups[0] if self.optimizer and self.optimizer.param_groups else {}
        lr = group.get("lr")
        if lr is None:
            lr = getattr(self, "learning_rate", 0.0)
        kind = self._turbocore_native_update_simple_optimizer_kind
        config: Dict[str, Any] = {
            "optimizer_kind": kind,
            "lr": float(lr or 0.0),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "block_size": int(group.get("block_size", 128) or 128),
            "require_native_cuda": bool(self._turbocore_native_update_require_native_cuda),
        }
        if kind == "lion":
            betas = group.get("betas", (0.9, 0.99))
            config["betas"] = [float(betas[0]), float(betas[1])]
        elif kind in {"sgd", "sgd_nesterov"}:
            config["momentum"] = float(group.get("momentum", 0.9))
        elif kind == "sign_momentum":
            config["momentum"] = float(group.get("momentum", group.get("beta", 0.9)))
        elif kind == "qhm":
            config["momentum"] = float(group.get("momentum", 0.0))
            config["nu"] = float(group.get("nu", 1.0))
        elif kind == "accsgd":
            config["kappa"] = float(group.get("kappa", 1000.0))
            config["xi"] = float(group.get("xi", 10.0))
            config["constant"] = float(group.get("constant", 0.7))
        elif kind == "rmsprop":
            config["alpha"] = float(group.get("alpha", 0.99))
            config["eps"] = float(group.get("eps", 1e-8))
            config["momentum"] = float(group.get("momentum", 0.9))
            config["centered"] = bool(group.get("centered", False))
        elif kind == "lars":
            config["momentum"] = float(group.get("momentum", 0.9))
            config["dampening"] = float(group.get("dampening", 0.0))
            config["trust_coefficient"] = float(group.get("trust_coefficient", 1e-3))
        elif kind == "sgdp":
            config["momentum"] = float(group.get("momentum", 0.0))
            config["dampening"] = float(group.get("dampening", 0.0))
        elif kind == "gravity":
            config["beta"] = float(group.get("beta", 0.9))
        elif kind == "aggmo":
            betas = group.get("betas", (0.0, 0.9, 0.99))
            config["betas"] = [float(betas[0]), float(betas[1] if len(betas) >= 2 else 0.9)]
            config["beta"] = float(betas[2] if len(betas) >= 3 else 0.99)
        elif kind == "asgd":
            config["beta"] = float(group.get("theta", 1.0))
            config["dampening"] = float(group.get("dampening", 1.0))
            config["eps"] = float(group.get("eps", 1e-5))
        elif kind == "madgrad":
            config["momentum"] = float(group.get("momentum", 0.9))
            config["eps"] = float(group.get("eps", 1e-6))
        elif kind == "nero":
            config["beta"] = float(group.get("beta", 0.999))
            config["eps"] = float(group.get("eps", 1e-8))
        elif kind == "vsgd":
            config["alpha"] = float(group.get("tau1", 0.81))
            config["beta"] = float(group.get("tau2", 0.9))
            config["eps"] = float(group.get("eps", 1e-8))
        return config

    def _turbocore_native_update_stream_lifetime_lease_request(self) -> Dict[str, Any]:
        explicit_training = bool(
            self._turbocore_native_update_gate.config.dispatch_enabled
            and self._turbocore_native_update_training_path_enabled
            and self._turbocore_native_update_require_native_cuda
        )
        recovery_ready = bool(
            explicit_training
            and self._turbocore_native_update_runtime_synchronization_policy == "borrowed_stream_event_chain"
        )
        return build_single_step_lifetime_lease_request(
            explicit_training_context=explicit_training,
            recovery_ready=recovery_ready,
            lease_scope="native_update_training_step",
        )

    def _turbocore_native_update_direct_grad_executor_enabled(self) -> bool:
        return bool(
            self._turbocore_native_update_gate.config.dispatch_enabled
            and self._turbocore_native_update_training_path_enabled
            and self._turbocore_native_update_require_native_cuda
            and not self._turbocore_native_update_quantized_optimizer_kind
            and self._turbocore_update_shadow.enabled
            and self._turbocore_update_shadow.config.direct_grad
        )

    def _prepare_turbocore_native_update_direct_grad_executor_before_backward(
        self,
        trainable_params: List[torch.nn.Parameter],
    ) -> Dict[str, Any]:
        if not self._turbocore_native_update_direct_grad_executor_enabled():
            return {}
        started = time.perf_counter()
        try:
            executor = self._get_turbocore_native_update_training_executor(trainable_params)
            update_executor = getattr(executor, "executor", None)
            binding = getattr(update_executor, "direct_grad_binding", None)
            if binding is None:
                return {
                    "schema_version": 1,
                    "stage": "before_backward",
                    "enabled": False,
                    "reason": "native_update_direct_grad_binding_missing",
                    "training_path_enabled": bool(self._turbocore_native_update_training_path_enabled),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
                }
            binding.set_active(True)
            reset_owner_grad = bool(getattr(self, "_current_accumulation_group_start", True))
            if reset_owner_grad:
                binding.zero_owner_grad()
            return {
                "schema_version": 1,
                "stage": "before_backward",
                "enabled": True,
                "training_path_enabled": bool(self._turbocore_native_update_training_path_enabled),
                "native_dispatch_enabled": bool(self._turbocore_native_update_gate.config.dispatch_enabled),
                "direct_grad_to_training_executor_owner": True,
                "reset_owner_grad": reset_owner_grad,
                "snapshot": binding.snapshot(),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            }
        except Exception as exc:
            logger.debug("TurboCore native direct-grad executor prepare skipped: %s", exc)
            return {
                "schema_version": 1,
                "stage": "before_backward",
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
                "training_path_enabled": bool(self._turbocore_native_update_training_path_enabled),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 4),
            }

    def _get_turbocore_native_update_training_executor(self, trainable_params: List[torch.nn.Parameter]) -> Any:
        if self._turbocore_native_update_training_executor is None:
            if self._turbocore_native_update_quantized_optimizer_kind == "kahan_adamw8bit":
                self._turbocore_native_update_training_executor = build_kahan_adamw8bit_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {"adamw8bit", "paged_adamw8bit"}:
                from core.turbocore_paged_adamw8bit_training_executor import build_paged_adamw8bit_training_executor

                self._turbocore_native_update_training_executor = build_paged_adamw8bit_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {"paged_adamw", "paged_adamw32bit"}:
                from core.turbocore_paged_adamw32_training_executor import build_paged_adamw32_training_executor

                self._turbocore_native_update_training_executor = build_paged_adamw32_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {
                "lion8bit",
                "paged_lion8bit",
                "sgd_nesterov8bit",
            }:
                from core.turbocore_simple_quantized_optimizer_training_executor import (
                    build_simple_quantized_optimizer_training_executor,
                )

                self._turbocore_native_update_training_executor = build_simple_quantized_optimizer_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "automagicpp":
                from core.turbocore_automagicpp_training_executor import build_automagicpp_training_executor

                self._turbocore_native_update_training_executor = build_automagicpp_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "anima_factored_adamw":
                from core.turbocore_anima_factored_adamw_training_executor import (
                    build_anima_factored_adamw_training_executor,
                )

                self._turbocore_native_update_training_executor = build_anima_factored_adamw_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adafactor":
                from core.turbocore_adafactor_training_executor import build_adafactor_training_executor

                self._turbocore_native_update_training_executor = build_adafactor_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamw_schedule_free":
                from core.turbocore_adamw_schedule_free_training_executor import (
                    build_adamw_schedule_free_training_executor,
                )

                self._turbocore_native_update_training_executor = build_adamw_schedule_free_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {"prodigy", "dadapt"}:
                from core.turbocore_adaptive_lr_training_executor import build_adaptive_lr_training_executor

                self._turbocore_native_update_training_executor = build_adaptive_lr_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {"muon", "distributedmuon"}:
                from core.turbocore_muon_training_executor import build_muon_training_executor

                self._turbocore_native_update_training_executor = build_muon_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind in {"adamuon", "adago"}:
                from core.turbocore_plugin_muon_family_adamw_training_executor import (
                    build_plugin_muon_family_adamw_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_muon_family_adamw_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "sgdsai":
                from core.turbocore_sgdsai_training_executor import build_sgdsai_training_executor

                self._turbocore_native_update_training_executor = build_sgdsai_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "sm3":
                from core.turbocore_plugin_sm3_training_executor import build_plugin_sm3_training_executor

                self._turbocore_native_update_training_executor = build_plugin_sm3_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "spam":
                from core.turbocore_plugin_spam_training_executor import build_plugin_spam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_spam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "stablespam":
                from core.turbocore_plugin_stablespam_training_executor import (
                    build_plugin_stablespam_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_stablespam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adadelta":
                from core.turbocore_plugin_adadelta_training_executor import (
                    build_plugin_adadelta_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_adadelta_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ftrl":
                from core.turbocore_plugin_ftrl_training_executor import build_plugin_ftrl_training_executor

                self._turbocore_native_update_training_executor = build_plugin_ftrl_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "diffgrad":
                from core.turbocore_plugin_diffgrad_training_executor import build_plugin_diffgrad_training_executor

                self._turbocore_native_update_training_executor = build_plugin_diffgrad_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adabelief":
                from core.turbocore_plugin_adabelief_training_executor import (
                    build_plugin_adabelief_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_adabelief_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adabound":
                from core.turbocore_plugin_adabound_training_executor import build_plugin_adabound_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adabound_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "laprop":
                from core.turbocore_plugin_laprop_training_executor import build_plugin_laprop_training_executor

                self._turbocore_native_update_training_executor = build_plugin_laprop_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adai":
                from core.turbocore_plugin_adai_training_executor import build_plugin_adai_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adai_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adopt":
                from core.turbocore_plugin_adopt_training_executor import build_plugin_adopt_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adopt_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "msvag":
                from core.turbocore_plugin_msvag_training_executor import build_plugin_msvag_training_executor

                self._turbocore_native_update_training_executor = build_plugin_msvag_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ademamix":
                from core.turbocore_plugin_ademamix_training_executor import (
                    build_plugin_ademamix_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_ademamix_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "simplifiedademamix":
                from core.turbocore_plugin_simplifiedademamix_training_executor import (
                    build_plugin_simplifiedademamix_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_simplifiedademamix_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "a2grad":
                from core.turbocore_plugin_a2grad_training_executor import build_plugin_a2grad_training_executor

                self._turbocore_native_update_training_executor = build_plugin_a2grad_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "avagrad":
                from core.turbocore_plugin_avagrad_training_executor import build_plugin_avagrad_training_executor

                self._turbocore_native_update_training_executor = build_plugin_avagrad_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adanorm":
                from core.turbocore_plugin_adanorm_training_executor import build_plugin_adanorm_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adanorm_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "bcos":
                from core.turbocore_plugin_bcos_training_executor import build_plugin_bcos_training_executor

                self._turbocore_native_update_training_executor = build_plugin_bcos_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adagc":
                from core.turbocore_plugin_adagc_training_executor import build_plugin_adagc_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adagc_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adasmooth":
                from core.turbocore_plugin_adasmooth_training_executor import build_plugin_adasmooth_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adasmooth_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adapnm":
                from core.turbocore_plugin_adapnm_training_executor import build_plugin_adapnm_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adapnm_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adan":
                from core.turbocore_plugin_adan_training_executor import build_plugin_adan_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adan_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ano":
                from core.turbocore_plugin_ano_training_executor import build_plugin_ano_training_executor

                self._turbocore_native_update_training_executor = build_plugin_ano_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "amos":
                from core.turbocore_plugin_amos_training_executor import build_plugin_amos_training_executor

                self._turbocore_native_update_training_executor = build_plugin_amos_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "apollo":
                from core.turbocore_plugin_apollo_training_executor import build_plugin_apollo_training_executor

                self._turbocore_native_update_training_executor = build_plugin_apollo_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "fira":
                from core.turbocore_plugin_fira_training_executor import build_plugin_fira_training_executor

                self._turbocore_native_update_training_executor = build_plugin_fira_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "galore":
                from core.turbocore_plugin_galore_training_executor import build_plugin_galore_training_executor

                self._turbocore_native_update_training_executor = build_plugin_galore_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "focus":
                from core.turbocore_plugin_focus_training_executor import build_plugin_focus_training_executor

                self._turbocore_native_update_training_executor = build_plugin_focus_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "conda":
                from core.turbocore_plugin_conda_training_executor import build_plugin_conda_training_executor

                self._turbocore_native_update_training_executor = build_plugin_conda_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "grams":
                from core.turbocore_plugin_grams_training_executor import build_plugin_grams_training_executor

                self._turbocore_native_update_training_executor = build_plugin_grams_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "srmm":
                from core.turbocore_plugin_srmm_training_executor import build_plugin_srmm_training_executor

                self._turbocore_native_update_training_executor = build_plugin_srmm_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "splus":
                from core.turbocore_plugin_splus_training_executor import build_plugin_splus_training_executor

                self._turbocore_native_update_training_executor = build_plugin_splus_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "tam":
                from core.turbocore_plugin_tam_training_executor import build_plugin_tam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_tam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "swats":
                from core.turbocore_plugin_swats_training_executor import build_plugin_swats_training_executor

                self._turbocore_native_update_training_executor = build_plugin_swats_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "sophiah":
                from core.turbocore_plugin_sophiah_training_executor import build_plugin_sophiah_training_executor

                self._turbocore_native_update_training_executor = build_plugin_sophiah_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "racs":
                from core.turbocore_plugin_racs_training_executor import build_plugin_racs_training_executor

                self._turbocore_native_update_training_executor = build_plugin_racs_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "kate":
                from core.turbocore_plugin_kate_training_executor import build_plugin_kate_training_executor

                self._turbocore_native_update_training_executor = build_plugin_kate_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "rose":
                from core.turbocore_plugin_rose_training_executor import build_plugin_rose_training_executor

                self._turbocore_native_update_training_executor = build_plugin_rose_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "mars":
                from core.turbocore_plugin_mars_training_executor import build_plugin_mars_training_executor

                self._turbocore_native_update_training_executor = build_plugin_mars_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "aida":
                from core.turbocore_plugin_aida_training_executor import build_plugin_aida_training_executor

                self._turbocore_native_update_training_executor = build_plugin_aida_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adatam":
                from core.turbocore_plugin_adatam_training_executor import build_plugin_adatam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adatam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adashift":
                from core.turbocore_plugin_adashift_training_executor import build_plugin_adashift_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adashift_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adammini":
                from core.turbocore_plugin_adammini_training_executor import build_plugin_adammini_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adammini_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "came":
                from core.turbocore_plugin_came_training_executor import build_plugin_came_training_executor

                self._turbocore_native_update_training_executor = build_plugin_came_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adalite":
                from core.turbocore_plugin_adalite_training_executor import build_plugin_adalite_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adalite_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "apollodqn":
                from core.turbocore_plugin_apollodqn_training_executor import build_plugin_apollodqn_training_executor

                self._turbocore_native_update_training_executor = build_plugin_apollodqn_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "emolynx":
                from core.turbocore_plugin_emolynx_training_executor import build_plugin_emolynx_training_executor

                self._turbocore_native_update_training_executor = build_plugin_emolynx_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "emonavi":
                from core.turbocore_plugin_emonavi_training_executor import build_plugin_emonavi_training_executor

                self._turbocore_native_update_training_executor = build_plugin_emonavi_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "emofact":
                from core.turbocore_plugin_emofact_training_executor import build_plugin_emofact_training_executor

                self._turbocore_native_update_training_executor = build_plugin_emofact_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "scion":
                from core.turbocore_plugin_scion_training_executor import build_plugin_scion_training_executor

                self._turbocore_native_update_training_executor = build_plugin_scion_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "scionlight":
                from core.turbocore_plugin_scionlight_training_executor import build_plugin_scionlight_training_executor

                self._turbocore_native_update_training_executor = build_plugin_scionlight_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "spectralsphere":
                from core.turbocore_plugin_spectralsphere_training_executor import (
                    build_plugin_spectralsphere_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_spectralsphere_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "demo":
                from core.turbocore_plugin_demo_training_executor import build_plugin_demo_training_executor

                self._turbocore_native_update_training_executor = build_plugin_demo_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "alig":
                from core.turbocore_plugin_alig_training_executor import build_plugin_alig_training_executor

                self._turbocore_native_update_training_executor = build_plugin_alig_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "alice":
                from core.turbocore_plugin_alice_training_executor import build_plugin_alice_training_executor

                self._turbocore_native_update_training_executor = build_plugin_alice_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adahessian":
                from core.turbocore_plugin_adahessian_training_executor import (
                    build_plugin_adahessian_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_adahessian_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "kron":
                from core.turbocore_plugin_kron_training_executor import build_plugin_kron_training_executor

                self._turbocore_native_update_training_executor = build_plugin_kron_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "lorarite":
                from core.turbocore_plugin_lorarite_training_executor import build_plugin_lorarite_training_executor

                self._turbocore_native_update_training_executor = build_plugin_lorarite_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "shampoo":
                from core.turbocore_plugin_shampoo_family_training_executor import (
                    build_plugin_shampoo_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_shampoo_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "scalableshampoo":
                from core.turbocore_plugin_shampoo_family_training_executor import (
                    build_plugin_scalableshampoo_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_scalableshampoo_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "soap":
                from core.turbocore_plugin_shampoo_family_training_executor import build_plugin_soap_training_executor

                self._turbocore_native_update_training_executor = build_plugin_soap_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "pnm":
                from core.turbocore_pnm_training_executor import build_pnm_training_executor

                self._turbocore_native_update_training_executor = build_pnm_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "schedulefree_sgd":
                from core.turbocore_plugin_schedulefree_sgd_training_executor import (
                    build_plugin_schedulefree_sgd_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_schedulefree_sgd_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "schedulefree_radam":
                from core.turbocore_plugin_schedulefree_radam_training_executor import (
                    build_plugin_schedulefree_radam_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_schedulefree_radam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "radam":
                from core.turbocore_plugin_radam_training_executor import build_plugin_radam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_radam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "padam":
                from core.turbocore_plugin_padam_training_executor import build_plugin_padam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_padam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "yogi":
                from core.turbocore_plugin_yogi_training_executor import build_plugin_yogi_training_executor

                self._turbocore_native_update_training_executor = build_plugin_yogi_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "dualadam":
                from core.turbocore_plugin_dualadam_training_executor import build_plugin_dualadam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_dualadam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "exadam":
                from core.turbocore_plugin_exadam_training_executor import build_plugin_exadam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_exadam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "qhadam":
                from core.turbocore_plugin_qhadam_training_executor import build_plugin_qhadam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_qhadam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "nadam":
                from core.turbocore_plugin_nadam_training_executor import build_plugin_nadam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_nadam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "grokfastadamw":
                from core.turbocore_plugin_grokfastadamw_training_executor import (
                    build_plugin_grokfastadamw_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_grokfastadamw_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ranger":
                from core.turbocore_plugin_ranger_training_executor import build_plugin_ranger_training_executor

                self._turbocore_native_update_training_executor = build_plugin_ranger_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ranger21":
                from core.turbocore_plugin_ranger21_training_executor import build_plugin_ranger21_training_executor

                self._turbocore_native_update_training_executor = build_plugin_ranger21_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "ranger25":
                from core.turbocore_plugin_ranger25_training_executor import build_plugin_ranger25_training_executor

                self._turbocore_native_update_training_executor = build_plugin_ranger25_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "novograd":
                from core.turbocore_plugin_novograd_training_executor import build_plugin_novograd_training_executor

                self._turbocore_native_update_training_executor = build_plugin_novograd_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "stableadamw":
                from core.turbocore_plugin_stableadamw_training_executor import (
                    build_plugin_stableadamw_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_stableadamw_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamwsn":
                from core.turbocore_plugin_adamwsn_training_executor import build_plugin_adamwsn_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamwsn_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adams":
                from core.turbocore_plugin_adams_training_executor import build_plugin_adams_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adams_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "lamb":
                from core.turbocore_plugin_lamb_training_executor import build_plugin_lamb_training_executor

                self._turbocore_native_update_training_executor = build_plugin_lamb_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "fadam":
                from core.turbocore_plugin_fadam_training_executor import build_plugin_fadam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_fadam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "flashadamw":
                from core.turbocore_plugin_flashadamw_training_executor import (
                    build_plugin_flashadamw_training_executor,
                )

                self._turbocore_native_update_training_executor = build_plugin_flashadamw_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adam":
                from core.turbocore_plugin_adam_training_executor import build_plugin_adam_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adam_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamax":
                from core.turbocore_plugin_adamax_training_executor import build_plugin_adamax_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamax_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamc":
                from core.turbocore_plugin_adamc_training_executor import build_plugin_adamc_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamc_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamg":
                from core.turbocore_plugin_adamg_training_executor import build_plugin_adamg_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamg_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamod":
                from core.turbocore_plugin_adamod_training_executor import build_plugin_adamod_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamod_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_quantized_optimizer_kind == "adamp":
                from core.turbocore_plugin_adamp_training_executor import build_plugin_adamp_training_executor

                self._turbocore_native_update_training_executor = build_plugin_adamp_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
            elif self._turbocore_native_update_simple_optimizer_kind:
                self._turbocore_native_update_training_executor = build_simple_optimizer_training_executor(
                    params=trainable_params,
                    config=self._turbocore_simple_optimizer_training_executor_config(),
                )
            else:
                self._turbocore_native_update_training_executor = build_native_update_training_executor(
                    optimizer=self.optimizer,
                    params=trainable_params,
                    config=self._turbocore_native_update_training_executor_config(),
                )
        return self._turbocore_native_update_training_executor

    def _sync_turbocore_native_update_training_executor_to_pytorch(self, reason: str) -> Dict[str, Any]:
        executor = getattr(self, "_turbocore_native_update_training_executor", None)
        sync = getattr(executor, "sync_optimizer_state_to_pytorch", None)
        if not callable(sync):
            return {}
        try:
            return dict(sync(reason=reason) or {})
        except Exception as exc:
            logger.debug("TurboCore native update optimizer-state sync skipped: %s", exc)
            disable = getattr(self._turbocore_native_update_dispatch_runtime, "disable_for_run", None)
            disabled_state = dict(disable("native_update_optimizer_state_sync_error") or {}) if callable(disable) else {}
            return {
                "schema_version": 1,
                "synced": False,
                "error": f"{type(exc).__name__}: {exc}",
                "reason": str(reason or "sync_error"),
                "disabled_for_run": bool(disabled_state.get("disabled_for_run", False)),
                "disable_reason": str(disabled_state.get("disable_reason", "") or ""),
            }

    def _can_retain_turbocore_native_update_gate(
        self,
        previous_gate: Dict[str, Any],
        shadow_report: Dict[str, Any],
        dispatch_runtime_report: Dict[str, Any],
    ) -> bool:
        return can_retain_native_update_probe_evidence(
            previous_gate=previous_gate,
            shadow_report=shadow_report,
            dispatch_runtime_report=dispatch_runtime_report,
            defer_state_sync=bool(self._turbocore_native_update_defer_state_sync),
        )

    def _close_turbocore_native_update_training_executor(self) -> None:
        executor = getattr(self, "_turbocore_native_update_training_executor", None)
        self._turbocore_native_update_training_executor = None
        if executor is None:
            return
        close = getattr(executor, "close", None)
        if callable(close):
            close()

    def get_turbocore_update_checkpoint_state(self) -> Dict[str, Any]:
        if not getattr(self, "_turbocore_update_shadow", None) or not self._turbocore_update_shadow.enabled:
            return {
                "schema_version": 1,
                "state": "turbocore_update_shadow_checkpoint_v0",
                "enabled": False,
                "training_path_enabled": False,
            }
        return self._turbocore_update_shadow.checkpoint_state(
            include_owner_state=bool(getattr(self, "_turbocore_update_shadow_save_owner_state", False))
        )

    def load_turbocore_update_checkpoint_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not getattr(self, "_turbocore_update_shadow", None):
            return {
                "schema_version": 1,
                "state": "turbocore_update_shadow_checkpoint_v0",
                "loaded": False,
                "reason": "shadow_unavailable",
                "training_path_enabled": False,
            }
        return self._turbocore_update_shadow.load_checkpoint_state(state, self._get_trainable_params())


__all__ = ["TrainingLoopTurbocoreRuntimeMixin"]

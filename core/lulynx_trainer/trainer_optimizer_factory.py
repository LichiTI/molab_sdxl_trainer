# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Optimizer + LR-scheduler construction for :class:`LulynxTrainer`.

Extracted verbatim from ``trainer.py`` as a mixin to keep the trainer file
navigable. Behaviour is unchanged: these run as bound methods of the trainer
instance (same ``self`` semantics, same call sites). MRO resolves sibling
methods that stay on ``LulynxTrainer`` (e.g. ``_build_anima_grouped_param_groups``,
``_attach_optimizer_profiles_to_training_loop``) through ``self``.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

import torch

from .config import OptimizerType, SchedulerType


class TrainerOptimizerFactoryMixin:
    """Optimizer-backend factories, param-group builders, mn-LoRA config
    helpers, the LR-finder step fn, and LR-scheduler construction."""

    def _parse_custom_args(self, raw: Any) -> Dict[str, Any]:
        """Parse UI custom arg strings without eval or code execution."""
        if not raw:
            return {}
        if isinstance(raw, dict):
            return dict(raw)
        text = str(raw).strip()
        if not text:
            return {}

        if text.startswith("{"):
            try:
                import json
                parsed = json.loads(text)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except Exception:
                pass

        def split_chunks(value: str) -> List[str]:
            chunks: List[str] = []
            current: List[str] = []
            depth = 0
            quote = ""
            index = 0
            while index < len(value):
                char = value[index]
                if quote:
                    current.append(char)
                    if char == quote:
                        quote = ""
                    index += 1
                    continue
                if char in "'\"":
                    quote = char
                    current.append(char)
                    index += 1
                    continue
                if char in "([{":
                    depth += 1
                elif char in ")]}" and depth > 0:
                    depth -= 1
                is_space_separator = False
                if char.isspace() and depth == 0:
                    rest = value[index + 1 :]
                    is_space_separator = bool(re.match(r"\s*[A-Za-z_][\w.-]*\s*=", rest))
                if (char in "\n,;" or is_space_separator) and depth == 0:
                    chunk = "".join(current).strip()
                    if chunk:
                        chunks.append(chunk)
                    current = []
                else:
                    current.append(char)
                index += 1
            chunk = "".join(current).strip()
            if chunk:
                chunks.append(chunk)
            return chunks

        result: Dict[str, Any] = {}
        for chunk in split_chunks(text):
            if not chunk.strip() or "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            try:
                result[key] = ast.literal_eval(value)
            except Exception:
                lowered = value.lower()
                if lowered in {"true", "yes", "on"}:
                    result[key] = True
                elif lowered in {"false", "no", "off"}:
                    result[key] = False
                else:
                    result[key] = value
        return result

    def _filtered_custom_args(self, raw: Any, allowed: set[str], label: str) -> Dict[str, Any]:
        parsed = self._parse_custom_args(raw)
        filtered: Dict[str, Any] = {}
        for key, value in parsed.items():
            if key in allowed:
                filtered[key] = value
            else:
                self._log(f"Ignoring unsupported {label} key: {key}")
        return filtered

    def _optimizer_allowed_args(self) -> set[str]:
        opt = self.config.optimizer
        if opt in {
            OptimizerType.ADAMW,
            OptimizerType.ADAMW_8BIT,
            OptimizerType.PAGED_ADAMW,
            OptimizerType.PAGED_ADAMW_32BIT,
            OptimizerType.PAGED_ADAMW_8BIT,
            OptimizerType.KAHAN_ADAMW_8BIT,
        }:
            return {"betas", "eps", "amsgrad", "foreach", "maximize", "capturable", "fused"}
        if opt == OptimizerType.ANIMA_FACTORED_ADAMW:
            return {"betas", "eps", "min_dim", "min_numel", "factored_eps"}
        if opt == OptimizerType.PRODIGY:
            return {
                "betas", "beta3", "eps", "decouple", "use_bias_correction",
                "safeguard_warmup", "growth_rate", "d0", "d_coef",
            }
        if opt == OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE:
            return {
                "betas", "beta3", "eps", "d0", "d_coef", "weight_decay_by_lr",
                "d_limiter", "prodigy_steps", "schedulefree_c", "split_groups",
                "split_groups_mean", "factored", "factored_fp32",
                "use_bias_correction", "use_stableadamw", "use_schedulefree",
                "use_speed", "stochastic_rounding", "fused_back_pass",
                "use_cautious", "use_grams", "use_adopt", "use_orthograd",
                "use_focus",
            }
        if opt == OptimizerType.AUTOMAGIC_PLUS_PLUS:
            return {
                "min_lr", "max_lr", "lr_bump", "lr_up", "lr_down", "lr_adapt_mode",
                "eps", "clip_threshold", "beta2", "beta1", "weight_decay_mode",
                "max_update_rms_ratio", "sign_eps", "lr_granularity",
                "agreement_threshold", "do_parameter_swapping",
                "parameter_swapping_factor", "swap_interval",
            }
        if opt == OptimizerType.AUTO_PRODIGY:
            return {
                "betas", "beta3", "eps", "d0", "d_coef", "growth_rate",
                "safeguard_warmup", "max_update_rms_ratio", "damping",
            }
        if opt == OptimizerType.ADAFACTOR:
            return {"eps", "clip_threshold", "decay_rate", "beta1", "weight_decay", "scale_parameter", "relative_step", "warmup_init"}
        if opt in {OptimizerType.LION, OptimizerType.LION_8BIT, OptimizerType.PAGED_LION_8BIT}:
            return {"betas", "use_triton", "decoupled_weight_decay"}
        if opt in {OptimizerType.SGD_NESTEROV, OptimizerType.SGD_NESTEROV_8BIT}:
            return {"momentum", "dampening", "nesterov", "maximize", "foreach"}
        if opt in {
            OptimizerType.DADAPTATION,
            OptimizerType.DADAPT_ADAM_PREPRINT,
            OptimizerType.DADAPT_ADAGRAD,
            OptimizerType.DADAPT_ADAM,
            OptimizerType.DADAPT_ADAN,
            OptimizerType.DADAPT_ADAN_IP,
            OptimizerType.DADAPT_LION,
            OptimizerType.DADAPT_SGD,
        }:
            return {"betas", "eps", "momentum", "growth_rate", "log_every", "decouple", "d0"}
        if opt in {
            OptimizerType.ADAMW_SCHEDULE_FREE,
            OptimizerType.RADAM_SCHEDULE_FREE,
            OptimizerType.SGD_SCHEDULE_FREE,
            OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
        }:
            return {"betas", "eps", "momentum", "warmup_steps", "r", "weight_lr_power", "foreach"}
        if opt == OptimizerType.PYTORCH_OPTIMIZER:
            return {
                "name", "optimizer_name", "optimizer", "betas", "eps", "momentum",
                "weight_decay", "decouple", "fixed_decay", "rectify", "n_sma_threshold",
                "degenerated_to_sgd", "amsgrad", "foreach", "maximize", "centralize_gradients",
                "normalize_gradients", "adam_debias", "stable_weight_decay",
            }
        if opt == OptimizerType.GENERIC:
            return {
                "name", "betas", "eps", "momentum", "weight_decay",
                "amsgrad", "foreach", "maximize", "nesterov", "dampening",
                "growth_rate", "d0", "d_coef", "decouple", "warmup_steps",
                "r", "weight_lr_power", "capturable", "fused",
            }
        return set()

    def _is_schedule_free_optimizer(self) -> bool:
        return self.config.optimizer in {
            OptimizerType.ADAMW_SCHEDULE_FREE,
            OptimizerType.RADAM_SCHEDULE_FREE,
            OptimizerType.SGD_SCHEDULE_FREE,
            OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
        }

    @staticmethod
    def _normalize_optimizer_backend(raw: Any) -> str:
        value = str(raw or "auto").strip().lower().replace("-", "_")
        aliases = {
            "": "auto",
            "default": "auto",
            "torch": "torch_adamw",
            "adamw": "torch_adamw",
            "foreach": "foreach_adamw",
            "multi_tensor": "foreach_adamw",
            "fused": "torch_fused",
            "torchfused": "torch_fused",
            "bnb": "bnb_8bit",
            "bitsandbytes": "bnb_8bit",
            "bitsandbytes_8bit": "bnb_8bit",
            "torchao": "ao_8bit",
            "torchao_8bit": "ao_8bit",
            "ao": "ao_8bit",
            "compile": "compiled_step",
            "compiled": "compiled_step",
            "lulynx": "lulynx_fused",
            "lulynx_fused_adamw": "lulynx_fused",
        }
        value = aliases.get(value.replace(" ", ""), value)
        if value not in {"auto", "torch_adamw", "foreach_adamw", "torch_fused", "bnb_8bit", "ao_8bit", "compiled_step", "apex", "lulynx_fused"}:
            return "auto"
        return value

    def _set_optimizer_backend_profile(
        self,
        requested: str,
        resolved: str,
        *,
        optimizer_type: str,
        optimizer_class: str = "",
        fallback_reason: str = "",
        notes: Optional[List[str]] = None,
    ) -> None:
        self._optimizer_backend_profile = {
            "requested": requested,
            "resolved": resolved,
            "optimizer_type": optimizer_type,
            "optimizer_class": optimizer_class,
            "fallback_reason": fallback_reason,
            "notes": list(notes or []),
        }
        self._attach_optimizer_profiles_to_training_loop()

    def _create_torch_adamw_optimizer(
        self,
        trainable_params,
        optimizer_args: Dict[str, Any],
        *,
        requested: str,
        resolved: str = "torch_adamw",
        fallback_reason: str = "",
        notes: Optional[List[str]] = None,
    ):
        kwargs = dict(optimizer_args)
        try:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **kwargs,
            )
        except TypeError as exc:
            if "fused" not in kwargs:
                raise
            kwargs.pop("fused", None)
            fallback_reason = fallback_reason or f"torch AdamW rejected fused=True: {exc}"
            resolved = "torch_adamw"
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **kwargs,
            )
        self._set_optimizer_backend_profile(
            requested,
            resolved,
            optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
            optimizer_class=type(optimizer).__name__,
            fallback_reason=fallback_reason,
            notes=notes,
        )
        return optimizer

    def _create_bnb_adamw8bit_optimizer(
        self,
        trainable_params,
        optimizer_args: Dict[str, Any],
        *,
        requested: str,
        fallback_reason: str = "",
        notes: Optional[List[str]] = None,
    ):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **optimizer_args,
            )
            self._set_optimizer_backend_profile(
                requested,
                "bnb_8bit",
                optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
                optimizer_class=type(optimizer).__name__,
                fallback_reason=fallback_reason,
                notes=notes,
            )
            return optimizer
        except ImportError:
            self._log("bitsandbytes not available, falling back to AdamW")
            return self._create_torch_adamw_optimizer(
                trainable_params,
                optimizer_args,
                requested=requested,
                resolved="torch_adamw",
                fallback_reason="bitsandbytes is not available",
            )

    def _create_ao_adamw8bit_optimizer(
        self,
        trainable_params,
        optimizer_args: Dict[str, Any],
        *,
        requested: str,
    ):
        """torchao AdamW8bit: 8-bit optimizer state with torch.compile-fused launches.

        Keeps the bnb-style VRAM saving but issues a handful of kernels per step
        instead of one dispatch per parameter block, so the optimizer-phase GPU
        stall of bnb AdamW8bit does not occur. Requires a working Triton JIT
        (the python-flashattention env ships one on Windows). Falls back to
        bnb → torch AdamW when torchao is unavailable.
        """
        ao_adamw8bit = None
        import_error = ""
        for module_path in ("torchao.optim", "torchao.prototype.low_bit_optim"):
            try:
                module = __import__(module_path, fromlist=["AdamW8bit"])
                ao_adamw8bit = getattr(module, "AdamW8bit", None)
                if ao_adamw8bit is not None:
                    break
            except Exception as exc:  # noqa: BLE001 - import surface varies across torchao versions
                import_error = f"{type(exc).__name__}: {exc}"
        if ao_adamw8bit is None:
            reason = f"torchao AdamW8bit unavailable ({import_error or 'no AdamW8bit export'})"
            self._log(f"{reason}; falling back to bitsandbytes AdamW8bit")
            return self._create_bnb_adamw8bit_optimizer(
                trainable_params,
                optimizer_args,
                requested=requested,
                fallback_reason=reason,
            )

        kwargs = dict(optimizer_args)
        dropped = [key for key in kwargs if key not in {"betas", "eps", "amsgrad"}]
        for key in dropped:
            kwargs.pop(key, None)
        notes = ["torchao 8-bit state with compiled multi-tensor launches (needs Triton JIT)."]
        if dropped:
            notes.append(f"Ignored torch-AdamW-only args for ao_8bit: {', '.join(sorted(dropped))}.")
        try:
            optimizer = ao_adamw8bit(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - constructor contract varies across torchao versions
            reason = f"torchao AdamW8bit rejected construction: {type(exc).__name__}: {exc}"
            self._log(f"{reason}; falling back to bitsandbytes AdamW8bit")
            return self._create_bnb_adamw8bit_optimizer(
                trainable_params,
                optimizer_args,
                requested=requested,
                fallback_reason=reason,
            )
        self._set_optimizer_backend_profile(
            requested,
            "ao_8bit",
            optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
            optimizer_class=type(optimizer).__name__,
            notes=notes,
        )
        return optimizer

    def _create_apex_adamw_optimizer(
        self,
        trainable_params,
        optimizer_args: Dict[str, Any],
        *,
        requested: str,
    ):
        try:
            from apex.optimizers import FusedAdam
        except Exception as exc:
            return self._create_torch_adamw_optimizer(
                trainable_params,
                optimizer_args,
                requested=requested,
                resolved="torch_adamw",
                fallback_reason=f"apex FusedAdam unavailable: {type(exc).__name__}: {exc}",
            )
        kwargs = dict(optimizer_args)
        for unsupported in ("amsgrad", "foreach", "fused"):
            kwargs.pop(unsupported, None)
        optimizer = FusedAdam(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_w_mode=True,
            **kwargs,
        )
        self._set_optimizer_backend_profile(
            requested,
            "apex",
            optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
            optimizer_class=type(optimizer).__name__,
            notes=["Unsupported AdamW-only args are dropped for apex FusedAdam."],
        )
        return optimizer

    def _create_auto_probed_adamw_optimizer(
        self,
        trainable_params,
        optimizer_args: Dict[str, Any],
        *,
        requested: str,
    ):
        """``optimizer_backend=auto`` probe chain for the AdamW family (non-8bit).

        Kernel-launch hygiene: prefer the fused single-kernel step, fall back to
        the multi-tensor ``foreach`` path, then plain AdamW. Construction-time
        ``RuntimeError`` is how torch rejects fused/foreach for unsupported
        param devices/dtypes, so each rung is probed by simply constructing.
        Explicit ``fused``/``foreach`` in optimizer_args pins the user's choice
        and skips probing entirely.
        """
        kwargs = dict(optimizer_args)
        if "fused" in kwargs or "foreach" in kwargs:
            return self._create_torch_adamw_optimizer(
                trainable_params,
                kwargs,
                requested=requested,
                resolved="torch_adamw",
                notes=["optimizer_args pin fused/foreach explicitly; auto probe skipped."],
            )
        if not torch.cuda.is_available():
            return self._create_torch_adamw_optimizer(
                trainable_params,
                kwargs,
                requested=requested,
                resolved="torch_adamw",
                fallback_reason="CUDA unavailable; fused/foreach probes skipped",
            )

        fused_kwargs = dict(kwargs)
        fused_kwargs["fused"] = True
        try:
            return self._create_torch_adamw_optimizer(
                trainable_params,
                fused_kwargs,
                requested=requested,
                resolved="torch_fused",
                notes=["auto probe selected fused AdamW (single-kernel optimizer step)."],
            )
        except (RuntimeError, ValueError) as exc:
            fused_reason = f"fused probe rejected: {type(exc).__name__}: {exc}"

        foreach_kwargs = dict(kwargs)
        foreach_kwargs["foreach"] = True
        try:
            return self._create_torch_adamw_optimizer(
                trainable_params,
                foreach_kwargs,
                requested=requested,
                resolved="foreach_adamw",
                fallback_reason=fused_reason,
                notes=["auto probe selected foreach AdamW (multi-tensor batched launches)."],
            )
        except (RuntimeError, ValueError) as exc:
            chain_reason = f"{fused_reason}; foreach probe rejected: {type(exc).__name__}: {exc}"

        return self._create_torch_adamw_optimizer(
            trainable_params,
            dict(kwargs),
            requested=requested,
            resolved="torch_adamw",
            fallback_reason=chain_reason,
        )

    def _create_adamw_backend_optimizer(self, trainable_params, optimizer_args: Dict[str, Any]):
        requested = self._normalize_optimizer_backend(getattr(self.config, "optimizer_backend", "auto"))
        opt = self.config.optimizer
        notes: List[str] = []

        if requested in {"auto", "compiled_step"}:
            # compiled_step constructs like auto: the generic torch.compile wrap
            # happens later in _maybe_wrap_compiled_step (skipped if fused won).
            if opt == OptimizerType.ADAMW_8BIT:
                # Respect the user's 8-bit-state choice; just surface the trade-off once.
                self._log(
                    "optimizer_backend=auto keeps bnb AdamW8bit (8-bit state, per-parameter "
                    "micro-kernel dispatch). Switching optimizer to AdamW resolves to the fused "
                    "single-kernel step and typically removes the optimizer-phase GPU stall (~25% step time)."
                )
                return self._create_bnb_adamw8bit_optimizer(trainable_params, optimizer_args, requested=requested)
            return self._create_auto_probed_adamw_optimizer(trainable_params, optimizer_args, requested=requested)

        if requested == "bnb_8bit":
            return self._create_bnb_adamw8bit_optimizer(trainable_params, optimizer_args, requested=requested)

        if requested == "ao_8bit":
            return self._create_ao_adamw8bit_optimizer(trainable_params, optimizer_args, requested=requested)

        if requested == "foreach_adamw":
            kwargs = dict(optimizer_args)
            if "foreach" in kwargs and kwargs["foreach"] is not True:
                notes.append("optimizer_args.foreach overrides optimizer_backend=foreach_adamw.")
                return self._create_torch_adamw_optimizer(
                    trainable_params,
                    kwargs,
                    requested=requested,
                    resolved="torch_adamw",
                    fallback_reason="foreach was explicitly disabled in optimizer_args",
                    notes=notes,
                )
            kwargs.setdefault("foreach", True)
            return self._create_torch_adamw_optimizer(
                trainable_params,
                kwargs,
                requested=requested,
                resolved="foreach_adamw",
                notes=notes,
            )

        if requested == "torch_fused":
            kwargs = dict(optimizer_args)
            if "fused" in kwargs and kwargs["fused"] is not True:
                notes.append("optimizer_args.fused overrides optimizer_backend=torch_fused.")
                return self._create_torch_adamw_optimizer(
                    trainable_params,
                    kwargs,
                    requested=requested,
                    resolved="torch_adamw",
                    fallback_reason="fused was explicitly disabled in optimizer_args",
                    notes=notes,
                )
            if not torch.cuda.is_available():
                return self._create_torch_adamw_optimizer(
                    trainable_params,
                    kwargs,
                    requested=requested,
                    resolved="torch_adamw",
                    fallback_reason="torch fused AdamW requires CUDA",
                )
            kwargs.setdefault("fused", True)
            return self._create_torch_adamw_optimizer(
                trainable_params,
                kwargs,
                requested=requested,
                resolved="torch_fused",
                notes=notes,
            )

        if requested == "apex":
            return self._create_apex_adamw_optimizer(trainable_params, optimizer_args, requested=requested)

        if requested == "lulynx_fused":
            from .fused_adamw import FusedAdamW
            kwargs = dict(optimizer_args)
            dropped = []
            for unsupported in ("foreach", "fused"):
                if unsupported in kwargs:
                    kwargs.pop(unsupported, None)
                    dropped.append(unsupported)
            optimizer = FusedAdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **kwargs,
            )
            notes = ["Lulynx fused AdamW is a pure PyTorch compatibility backend, not a custom CUDA fused kernel."]
            if dropped:
                notes.append(f"Ignored PyTorch AdamW-only args for lulynx_fused: {', '.join(dropped)}.")
            self._set_optimizer_backend_profile(
                requested,
                "lulynx_fused",
                optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
                optimizer_class=type(optimizer).__name__,
                notes=notes,
            )
            return optimizer

        return self._create_torch_adamw_optimizer(
            trainable_params,
            optimizer_args,
            requested=requested,
            resolved="torch_adamw",
        )

    def _resolve_prodigy_d_args(self, optimizer_args: Dict[str, Any]) -> tuple[Dict[str, Any], float, float]:
        resolved_args = dict(optimizer_args)
        raw_d0 = resolved_args.pop("d0", getattr(self.config, "opt_prodigy_d0", 1e-6))
        raw_d_coef = resolved_args.pop("d_coef", getattr(self.config, "opt_prodigy_d_coef", 1.0))
        try:
            d0 = float(1e-6 if raw_d0 in {None, ""} else raw_d0)
        except (TypeError, ValueError):
            d0 = 1e-6
        try:
            d_coef = float(1.0 if raw_d_coef in {None, ""} else raw_d_coef)
        except (TypeError, ValueError):
            d_coef = 1.0
        return resolved_args, d0, d_coef

    def _create_bitsandbytes_optimizer(self, trainable_params, optimizer_args: Dict[str, Any]):
        try:
            import bitsandbytes as bnb
        except ImportError:
            self._log("bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        opt = self.config.optimizer
        class_name_by_type = {
            OptimizerType.PAGED_ADAMW: "PagedAdamW",
            OptimizerType.PAGED_ADAMW_32BIT: "PagedAdamW32bit",
            OptimizerType.PAGED_ADAMW_8BIT: "PagedAdamW8bit",
            OptimizerType.PAGED_LION_8BIT: "PagedLion8bit",
            OptimizerType.SGD_NESTEROV_8BIT: "SGD8bit",
        }
        class_name = class_name_by_type[opt]
        optimizer_class = getattr(bnb.optim, class_name, None)
        if optimizer_class is None:
            self._log(f"bitsandbytes optimizer {class_name} is unavailable, falling back to AdamW")
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        kwargs = dict(optimizer_args)
        if opt == OptimizerType.SGD_NESTEROV_8BIT:
            kwargs.setdefault("momentum", 0.9)
            kwargs["nesterov"] = True
        return optimizer_class(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **kwargs,
        )

    def _create_dadapt_optimizer(self, trainable_params, optimizer_args: Dict[str, Any]):
        opt = self.config.optimizer
        try:
            import dadaptation
            import dadaptation.experimental as dadapt_experimental
        except ImportError:
            plugin_name_by_type = {
                OptimizerType.DADAPTATION: "DAdaptAdam",
                OptimizerType.DADAPT_ADAM_PREPRINT: "DAdaptAdam",
                OptimizerType.DADAPT_ADAGRAD: "DAdaptAdaGrad",
                OptimizerType.DADAPT_ADAM: "DAdaptAdam",
                OptimizerType.DADAPT_ADAN: "DAdaptAdan",
                OptimizerType.DADAPT_ADAN_IP: "DAdaptAdan",
                OptimizerType.DADAPT_LION: "DAdaptLion",
                OptimizerType.DADAPT_SGD: "DAdaptSGD",
            }
            plugin_name = plugin_name_by_type.get(opt, "DAdaptAdam")
            plugin_args = dict(optimizer_args)
            if opt == OptimizerType.DADAPT_ADAGRAD:
                plugin_args.setdefault("eps", 1e-6)
            plugin_args.setdefault("name", plugin_name)
            try:
                from .optimizer_plugin_bridge import create_pytorch_optimizer

                self._log(f"dadaptation not available, using pytorch_optimizer {plugin_name} route")
                return create_pytorch_optimizer(
                    trainable_params,
                    optimizer_name=plugin_name,
                    lr=1.0,
                    weight_decay=self.config.weight_decay,
                    optimizer_args=plugin_args,
                )
            except Exception as exc:
                self._log(f"pytorch_optimizer {plugin_name} route unavailable, falling back to AdamW: {exc}")
                return torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )

        if opt in {OptimizerType.DADAPTATION, OptimizerType.DADAPT_ADAM_PREPRINT}:
            optimizer_class = dadapt_experimental.DAdaptAdamPreprint
        elif opt == OptimizerType.DADAPT_ADAGRAD:
            optimizer_args = dict(optimizer_args)
            optimizer_args.setdefault("eps", 1e-6)
            optimizer_class = dadaptation.DAdaptAdaGrad
        elif opt == OptimizerType.DADAPT_ADAM:
            optimizer_class = dadaptation.DAdaptAdam
        elif opt == OptimizerType.DADAPT_ADAN:
            optimizer_class = dadaptation.DAdaptAdan
        elif opt == OptimizerType.DADAPT_ADAN_IP:
            optimizer_class = dadapt_experimental.DAdaptAdanIP
        elif opt == OptimizerType.DADAPT_LION:
            optimizer_class = dadaptation.DAdaptLion
        else:
            optimizer_class = dadaptation.DAdaptSGD
        return optimizer_class(
            trainable_params,
            lr=1.0,
            weight_decay=self.config.weight_decay,
            **optimizer_args,
        )

    def _create_schedulefree_optimizer(self, trainable_params, optimizer_args: Dict[str, Any]):
        try:
            import schedulefree
        except ImportError:
            self._log("schedulefree not available, falling back to AdamW")
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        if self.config.optimizer == OptimizerType.RADAM_SCHEDULE_FREE:
            optimizer_class = schedulefree.RAdamScheduleFree
        elif self.config.optimizer == OptimizerType.SGD_SCHEDULE_FREE:
            optimizer_class = schedulefree.SGDScheduleFree
            optimizer_args = dict(optimizer_args)
            optimizer_args.setdefault("momentum", 0.9)
        else:
            optimizer_class = schedulefree.AdamWScheduleFree
        optimizer = optimizer_class(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            **optimizer_args,
        )
        if hasattr(optimizer, "train"):
            optimizer.train()
        return optimizer

    def _create_prodigy_plus_schedule_free_optimizer(self, trainable_params, optimizer_args: Dict[str, Any]):
        try:
            from prodigyplus import ProdigyPlusScheduleFree
        except ImportError:
            self._log("prodigyplus not available, falling back to AdamW")
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        resolved_args, d0, d_coef = self._resolve_prodigy_d_args(optimizer_args)
        optimizer = ProdigyPlusScheduleFree(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            d0=d0,
            d_coef=d_coef,
            **resolved_args,
        )
        if hasattr(optimizer, "train"):
            optimizer.train()
        return optimizer

    def _dora_param_groups(self, trainable_params) -> list:
        """Separate DoRA magnitude parameters into their own param group with weight_decay=0.

        DoRA's magnitude vector (``m``) should not be decayed — it represents a
        norm that should be free to grow/shrink without L2 penalty pushing it
        toward zero.  When DoRA is active, we split the parameter list into two
        groups: magnitude params with weight_decay=0 and everything else with
        the configured weight_decay.
        """
        if not bool(getattr(self.config, "use_dora", False)) and not any(
            getattr(self.config, k, False) for k in ("dora", "enable_dora")
        ):
            return trainable_params

        # If trainable_params is already a list of dicts (param groups),
        # iterate through and add magnitude-specific groups.
        if isinstance(trainable_params, list) and trainable_params and isinstance(trainable_params[0], dict):
            magnitude_params = []
            other_params = []
            for group in trainable_params:
                params = group.get("params", [])
                lr = group.get("lr", self.config.learning_rate)
                wd = group.get("weight_decay", self.config.weight_decay)
                for p in params:
                    if getattr(p, "_dora_magnitude", False):
                        magnitude_params.append(p)
                    else:
                        other_params.append(p)
            if not magnitude_params:
                return trainable_params
            return [
                {"params": other_params, "lr": lr, "weight_decay": wd},
                {"params": magnitude_params, "lr": lr, "weight_decay": float(getattr(self.config, "dora_magnitude_weight_decay", 0.0))},
            ]

        # If trainable_params is a flat list of parameters
        if isinstance(trainable_params, (list, tuple)):
            magnitude_params = []
            other_params = []
            for p in trainable_params:
                if getattr(p, "_dora_magnitude", False):
                    magnitude_params.append(p)
                else:
                    other_params.append(p)
            if not magnitude_params:
                return trainable_params
            dora_mag_wd = float(getattr(self.config, "dora_magnitude_weight_decay", 0.0))
            return [
                {"params": other_params, "weight_decay": self.config.weight_decay},
                {"params": magnitude_params, "weight_decay": dora_mag_wd},
            ]

        return trainable_params

    def _lora_plus_param_groups(self, trainable_params) -> list:
        if not bool(getattr(self.config, "lora_plus_enabled", False)):
            return trainable_params
        if isinstance(trainable_params, list) and trainable_params and isinstance(trainable_params[0], dict):
            self._mark_lora_plus_runtime_outcome(
                applied=False,
                fallback_reason="LoRA+ param-group split conflicts with pre-grouped optimizer params; keeping existing groups.",
                note="LoRA+ stayed profile-only because grouped optimizer params were already active.",
            )
            self._log("LoRA+ requested, but existing grouped LR/Block Weight param groups are active; keeping existing groups.")
            return trainable_params

        injected = getattr(self.lora_injector, "injected_layers", {}) if self.lora_injector is not None else {}
        from .lora_plus_param_groups import build_lora_plus_param_groups

        plan = build_lora_plus_param_groups(
            injected_layers=injected,
            trainable_params=trainable_params,
            base_lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            b_lr_ratio=float(getattr(self.config, "lora_plus_lr_ratio", 16.0) or 16.0),
        )
        self._lora_plus_runtime_details = {
            **plan.as_dict(),
            "b_lr_ratio": float(getattr(self.config, "lora_plus_lr_ratio", 16.0) or 16.0),
            "param_group_count": len(plan.param_groups),
        }
        if plan.applied:
            self._mark_lora_plus_runtime_outcome(applied=True, note=plan.note)
            self._log(
                "LoRA+ param groups enabled: "
                f"A={plan.lora_a_count}, B={plan.lora_b_count}, other={plan.other_count}, "
                f"B_lr_ratio={float(getattr(self.config, 'lora_plus_lr_ratio', 16.0) or 16.0)}"
            )
            return plan.param_groups
        self._mark_lora_plus_runtime_outcome(
            applied=False,
            fallback_reason=plan.fallback_reason,
            note=plan.note,
        )
        return trainable_params

    def _muon_param_groups(self, trainable_params) -> list:
        """Regroup params for the Muon optimizer: 2D matrices use Muon, the
        rest (bias/norm/magnitude/scalars) fall back to AdamW with no decay.

        No-op unless ``OptimizerType.MUON`` is selected.  Re-flattens whatever
        grouping arrived (DoRA / LoRA+ / prefix) and re-splits purely by rank,
        because Muon's A/B handling is the orthogonalization itself, not an
        lr-ratio split.
        """
        if self.config.optimizer != OptimizerType.MUON:
            return trainable_params

        seen: set[int] = set()
        flat: list = []

        def _collect(obj) -> None:
            if isinstance(obj, dict):
                for p in obj.get("params", []):
                    if isinstance(p, torch.nn.Parameter) and p.requires_grad and id(p) not in seen:
                        seen.add(id(p))
                        flat.append(p)
            elif isinstance(obj, torch.nn.Parameter):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    flat.append(obj)

        for item in list(trainable_params or []):
            _collect(item)

        muon_params = [p for p in flat if p.dim() == 2]
        other_params = [p for p in flat if p.dim() != 2]
        if not muon_params and not other_params:
            return trainable_params

        base_lr = float(self.config.learning_rate)
        wd = float(self.config.weight_decay)
        ratio = float(getattr(self.config, "muon_lr_ratio", 1.0) or 1.0)
        groups: list[dict[str, Any]] = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True, "lr": base_lr, "weight_decay": wd})
        if other_params:
            groups.append({"params": other_params, "use_muon": False, "lr": base_lr * ratio, "weight_decay": 0.0})
        self._log(
            f"Muon param groups: {len(muon_params)} 2D matrices (Muon), "
            f"{len(other_params)} non-2D params (AdamW fallback, no decay)"
        )
        return groups

    def _mark_lora_plus_runtime_outcome(
        self,
        *,
        applied: bool,
        fallback_reason: str = "",
        note: str = "",
    ) -> None:
        profile = getattr(self, "_advanced_optimizer_strategy_profile", None)
        if not profile:
            return
        if str(profile.get("resolved") or "") != "lora_plus":
            return

        from .advanced_optimizer_strategy import apply_lora_plus_runtime_outcome

        self._advanced_optimizer_strategy_profile = apply_lora_plus_runtime_outcome(
            profile,
            applied=applied,
            fallback_reason=fallback_reason,
            note=note,
            runtime_details=getattr(self, "_lora_plus_runtime_details", None),
        )
        self._attach_optimizer_profiles_to_training_loop()
    def _mark_galore_runtime_outcome(
        self,
        *,
        applied: bool,
        fallback_reason: str = "",
        note: str = "",
    ) -> None:
        profile = getattr(self, "_advanced_optimizer_strategy_profile", None)
        if not profile:
            return
        if str(profile.get("resolved") or "") != "galore":
            return

        from .advanced_optimizer_strategy import apply_galore_runtime_outcome

        self._advanced_optimizer_strategy_profile = apply_galore_runtime_outcome(
            profile,
            applied=applied,
            fallback_reason=fallback_reason,
            note=note,
        )
        self._attach_optimizer_profiles_to_training_loop()

    @staticmethod
    def _scheduler_allowed_args() -> set[str]:
        return {
            "eta_min", "t_0", "t_mult", "start_factor", "end_factor",
            "total_iters", "power", "max_lr", "pct_start", "anneal_strategy",
            "div_factor", "final_div_factor", "three_phase",
            "min_lr_ratio", "min_lr_rate", "num_cycles", "rules", "schedule",
            "ema_alpha", "min_delta", "relative_delta", "patience", "cooldown",
            "max_hold_steps", "late_loss_gamma", "lock_weight_threshold",
            "min_advance_ratio", "hold_on_improvement",
        }

    def _filter_optimizer_trainable_params(self, params: Any) -> Any:
        """Drop frozen/duplicate parameters before constructing the optimizer."""
        seen: set[int] = set()

        def keep_param(param: Any) -> bool:
            if not isinstance(param, torch.nn.Parameter):
                return False
            if not bool(getattr(param, "requires_grad", False)):
                return False
            ident = id(param)
            if ident in seen:
                return False
            seen.add(ident)
            return True

        if isinstance(params, list) and params and isinstance(params[0], dict):
            filtered_groups = []
            dropped = 0
            for group in params:
                original_params = list(group.get("params", []))
                group_params = [p for p in original_params if keep_param(p)]
                dropped += len(original_params) - len(group_params)
                if not group_params:
                    continue
                next_group = dict(group)
                next_group["params"] = group_params
                filtered_groups.append(next_group)
            if dropped:
                self._log(f"[runtime-opt] optimizer filtered {dropped} frozen/duplicate params from param groups")
            return filtered_groups

        flat_params = list(params or [])
        filtered = [p for p in flat_params if keep_param(p)]
        dropped = len(flat_params) - len(filtered)
        if dropped:
            self._log(f"[runtime-opt] optimizer filtered {dropped} frozen/duplicate params")
        return filtered

    def _optimizer_param_names(self) -> Dict[int, str]:
        names: Dict[int, str] = {}

        def add_named_parameters(prefix: str, module: Any) -> None:
            if module is None or not hasattr(module, "named_parameters"):
                return
            try:
                named_parameters = module.named_parameters()
            except Exception:
                return
            for name, param in named_parameters:
                full_name = f"{prefix}.{name}" if prefix else name
                names.setdefault(id(param), full_name)

        if self.model is not None:
            add_named_parameters("", self.model)
            seen_components: set[int] = set()
            component_names = (
                "unet",
                "dit",
                "transformer",
                "denoiser",
                "text_encoder",
                "text_encoder_1",
                "text_encoder_2",
                "text_encoder_3",
                "vae",
            )
            for component_name in component_names:
                component = getattr(self.model, component_name, None)
                if component is None:
                    continue
                component_id = id(component)
                if component_id in seen_components:
                    continue
                seen_components.add(component_id)
                add_named_parameters(component_name, component)
            try:
                model_components = vars(self.model)
            except TypeError:
                model_components = {}
            for component_name, component in model_components.items():
                if component_name in component_names:
                    continue
                component_id = id(component)
                if component_id in seen_components:
                    continue
                seen_components.add(component_id)
                add_named_parameters(component_name, component)
        injected = getattr(self.lora_injector, "injected_layers", {}) if self.lora_injector is not None else {}
        if isinstance(injected, dict):
            for prefix, layer in injected.items():
                if not hasattr(layer, "named_parameters"):
                    continue
                for name, param in layer.named_parameters():
                    names.setdefault(id(param), f"{prefix}.{name}")
        return names

    def _mn_lora_plus_plus_config(self) -> Dict[str, Any]:
        profile = str(getattr(self.config, "mn_lora_plus_plus_profile", "balanced") or "balanced").strip().lower()
        profiles: Dict[str, Dict[str, float]] = {
            "safe": {
                "lr_up": 1.01,
                "lr_down": 0.95,
                "min_mult": 0.25,
                "max_mult": 2.0,
                "lora_up_max_mult": 1.25,
                "protected_max_mult": 1.0,
                "update_rms_cap": 0.01,
            },
            "balanced": {
                "lr_up": 1.03,
                "lr_down": 0.90,
                "min_mult": 0.25,
                "max_mult": 2.5,
                "lora_up_max_mult": 1.5,
                "protected_max_mult": 1.0,
                "update_rms_cap": 0.02,
            },
            "aggressive": {
                "lr_up": 1.05,
                "lr_down": 0.85,
                "min_mult": 0.20,
                "max_mult": 3.0,
                "lora_up_max_mult": 2.0,
                "protected_max_mult": 1.0,
                "update_rms_cap": 0.03,
            },
        }
        if profile not in {*profiles, "custom"}:
            self._log(f"Unknown MN-LoRA++ profile {profile!r}, using balanced.")
            profile = "balanced"
        values = dict(profiles.get(profile, {}))
        if profile == "custom":
            values = {
                "lr_up": float(getattr(self.config, "mn_lora_plus_plus_lr_up", 1.01)),
                "lr_down": float(getattr(self.config, "mn_lora_plus_plus_lr_down", 0.95)),
                "min_mult": float(getattr(self.config, "mn_lora_plus_plus_min_mult", 0.25)),
                "max_mult": float(getattr(self.config, "mn_lora_plus_plus_max_mult", 2.0)),
                "lora_up_max_mult": float(getattr(self.config, "mn_lora_plus_plus_lora_up_max_mult", 1.25)),
                "protected_max_mult": float(getattr(self.config, "mn_lora_plus_plus_protected_max_mult", 1.0)),
                "update_rms_cap": float(getattr(self.config, "mn_lora_plus_plus_update_rms_cap", 0.01)),
            }
        values.update({
            "enabled": bool(getattr(self.config, "mn_lora_plus_plus_enabled", False)),
            "rank_adapt": bool(getattr(self.config, "mn_lora_plus_plus_rank_adapt", True)),
            "module_adapt": bool(getattr(self.config, "mn_lora_plus_plus_module_adapt", True)),
        })
        return values

    def _mn_lora_trust_region_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "mn_lora_trust_region_enabled", True)),
            "max_update_rms_ratio": float(getattr(self.config, "mn_lora_trust_region_max_update_rms_ratio", 0.01)),
            "max_update_norm_ratio": float(getattr(self.config, "mn_lora_trust_region_max_update_norm_ratio", 0.10)),
            "hotspot_only": bool(getattr(self.config, "mn_lora_trust_region_hotspot_only", False)),
        }

    def _mn_lora_kfac_lite_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "mn_lora_kfac_lite_enabled", False)),
            "ema_decay": float(getattr(self.config, "mn_lora_kfac_lite_ema_decay", 0.95)),
            "damping": float(getattr(self.config, "mn_lora_kfac_lite_damping", 1e-3)),
            "update_interval": int(getattr(self.config, "mn_lora_kfac_lite_update_interval", 1)),
            "precondition_interval": int(getattr(self.config, "mn_lora_kfac_lite_precondition_interval", 1)),
            "max_samples": int(getattr(self.config, "mn_lora_kfac_lite_max_samples", 2048)),
            "grad_clip": float(getattr(self.config, "mn_lora_kfac_lite_grad_clip", 3.0)),
            "stacked_grad_clip": float(getattr(self.config, "mn_lora_kfac_lite_stacked_grad_clip", 2.0)),
            "active_ratio": float(getattr(self.config, "mn_lora_kfac_lite_active_ratio", 0.40)),
            "warmup_steps": int(getattr(self.config, "mn_lora_kfac_lite_warmup_steps", 10)),
            "refresh_interval": int(getattr(self.config, "mn_lora_kfac_lite_refresh_interval", 10)),
            "min_active_modules": int(getattr(self.config, "mn_lora_kfac_lite_min_active_modules", 16)),
        }

    def _mn_lora_effective_delta_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "mn_lora_effective_delta_enabled", True)),
            "clip_enabled": bool(getattr(self.config, "mn_lora_effective_delta_clip_enabled", True)),
            "max_norm_ratio": float(getattr(self.config, "mn_lora_effective_delta_max_norm_ratio", 0.25)),
            "max_rms_ratio": float(getattr(self.config, "mn_lora_effective_delta_max_rms_ratio", 0.05)),
            "fisher_weighted": bool(getattr(self.config, "mn_lora_effective_delta_fisher_weighted", True)),
            "fisher_beta": float(getattr(self.config, "mn_lora_effective_delta_fisher_beta", 0.95)),
            "fisher_strength": float(getattr(self.config, "mn_lora_effective_delta_fisher_strength", 1.0)),
            "fisher_max_weight": float(getattr(self.config, "mn_lora_effective_delta_fisher_max_weight", 4.0)),
        }

    def _mn_lora_fisher_ewc_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "mn_lora_fisher_ewc_enabled", True)),
            "lambda_ewc": float(getattr(self.config, "mn_lora_fisher_ewc_lambda", 1e-4)),
            "fisher_beta": float(getattr(self.config, "mn_lora_fisher_ewc_beta", 0.95)),
            "start_step": int(getattr(self.config, "mn_lora_fisher_ewc_start_step", 1)),
            "update_interval": int(getattr(self.config, "mn_lora_fisher_ewc_update_interval", 5)),
            "max_penalty_norm_ratio": float(getattr(self.config, "mn_lora_fisher_ewc_max_penalty_norm_ratio", 0.25)),
        }

    def _mn_lora_gradient_conflict_config(self) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "mn_lora_gradient_conflict_enabled", False)),
            "conflict_threshold": float(getattr(self.config, "mn_lora_gradient_conflict_threshold", 0.0)),
            "protect_main_gradient": bool(getattr(self.config, "mn_lora_gradient_conflict_protect_main", True)),
            "reduction": "sum",
        }

    def _auto_prodigy_config(self) -> Dict[str, Any]:
        profile = str(getattr(self.config, "auto_prodigy_profile", "balanced") or "balanced").strip().lower()
        profiles: Dict[str, Dict[str, Any]] = {
            "safe": {
                "d0": 5e-7,
                "d_coef": 0.75,
                "growth_rate": 1.01,
                "max_update_rms_ratio": 0.005,
                "damping": 1.5,
                "beta3": 0.995,
                "safeguard_warmup": True,
            },
            "balanced": {
                "d0": 1e-6,
                "d_coef": 1.0,
                "growth_rate": 1.02,
                "max_update_rms_ratio": 0.01,
                "damping": 1.0,
                "beta3": 0.99,
                "safeguard_warmup": True,
            },
            "aggressive": {
                "d0": 3e-6,
                "d_coef": 1.5,
                "growth_rate": 1.08,
                "max_update_rms_ratio": 0.03,
                "damping": 0.75,
                "beta3": 0.98,
                "safeguard_warmup": True,
            },
        }
        if profile not in {*profiles, "custom"}:
            self._log(f"Unknown AutoProdigy profile {profile!r}, using balanced.")
            profile = "balanced"
        if profile == "custom":
            return {
                "d0": float(getattr(self.config, "auto_prodigy_d0", 1e-6)),
                "d_coef": float(getattr(self.config, "auto_prodigy_d_coef", 1.0)),
                "growth_rate": float(getattr(self.config, "auto_prodigy_growth_rate", 1.02)),
                "max_update_rms_ratio": float(getattr(self.config, "auto_prodigy_max_update_rms_ratio", 0.01)),
                "damping": float(getattr(self.config, "auto_prodigy_damping", 1.0)),
                "beta3": float(getattr(self.config, "auto_prodigy_beta3", 0.99)),
                "safeguard_warmup": bool(getattr(self.config, "auto_prodigy_safeguard_warmup", True)),
            }
        return dict(profiles[profile])

    def _make_lr_finder_step_fn(self, unet, optimizer, dataloader):
        """Build a step callable for LRFinder: one forward+backward+step cycle."""
        import torch
        _dl_iter = iter(dataloader)

        def _step():
            nonlocal _dl_iter
            try:
                batch = next(_dl_iter)
            except StopIteration:
                _dl_iter = iter(dataloader)
                batch = next(_dl_iter)
            images = batch.get("images") or batch.get("pixel_values")
            if images is None:
                return 0.0
            images = images.to(self.device, dtype=self.dtype)
            latents = self._encode_latents_with_vae(images)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device).long()
            noisy = self.model.noise_scheduler.add_noise(latents, noise, timesteps)
            captions = batch.get("captions", [""] * latents.shape[0])
            prompt_embeds = self._encode_prompt(captions)
            enc_hs = prompt_embeds if not isinstance(prompt_embeds, dict) else prompt_embeds.get("encoder_hidden_states", prompt_embeds)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                pred = unet(sample=noisy, timestep=timesteps, encoder_hidden_states=enc_hs).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        return _step

    def _create_optimizer(self):
        """创建优化器(统一出口:所有路线在此经过 compiled_step 包装缝)"""
        optimizer = self._create_optimizer_impl()
        return self._maybe_wrap_compiled_step(optimizer)

    def _maybe_wrap_compiled_step(self, optimizer):
        """Opt-in torch.compile wrap for any optimizer's step (optimizer_backend=compiled_step)."""
        if optimizer is None:
            return optimizer
        requested = self._normalize_optimizer_backend(getattr(self.config, "optimizer_backend", "auto"))
        if requested != "compiled_step":
            return optimizer
        profile = getattr(self, "_optimizer_backend_profile", None) or {}
        if profile.get("resolved") == "torch_fused":
            note = "compiled_step skipped: auto resolved to torch_fused (already a single-kernel step)"
            self._log(f"[compiled_step] {note}")
            profile.setdefault("notes", []).append(note)
            return optimizer
        from .compiled_step_optimizer import wrap_optimizer_step_compiled
        report = wrap_optimizer_step_compiled(optimizer, log=self._log)
        if profile:
            profile.setdefault("notes", []).extend(report.get("notes", []))
            if report.get("wrapped"):
                profile["resolved"] = f"compiled_step({profile.get('resolved') or type(optimizer).__name__})"
        else:
            self._set_optimizer_backend_profile(
                requested,
                f"compiled_step({type(optimizer).__name__})" if report.get("wrapped") else "eager",
                optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
                optimizer_class=type(optimizer).__name__,
                fallback_reason=str(report.get("skipped_reason") or ""),
                notes=list(report.get("notes", [])),
            )
        return optimizer

    def _create_optimizer_impl(self):
        """创建优化器"""
        from .advanced_optimizer_strategy import resolve_advanced_optimizer_strategy

        self._advanced_optimizer_strategy_profile = resolve_advanced_optimizer_strategy(self.config).as_dict()
        raw_optimizer_args = getattr(self.config, "optimizer_args", "")
        if self.config.optimizer in {OptimizerType.PYTORCH_OPTIMIZER, OptimizerType.GENERIC}:
            optimizer_args = self._parse_custom_args(raw_optimizer_args)
        else:
            optimizer_args = self._filtered_custom_args(
                raw_optimizer_args,
                self._optimizer_allowed_args(),
                "optimizer_args",
            )
        if self.config.semantic_tuner_enabled:
            trainable_params = self.trainable_params
        else:
            # Check for Anima grouped LR first
            anima_groups = self._build_anima_grouped_param_groups()
            if anima_groups is not None:
                trainable_params = anima_groups
            elif self._block_weight_manager and hasattr(self.lora_injector, "get_param_groups"):
                trainable_params = self.lora_injector.get_param_groups(
                    base_lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
                if not trainable_params:
                    raise ValueError("No optimizer parameter groups available after Block Weight filtering.")
            else:
                trainable_params = self.lora_injector.get_trainable_params()

        extra_param_sets = []
        easy_control = getattr(self, "_easy_control", None)
        if easy_control is not None:
            extra_param_sets.append(easy_control.get_trainable_params())
        ip_adapter = getattr(self, "_ip_adapter", None)
        if ip_adapter is not None:
            extra_param_sets.append(ip_adapter.get_trainable_params())
        easycontrol_v2_adapter = getattr(self, "_easycontrol_v2_adapter", None)
        if easycontrol_v2_adapter is not None:
            extra_param_sets.append(easycontrol_v2_adapter.get_trainable_params())
        repa_projector = getattr(self, "_repa_projector", None)
        if repa_projector is not None:
            extra_param_sets.append(list(repa_projector.parameters()))
        if bool(getattr(self.config, "reft_enabled", False)) and self.model is not None:
            from .reft import get_reft_params
            extra_param_sets.append(get_reft_params(self.model.unet))
        for extra_params in extra_param_sets:
            if not extra_params:
                continue
            if isinstance(trainable_params, list) and trainable_params and isinstance(trainable_params[0], dict):
                trainable_params.append({
                    "params": extra_params,
                    "lr": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                })
            else:
                trainable_params.extend(extra_params)

        # DoRA: separate magnitude params with weight_decay=0
        trainable_params = self._dora_param_groups(trainable_params)
        trainable_params = self._lora_plus_param_groups(trainable_params)

        # Prefix/Postfix tuning: add soft-prompt params to the optimizer (#113)
        prefix_length = int(getattr(self.config, "prefix_tuning_length", 0) or 0)
        postfix_length = int(getattr(self.config, "postfix_tuning_length", 0) or 0)
        if (prefix_length > 0 or postfix_length > 0) and self.model is not None:
            from .prefix_tuning import get_prefix_tuning_params
            sp_params = get_prefix_tuning_params(self.model)
            if sp_params:
                # Soft-prompt params use a separate param group with no weight decay
                if isinstance(trainable_params, list) and len(trainable_params) > 0 and isinstance(trainable_params[0], dict):
                    # Already a list of param groups — add a dedicated group
                    trainable_params.append({
                        "params": sp_params,
                        "lr": self.config.learning_rate,
                        "weight_decay": 0.0,  # no weight decay on soft prompts
                    })
                else:
                    # Flat param list — just extend
                    trainable_params.extend(sp_params)

        # Muon: regroup into 2D (Muon) vs non-2D (AdamW fallback). Runs last so
        # it re-flattens DoRA/LoRA+/prefix groups and re-classifies by rank.
        trainable_params = self._muon_param_groups(trainable_params)

        trainable_params = self._filter_optimizer_trainable_params(trainable_params)
        if not trainable_params:
            raise ValueError("No trainable parameters available for optimizer after filtering frozen params.")

        optimizer = None
        if self.config.optimizer == OptimizerType.ADAMW:
            optimizer = self._create_adamw_backend_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer == OptimizerType.ADAMW_8BIT:
            optimizer = self._create_adamw_backend_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer == OptimizerType.KAHAN_ADAMW_8BIT:
            from .kahan_adamw8bit import KahanAdamW8bit
            optimizer = KahanAdamW8bit(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **optimizer_args,
            )
            self._log(f"Using KahanAdamW8bit (8-bit moments + Kahan compensated summation)")
        elif self.config.optimizer in {
            OptimizerType.PAGED_ADAMW,
            OptimizerType.PAGED_ADAMW_32BIT,
            OptimizerType.PAGED_ADAMW_8BIT,
            OptimizerType.PAGED_LION_8BIT,
            OptimizerType.SGD_NESTEROV_8BIT,
        }:
            optimizer = self._create_bitsandbytes_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer == OptimizerType.PRODIGY:
            try:
                from prodigyopt import Prodigy
                prodigy_args, d0, d_coef = self._resolve_prodigy_d_args(optimizer_args)
                optimizer = Prodigy(
                    trainable_params,
                    lr=1.0,  # Prodigy 自动调整
                    weight_decay=self.config.weight_decay,
                    d0=d0,
                    d_coef=d_coef,
                    **prodigy_args,
                )
            except ImportError:
                self._log("prodigyopt not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    **optimizer_args,
                )
        elif self.config.optimizer in {
            OptimizerType.DADAPTATION,
            OptimizerType.DADAPT_ADAM_PREPRINT,
            OptimizerType.DADAPT_ADAGRAD,
            OptimizerType.DADAPT_ADAM,
            OptimizerType.DADAPT_ADAN,
            OptimizerType.DADAPT_ADAN_IP,
            OptimizerType.DADAPT_LION,
            OptimizerType.DADAPT_SGD,
        }:
            optimizer = self._create_dadapt_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer in {
            OptimizerType.ADAMW_SCHEDULE_FREE,
            OptimizerType.RADAM_SCHEDULE_FREE,
            OptimizerType.SGD_SCHEDULE_FREE,
        }:
            optimizer = self._create_schedulefree_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer == OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE:
            optimizer = self._create_prodigy_plus_schedule_free_optimizer(trainable_params, optimizer_args)
        elif self.config.optimizer == OptimizerType.PYTORCH_OPTIMIZER:
            from .optimizer_plugin_bridge import create_pytorch_optimizer
            optimizer = create_pytorch_optimizer(
                trainable_params,
                optimizer_name=str(
                    optimizer_args.get("name")
                    or optimizer_args.get("optimizer_name")
                    or optimizer_args.get("optimizer")
                    or ""
                ),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                optimizer_args=optimizer_args,
            )
            if hasattr(optimizer, "train"):
                optimizer.train()
        elif self.config.optimizer == OptimizerType.GENERIC:
            from .optimizer_plugin_bridge import create_generic_optimizer
            optimizer_name = str(
                optimizer_args.get("name")
                or optimizer_args.get("optimizer_name")
                or optimizer_args.get("optimizer")
                or ""
            ).strip()
            if not optimizer_name:
                raise ValueError("GenericOptimizer requires optimizer_args name=<class>")
            self._log(f"GenericOptimizer: resolving '{optimizer_name}'")
            optimizer = create_generic_optimizer(
                trainable_params,
                optimizer_name=optimizer_name,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                optimizer_args=optimizer_args,
            )
            if hasattr(optimizer, "train"):
                optimizer.train()
        elif self.config.optimizer == OptimizerType.AUTOMAGIC_PLUS_PLUS:
            from .automagic_plus_plus_optimizer import AutomagicPlusPlus
            self._log(
                "Automagic++ optimizer activated: Warehouse stable mode "
                "(tri-state sign, multiplicative local LR, FP32 second moment, relative update cap)."
            )
            optimizer = AutomagicPlusPlus(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **optimizer_args,
            )
        elif self.config.optimizer == OptimizerType.AUTO_PRODIGY:
            from .auto_prodigy_optimizer import AutoProdigy
            auto_prodigy_args = self._auto_prodigy_config()
            auto_prodigy_args.update(optimizer_args)
            self._log(
                "AutoProdigy optimizer activated: Warehouse global distance estimate "
                f"with schedule-free averaging and {getattr(self.config, 'auto_prodigy_profile', 'balanced')} profile."
            )
            optimizer = AutoProdigy(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **auto_prodigy_args,
            )
        elif self.config.optimizer == OptimizerType.ADAFACTOR:
            try:
                from transformers.optimization import Adafactor
                optimizer = Adafactor(
                    trainable_params,
                    lr=self.config.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                    warmup_init=False,
                    **optimizer_args,
                )
            except ImportError:
                self._log("transformers.optimization.Adafactor not available, falling back to AdamW")
                optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    **optimizer_args,
                )
        elif self.config.optimizer == OptimizerType.ANIMA_FACTORED_ADAMW:
            from .anima_factored_optimizer import AnimaFactoredAdamW

            optimizer = AnimaFactoredAdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **optimizer_args,
            )
            self._set_optimizer_backend_profile(
                "anima_factored_adamw",
                "anima_factored_adamw",
                optimizer_type=str(getattr(self.config.optimizer, "value", self.config.optimizer)),
                optimizer_class=type(optimizer).__name__,
                notes=[
                    "Experimental full-finetune optimizer: factored second moments for large 2D DiT weights.",
                ],
            )
        elif self.config.optimizer == OptimizerType.MUON:
            from .muon_optimizer import Muon

            muon_args = {
                "momentum": float(getattr(self.config, "muon_momentum", 0.95)),
                "ns_steps": int(getattr(self.config, "muon_ns_steps", 5)),
            }
            muon_args.update(optimizer_args)
            optimizer = Muon(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **muon_args,
            )
            self._log(
                "Muon optimizer activated: Newton-Schulz orthogonalized momentum "
                "on 2D LoRA factors; AdamW fallback on 1D/scalar params."
            )
        if optimizer is not None:
            if bool(getattr(self.config, "mn_lora_enabled", False)):
                from core.training_components.mn_lora.hijacker import wrap_optimizer
                from core.training_components.mn_lora.mn_presets import (
                    select_mnlora_preset,
                    split_mnlora_preset,
                )
                model_type_name = str(getattr(self.config.model_type, "value", self.config.model_type))
                mn_preset_name = str(getattr(self.config, "mn_lora_preset", "") or "")
                preset_configs = split_mnlora_preset(select_mnlora_preset(model_type_name, mn_preset_name))
                gsp_config = dict(preset_configs["gsp_config"])
                gsp_config.update({
                    "k_ratio": float(getattr(self.config, "mn_lora_k_ratio", gsp_config.get("k_ratio", 0.5))),
                    "update_interval": int(getattr(self.config, "mn_lora_update_interval", gsp_config.get("update_interval", 20))),
                    "adaptive_k": bool(getattr(self.config, "mn_lora_adaptive_k", True)),
                    "lazy_update": bool(getattr(self.config, "mn_lora_lazy_update", gsp_config.get("lazy_update", True))),
                    "residual_threshold": float(getattr(self.config, "mn_lora_residual_threshold", gsp_config.get("residual_threshold", 0.3))),
                    "min_k_ratio": float(getattr(self.config, "mn_lora_min_k_ratio", 0.2)),
                    "max_k_ratio": float(getattr(self.config, "mn_lora_max_k_ratio", 0.8)),
                    "lazy_threshold": float(getattr(self.config, "mn_lora_lazy_threshold", gsp_config.get("lazy_threshold", 0.5))),
                    "precondition_mode": str(getattr(self.config, "mn_lora_precondition_mode", "grad_ema") or "grad_ema"),
                    "svd_precond_beta": float(getattr(self.config, "mn_lora_svd_precond_beta", 0.5)),
                    "precond_min_scale": float(getattr(self.config, "mn_lora_precond_min_scale", 0.25)),
                    "precond_max_scale": float(getattr(self.config, "mn_lora_precond_max_scale", 4.0)),
                    "coord_curv_beta": float(getattr(self.config, "mn_lora_coord_curv_beta", 0.95)),
                    "precond_clip": float(getattr(self.config, "mn_lora_precond_clip", 3.0)),
                    "precond_eps": float(getattr(self.config, "mn_lora_precond_eps", 1e-6)),
                    "adaptive_sparse_enabled": bool(getattr(self.config, "mn_lora_adaptive_sparse_enabled", True)),
                    "adaptive_sparse_warmup_steps": int(getattr(self.config, "mn_lora_adaptive_sparse_warmup_steps", 10)),
                    "adaptive_sparse_refresh_interval": int(getattr(self.config, "mn_lora_adaptive_sparse_refresh_interval", 20)),
                    "adaptive_sparse_hot_ratio": float(getattr(self.config, "mn_lora_adaptive_sparse_hot_ratio", 0.20)),
                    "adaptive_sparse_warm_ratio": float(getattr(self.config, "mn_lora_adaptive_sparse_warm_ratio", 0.0)),
                    "adaptive_sparse_warm_interval": int(getattr(self.config, "mn_lora_adaptive_sparse_warm_interval", 4)),
                    "adaptive_sparse_cold_interval": int(getattr(self.config, "mn_lora_adaptive_sparse_cold_interval", 16)),
                    "adaptive_sparse_min_hot_layers": int(getattr(self.config, "mn_lora_adaptive_sparse_min_hot_layers", 16)),
                    "adaptive_sparse_zero_cold_after": int(getattr(self.config, "mn_lora_adaptive_sparse_zero_cold_after", 3)),
                })
                tgwd_config = dict(preset_configs["tgwd_config"])
                tgwd_config.update({
                    "alpha": float(getattr(self.config, "mn_lora_tgwd_alpha", 1.0)),
                    "n_probes": int(getattr(self.config, "mn_lora_tgwd_n_probes", 1)),
                    "probe_interval": int(getattr(self.config, "mn_lora_tgwd_probe_interval", tgwd_config.get("probe_interval", 50))),
                    "finite_diff_eps": float(getattr(self.config, "mn_lora_tgwd_finite_diff_eps", 1e-3)),
                })
                pilot_config = dict(preset_configs["pilot_config"])
                pilot_config.update({
                    "strategy": str(getattr(self.config, "mn_lora_pilot_strategy", pilot_config.get("strategy", "population"))),
                })
                optimizer = wrap_optimizer(
                    optimizer,
                    enable_gsp=bool(getattr(self.config, "mn_lora_gsp_enabled", True)),
                    enable_tgwd=bool(getattr(self.config, "mn_lora_tgwd_enabled", True)),
                    enable_pilot=True,
                    gsp_config=gsp_config,
                    tgwd_config=tgwd_config,
                    pilot_config=pilot_config,
                    plus_plus_config=self._mn_lora_plus_plus_config(),
                    kfac_lite_config=self._mn_lora_kfac_lite_config(),
                    trust_region_config=self._mn_lora_trust_region_config(),
                    effective_delta_config=self._mn_lora_effective_delta_config(),
                    fisher_ewc_config=self._mn_lora_fisher_ewc_config(),
                    gradient_conflict_config=self._mn_lora_gradient_conflict_config(),
                    lora_modules=getattr(self.lora_injector, "injected_layers", {}) if self.lora_injector is not None else {},
                    param_names=self._optimizer_param_names(),
                )
            return optimizer

        elif self.config.optimizer == OptimizerType.LION:
            try:
                from lion_pytorch import Lion
                return Lion(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
            except ImportError:
                self._log("lion_pytorch not available, falling back to AdamW")
                return torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
        elif self.config.optimizer == OptimizerType.LION_8BIT:
            try:
                import bitsandbytes as bnb
                lion8 = getattr(bnb.optim, "Lion8bit", None)
                if lion8 is not None:
                    return lion8(
                        trainable_params,
                        lr=self.config.learning_rate,
                        weight_decay=self.config.weight_decay,
                        **optimizer_args,
                    )
            except ImportError:
                pass

            try:
                from lion_pytorch import Lion
                self._log("Lion8bit unavailable; falling back to lion_pytorch.Lion")
                return Lion(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
            except ImportError:
                self._log("Lion optimizer not available, falling back to AdamW8bit/AdamW")
                try:
                    import bitsandbytes as bnb
                    return bnb.optim.AdamW8bit(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay, **optimizer_args)
                except ImportError:
                    return torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay, **optimizer_args)
        elif self.config.optimizer == OptimizerType.SGD_NESTEROV:
            sgd_args = {"momentum": 0.9, "nesterov": True}
            sgd_args.update(optimizer_args)
            return torch.optim.SGD(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **sgd_args,
            )
        elif self.config.optimizer == OptimizerType.DADAPT_ADAM:
            try:
                from dadaptation import DAdaptAdam
                return DAdaptAdam(
                    trainable_params,
                    lr=1.0,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
            except ImportError:
                self._log("dadaptation not available, falling back to AdamW")
                return torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
        elif self.config.optimizer == OptimizerType.DADAPT_ADAN:
            try:
                from dadaptation import DAdaptAdan
                return DAdaptAdan(
                    trainable_params,
                    lr=1.0,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
            except ImportError:
                self._log("dadaptation not available, falling back to AdamW")
                return torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
        elif self.config.optimizer == OptimizerType.DADAPT_SGD:
            try:
                from dadaptation import DAdaptSGD
                return DAdaptSGD(
                    trainable_params,
                    lr=1.0,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
            except ImportError:
                self._log("dadaptation not available, falling back to AdamW")
                return torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    **optimizer_args,
                )
        else:
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                **optimizer_args,
            )

    def _create_scheduler(self, optimizer, total_steps: int):
        """创建学习率调度器"""
        total_steps = max(int(total_steps), 1)
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        warmup_steps = max(min(warmup_steps, total_steps - 1), 0)
        num_cycles = getattr(self.config, "lr_scheduler_num_cycles", 1)
        scheduler_args = self._filtered_custom_args(
            getattr(self.config, "lr_scheduler_args", ""),
            self._scheduler_allowed_args(),
            "lr_scheduler_args",
        )
        loss_cosine_cycles = float(
            scheduler_args.get("num_cycles", max(float(num_cycles), 1.0) / 2.0)
        )
        loss_scheduler_kwargs = {
            "eta_min": float(scheduler_args.get("eta_min", 0.0)),
            "num_cycles": loss_cosine_cycles,
            "ema_alpha": float(
                scheduler_args.get(
                    "ema_alpha",
                    getattr(self.config, "loss_scheduler_ema_alpha", 0.1),
                )
            ),
            "min_delta": float(
                scheduler_args.get(
                    "min_delta",
                    getattr(self.config, "loss_scheduler_min_delta", 5e-4),
                )
            ),
            "relative_delta": float(
                scheduler_args.get(
                    "relative_delta",
                    getattr(self.config, "loss_scheduler_relative_delta", 1e-3),
                )
            ),
            "patience": int(
                scheduler_args.get(
                    "patience",
                    getattr(self.config, "loss_scheduler_patience", 8),
                )
            ),
            "cooldown": int(
                scheduler_args.get(
                    "cooldown",
                    getattr(self.config, "loss_scheduler_cooldown", 0),
                )
            ),
            "max_hold_steps": int(
                scheduler_args.get(
                    "max_hold_steps",
                    getattr(self.config, "loss_scheduler_max_hold_steps", 0),
                )
            ),
            "late_loss_gamma": float(
                scheduler_args.get(
                    "late_loss_gamma",
                    getattr(self.config, "loss_scheduler_late_gamma", 2.0),
                )
            ),
            "lock_weight_threshold": float(
                scheduler_args.get(
                    "lock_weight_threshold",
                    getattr(self.config, "loss_scheduler_lock_weight_threshold", 0.7),
                )
            ),
            "min_advance_ratio": float(
                scheduler_args.get(
                    "min_advance_ratio",
                    getattr(self.config, "loss_scheduler_min_advance_ratio", 0.25),
                )
            ),
            "hold_on_improvement": bool(scheduler_args.get("hold_on_improvement", True)),
        }

        if self.config.optimizer in {OptimizerType.AUTOMAGIC_PLUS_PLUS, OptimizerType.AUTO_PRODIGY} or self._is_schedule_free_optimizer():
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(optimizer, factor=1.0)
        if self.config.optimizer == OptimizerType.PYTORCH_OPTIMIZER:
            from .optimizer_plugin_bridge import is_schedulefree_like
            if is_schedulefree_like(optimizer):
                from torch.optim.lr_scheduler import ConstantLR
                return ConstantLR(optimizer, factor=1.0)
        if self.config.optimizer == OptimizerType.GENERIC:
            from .optimizer_plugin_bridge import is_schedulefree_like
            if is_schedulefree_like(optimizer):
                from torch.optim.lr_scheduler import ConstantLR
                return ConstantLR(optimizer, factor=1.0)

        if self.config.scheduler == SchedulerType.COSINE:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=float(scheduler_args.get("eta_min", 0.0)),
            )
        elif self.config.scheduler == SchedulerType.LOSS_GATED_COSINE:
            from .loss_aware_scheduler import LossAwareCosineScheduler
            return LossAwareCosineScheduler.gated(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                **loss_scheduler_kwargs,
            )
        elif self.config.scheduler == SchedulerType.LOSS_WEIGHTED_ANNEALED_COSINE:
            from .loss_aware_scheduler import LossAwareCosineScheduler
            return LossAwareCosineScheduler.weighted(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                **loss_scheduler_kwargs,
            )
        elif self.config.scheduler == SchedulerType.COSINE_RESTARTS:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            t_0 = int(scheduler_args.get("t_0", max(int(total_steps / max(num_cycles, 1)), 1)))
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(t_0, 1),
                T_mult=max(int(scheduler_args.get("t_mult", 1)), 1),
                eta_min=float(scheduler_args.get("eta_min", 0.0)),
            )
        elif self.config.scheduler == SchedulerType.COSINE_WITH_MIN_LR:
            from torch.optim.lr_scheduler import LambdaLR
            import math
            min_lr_ratio = float(scheduler_args.get("min_lr_ratio", scheduler_args.get("min_lr_rate", 0.0)))
            cycles = float(scheduler_args.get("num_cycles", max(float(num_cycles), 1.0))) / 2.0
            active_steps = max(total_steps - warmup_steps, 1)

            def _cosine_with_min_lr(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return max(float(step + 1) / float(warmup_steps), min_lr_ratio)
                progress = min(max(float(step - warmup_steps) / float(active_steps), 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            return LambdaLR(optimizer, lr_lambda=_cosine_with_min_lr)
        elif self.config.scheduler == SchedulerType.LINEAR:
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                optimizer,
                start_factor=float(scheduler_args.get("start_factor", 1.0)),
                end_factor=float(scheduler_args.get("end_factor", 0.0)),
                total_iters=max(int(scheduler_args.get("total_iters", total_steps)), 1),
            )
        elif self.config.scheduler == SchedulerType.CONSTANT:
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(optimizer, factor=1.0)
        elif self.config.scheduler == SchedulerType.CONSTANT_WARMUP:
            from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
            if warmup_steps > 0:
                warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
                constant_sched = ConstantLR(optimizer, factor=1.0)
                return SequentialLR(optimizer, schedulers=[warmup_sched, constant_sched], milestones=[warmup_steps])
            return ConstantLR(optimizer, factor=1.0)
        elif self.config.scheduler == SchedulerType.POLYNOMIAL:
            from torch.optim.lr_scheduler import PolynomialLR
            return PolynomialLR(
                optimizer,
                total_iters=max(int(scheduler_args.get("total_iters", total_steps)), 1),
                power=float(scheduler_args.get("power", 1.0)),
            )
        elif self.config.scheduler == SchedulerType.PIECEWISE_CONSTANT:
            from torch.optim.lr_scheduler import LambdaLR
            raw_rules = str(scheduler_args.get("rules", scheduler_args.get("schedule", "")) or "").strip()
            boundaries: list[tuple[int, float]] = []
            for chunk in raw_rules.replace(";", ",").split(","):
                if not chunk.strip() or ":" not in chunk:
                    continue
                left, right = chunk.split(":", 1)
                try:
                    boundaries.append((max(int(left.strip()), 0), float(right.strip())))
                except ValueError:
                    continue
            boundaries.sort(key=lambda item: item[0])

            def _piecewise(step):
                factor = 1.0
                for boundary, value in boundaries:
                    if step >= boundary:
                        factor = value
                    else:
                        break
                return factor

            return LambdaLR(optimizer, lr_lambda=_piecewise)
        elif self.config.scheduler == SchedulerType.TSD:
            # Warmup Stable Decay: warmup -> constant plateau -> cosine decay
            from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR
            stable_steps = max(int(total_steps * 0.6), 1)
            decay_steps = max(total_steps - warmup_steps - stable_steps, 1)
            schedulers = [
                ConstantLR(optimizer, factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=stable_steps),
                CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=0),
            ]
            milestones = [warmup_steps, warmup_steps + stable_steps]
            return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        elif self.config.scheduler == SchedulerType.ONE_CYCLE:
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                optimizer,
                max_lr=float(scheduler_args.get("max_lr", self.config.learning_rate)),
                total_steps=total_steps,
                pct_start=float(scheduler_args.get("pct_start", 0.3)),
                anneal_strategy=scheduler_args.get("anneal_strategy", "cos"),
                div_factor=float(scheduler_args.get("div_factor", 25.0)),
                final_div_factor=float(scheduler_args.get("final_div_factor", 1e4)),
                three_phase=bool(scheduler_args.get("three_phase", False)),
            )
        elif self.config.scheduler == SchedulerType.INVERSE_SQRT:
            from torch.optim.lr_scheduler import LambdaLR
            import math
            warmup_steps_inv = max(warmup_steps, 1)
            def _inverse_sqrt(step):
                if step < warmup_steps_inv:
                    return (step + 1) / warmup_steps_inv
                return math.sqrt(warmup_steps_inv) / math.sqrt(step + 1)
            return LambdaLR(optimizer, lr_lambda=_inverse_sqrt)
        elif self.config.scheduler == SchedulerType.ADAFACTOR:
            # Adafactor uses its own internal LR schedule; wrap in constant so the
            # trainer loop can still call .step() without error.
            from torch.optim.lr_scheduler import ConstantLR
            return ConstantLR(optimizer, factor=1.0)
        elif self.config.scheduler == SchedulerType.RESTART_LINEAR:
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, ConstantLR
            if warmup_steps > 0:
                warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
            else:
                warmup_sched = ConstantLR(optimizer, factor=1.0, total_iters=1)
            t_0 = int(scheduler_args.get("t_0", max(int(total_steps / max(num_cycles, 1)), 1)))
            decay_sched = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=float(scheduler_args.get("eta_min", 0.0)),
                total_iters=max(t_0, 1),
            )
            return SequentialLR(optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps])
        else:
            # 默认 Cosine
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))


__all__ = ["TrainerOptimizerFactoryMixin"]

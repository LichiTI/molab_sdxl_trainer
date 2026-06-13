# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Artifact I/O cluster extracted verbatim from ``trainer.py`` as a mixin.

These methods (retention pruning, adapter/semantic-tuner state-dict assembly,
state-dict file read/write, model save / merged-checkpoint export / optimizer
state save+load) run as bound methods of the trainer instance — identical
``self`` semantics, identical call sites. Behaviour is unchanged; this split
only keeps ``trainer.py`` navigable.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..constants import (
    EXT_PT,
    EXT_SAFETENSORS,
    FILENAME_MODEL_TEMPLATE,
    FILENAME_STATE_TEMPLATE,
)
from .anima_train_norm_compat import export_anima_train_norm_state_dict
from .lokr_export_rules import export_lokr_state_dict
from .sdxl_lora_key_export import export_sdxl_compatible_lora_keys
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


class TrainerArtifactIoMixin:
    def _parse_saved_artifact(self, path: Path, *, state: bool) -> Optional[tuple[str, int]]:
        output_name = str(getattr(self.config, "output_name", "") or "")
        if not output_name:
            return None

        if state:
            step_match = re.match(
                rf"^{re.escape(output_name)}-step(\d+)-state{re.escape(EXT_PT)}$",
                path.name,
            )
            if step_match:
                return ("step", int(step_match.group(1)))

            epoch_match = re.match(
                rf"^{re.escape(output_name)}-(\d+)-state{re.escape(EXT_PT)}$",
                path.name,
            )
            if epoch_match:
                return ("epoch", int(epoch_match.group(1)))
            return None

        ext = re.escape(self._get_model_save_extension())
        step_match = re.match(rf"^{re.escape(output_name)}-step(\d+){ext}$", path.name)
        if step_match:
            return ("step", int(step_match.group(1)))

        epoch_match = re.match(rf"^{re.escape(output_name)}-(\d+){ext}$", path.name)
        if epoch_match:
            return ("epoch", int(epoch_match.group(1)))
        return None

    def _retention_step_window(self, *, state: bool) -> int:
        if state:
            override = max(int(getattr(self.config, "save_last_n_steps_state", 0) or 0), 0)
            if override > 0:
                return override
        return max(int(getattr(self.config, "save_last_n_steps", 0) or 0), 0)

    def _retention_epoch_count(self, *, state: bool) -> int:
        if state:
            override = max(int(getattr(self.config, "save_last_n_epochs_state", 0) or 0), 0)
            if override > 0:
                return override
        direct = max(int(getattr(self.config, "save_last_n_epochs", 0) or 0), 0)
        if direct > 0:
            return direct
        return max(int(getattr(self.config, "checkpoint_keep_last", 0) or 0), 0)

    def _retention_fallback_count(self) -> int:
        return max(int(getattr(self.config, "checkpoint_keep_last", 0) or 0), 0)

    def _prune_saved_artifacts(self, *, state: bool, mode: str, current_value: int) -> None:
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_dir():
            return

        parsed: List[tuple[Path, int]] = []
        for path in output_dir.iterdir():
            if not path.is_file():
                continue
            artifact = self._parse_saved_artifact(path, state=state)
            if artifact is None:
                continue
            artifact_mode, artifact_value = artifact
            if artifact_mode == mode:
                parsed.append((path, artifact_value))

        if not parsed:
            return

        stale: List[Path] = []
        if mode == "step":
            keep_window = self._retention_step_window(state=state)
            if keep_window > 0:
                threshold = current_value - keep_window
                stale = [path for path, value in parsed if value <= threshold]
            else:
                keep_last = self._retention_fallback_count()
                if keep_last > 0 and len(parsed) > keep_last:
                    parsed.sort(key=lambda item: item[1])
                    stale = [path for path, _ in parsed[:-keep_last]]
        else:
            keep_last = self._retention_epoch_count(state=state)
            if keep_last > 0 and len(parsed) > keep_last:
                parsed.sort(key=lambda item: item[1])
                stale = [path for path, _ in parsed[:-keep_last]]

        for stale_path in stale:
            try:
                stale_path.unlink()
                kind = "state" if state else "checkpoint"
                self._log(f"Pruned old {kind}: {stale_path.name}")
            except Exception as e:
                logger.warning(f"Failed to prune saved artifact {stale_path}: {e}")
    def _build_semantic_tuner_state_dict(self) -> Dict[str, torch.Tensor]:
        save_dict: Dict[str, torch.Tensor] = {}
        ctx = self.te_manager.get_semantic_context() if self.te_manager else None

        if hasattr(self, "sidecar_net"):
            for name, param in self.sidecar_net.state_dict().items():
                save_dict[f"sidecar.{name}"] = param

        if ctx and ctx.get("projector"):
            for name, param in ctx["projector"].state_dict().items():
                save_dict[f"projector.{name}"] = param

        return save_dict

    def _load_semantic_tuner_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]

        sidecar_state = {
            key[len("sidecar."):]: value
            for key, value in state_dict.items()
            if key.startswith("sidecar.")
        }
        projector_state = {
            key[len("projector."):]: value
            for key, value in state_dict.items()
            if key.startswith("projector.")
        }

        if sidecar_state and hasattr(self, "sidecar_net"):
            self.sidecar_net.load_state_dict(sidecar_state, strict=False)

        ctx = self.te_manager.get_semantic_context() if self.te_manager else None
        projector = ctx.get("projector") if ctx else None
        if projector_state and projector is not None:
            projector.load_state_dict(projector_state, strict=False)

    def _get_current_adapter_state_dict(self, use_ema: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        if getattr(self.config, "semantic_tuner_enabled", False):
            return self._build_semantic_tuner_state_dict()

        if use_ema and self._ema_tracker:
            return self._ema_tracker.get_ema_state_dict()

        # Textual Inversion: save the learned concept embedding only
        if getattr(self, "_ti_trainer", None) is not None:
            return self._ti_trainer.concept_embedding.state_dict()

        state_dict: Optional[Dict[str, torch.Tensor]] = None
        if self.lora_injector and hasattr(self.lora_injector, "get_lora_state_dict"):
            state_dict = self.lora_injector.get_lora_state_dict()

        if self._easy_control is not None:
            if state_dict is None:
                state_dict = {}
            for key, value in self._easy_control.state_dict().items():
                state_dict[f"easy_control.{key}"] = value

        if self._ip_adapter is not None:
            if state_dict is None:
                state_dict = {}
            for key, value in self._ip_adapter.projector.state_dict().items():
                state_dict[f"ip_adapter.projector.{key}"] = value

        if self._easycontrol_v2_adapter is not None:
            if state_dict is None:
                state_dict = {}
            for key, value in self._easycontrol_v2_adapter.state_dict().items():
                state_dict[f"easycontrol_v2.{key}"] = value

        if bool(getattr(self.config, "reft_enabled", False)) and self.model is not None:
            from .reft import get_reft_state_dict
            reft_state = get_reft_state_dict(self.model.unet)
            if reft_state:
                if state_dict is None:
                    state_dict = {}
                state_dict.update(reft_state)

        if self._repa_projector is not None:
            if state_dict is None:
                state_dict = {}
            for key, value in self._repa_projector.state_dict().items():
                state_dict[f"repa_projector.{key}"] = value

        # Include prefix/postfix tuning parameters (#113)
        prefix_length = int(getattr(self.config, "prefix_tuning_length", 0) or 0)
        postfix_length = int(getattr(self.config, "postfix_tuning_length", 0) or 0)
        if (prefix_length > 0 or postfix_length > 0) and self.model is not None:
            from .prefix_tuning import get_prefix_tuning_state_dict
            sp_state = get_prefix_tuning_state_dict(self.model)
            if sp_state:
                if state_dict is None:
                    state_dict = {}
                state_dict.update(sp_state)

        return state_dict

    def _prepare_adapter_init_export_for_save(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]],
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        strategy = str(getattr(self.config, "adapter_init_strategy", "default") or "default").strip().lower().replace("-", "_")
        if strategy not in {"pissa", "olora", "loftq"} or not self.lora_injector:
            return state_dict, metadata

        export_mode = str(getattr(self.config, "adapter_init_export_mode", "auto") or "auto").strip().lower().replace("-", "_")
        if export_mode in {"", "none", "off", "native", "training"}:
            export_mode = "raw"
        if strategy == "pissa" and export_mode == "auto":
            export_mode = str(getattr(self.config, "pissa_export_mode", "lora_compatible") or "lora_compatible").strip().lower().replace("-", "_")
        elif export_mode == "auto":
            export_mode = "lora_compatible"
        aliases = {
            "compatible": "lora_compatible",
            "standard": "lora_compatible",
            "standard_lora": "lora_compatible",
            "lora_compatible_export": "lora_compatible",
            "fast": "approximate",
            "quick": "approximate",
        }
        export_mode = aliases.get(export_mode.replace(" ", "_"), export_mode)
        if export_mode not in {"raw", "lora_compatible", "approximate"}:
            export_mode = "raw"
        if export_mode == "raw" or not hasattr(self.lora_injector, "export_adapter_init_state_dict"):
            return state_dict, metadata

        if metadata is None:
            metadata = {}
        metadata["ss_adapter_init_strategy"] = strategy
        metadata["ss_adapter_init_export_mode"] = export_mode
        state_dict = self.lora_injector.export_adapter_init_state_dict(state_dict, export_mode)
        if export_mode == "lora_compatible":
            base_rank = int(getattr(self.config, "network_dim", 0) or 0)
            base_alpha = float(getattr(self.config, "network_alpha", 0.0) or 0.0)
            if base_rank > 0:
                metadata["ss_network_dim"] = str(base_rank * 2)
            if base_alpha > 0:
                doubled_alpha = base_alpha * 2
                metadata["ss_network_alpha"] = str(int(doubled_alpha) if doubled_alpha.is_integer() else doubled_alpha)
        return state_dict, metadata

    def _prepare_anima_lokr_export_for_save(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]],
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        """Delegate LoKr export shaping to the Warehouse compatibility rules."""
        model_arch = getattr(getattr(self.config, "model_type", ""), "value", getattr(self.config, "model_type", ""))
        if str(model_arch).lower() != "anima":
            return state_dict, metadata

        network_module = getattr(getattr(self.config, "network_module", ""), "value", getattr(self.config, "network_module", ""))
        lycoris_algo = getattr(getattr(self.config, "lycoris_algo", ""), "value", getattr(self.config, "lycoris_algo", ""))
        if str(network_module) != "lycoris.locon" or str(lycoris_algo).lower() != "lokr":
            return state_dict, metadata

        export_mode = str(getattr(self.config, "lokr_export_mode", "native") or "native").strip().lower()
        state_dict, metadata = export_lokr_state_dict(state_dict, metadata, export_mode=export_mode)
        if getattr(self, "lora_injector", None) is not None and bool(getattr(self.config, "lokr_train_norm", False) or getattr(self.config, "lycoris_train_norm", False)):
            state_dict, metadata = export_anima_train_norm_state_dict(
                state_dict,
                metadata,
                injector=self.lora_injector,
            )
        return state_dict, metadata

    def _prepare_thin_svd_export_for_save(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]],
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        """Apply Thin-SVD compression to LoRA weights before saving.

        Thin-SVD reduces inference memory/compute by compressing LoRA ranks while
        preserving the learned delta. This is applied at export time only and does
        not affect training.
        """
        if not getattr(self.config, "thin_svd_export_enabled", False):
            return state_dict, metadata

        target_rank = int(getattr(self.config, "thin_svd_export_rank", 0))
        if target_rank <= 0:
            self._log("thin_svd_export_enabled=True but thin_svd_export_rank <= 0, skipping Thin-SVD export")
            return state_dict, metadata

        original_rank = int(getattr(self.config, "network_dim", 0))
        if target_rank >= original_rank:
            self._log(f"Thin-SVD target rank {target_rank} >= original rank {original_rank}, skipping compression")
            return state_dict, metadata

        self._log(f"Applying Thin-SVD export compression: {original_rank} -> {target_rank}")

        compressed_state = {}
        compression_count = 0

        # Group LoRA pairs by base key
        lora_pairs = {}
        for key in state_dict.keys():
            if ".lora_down." in key:
                base_key = key.replace(".lora_down.", ".")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["down"] = key
            elif ".lora_up." in key:
                base_key = key.replace(".lora_up.", ".")
                if base_key not in lora_pairs:
                    lora_pairs[base_key] = {}
                lora_pairs[base_key]["up"] = key

        # Process each LoRA pair
        for base_key, pair in lora_pairs.items():
            if "down" in pair and "up" in pair:
                down_key = pair["down"]
                up_key = pair["up"]

                lora_down = state_dict[down_key]
                lora_up = state_dict[up_key]

                # Reconstruct full delta: W = B @ A (where B=lora_up, A=lora_down)
                reconstructed = lora_up @ lora_down

                # SVD decomposition
                try:
                    U, S, Vh = torch.linalg.svd(reconstructed, full_matrices=False)
                except Exception as e:
                    self._log(f"Thin-SVD failed for {base_key}: {e}, keeping original")
                    compressed_state[down_key] = lora_down
                    compressed_state[up_key] = lora_up
                    continue

                # Truncate to target rank
                U_thin = U[:, :target_rank]
                S_thin = S[:target_rank]
                Vh_thin = Vh[:target_rank, :]

                # Distribute singular values via sqrt (balanced factorization)
                S_sqrt = torch.diag(S_thin.sqrt())
                compressed_down = (S_sqrt @ Vh_thin).contiguous()
                compressed_up = (U_thin @ S_sqrt).contiguous()

                compressed_state[down_key] = compressed_down
                compressed_state[up_key] = compressed_up
                compression_count += 1

        # Copy non-LoRA weights
        for key, value in state_dict.items():
            if key not in compressed_state:
                compressed_state[key] = value

        # Update metadata
        if metadata is None:
            metadata = {}
        metadata = metadata.copy()
        metadata["ss_thin_svd_export"] = "true"
        metadata["ss_thin_svd_rank"] = str(target_rank)
        metadata["ss_thin_svd_original_rank"] = str(original_rank)
        metadata["ss_thin_svd_compressed_layers"] = str(compression_count)

        self._log(f"Thin-SVD compression applied to {compression_count} LoRA layer pairs")
        return compressed_state, metadata

    def _prepare_sdxl_compatible_lora_export_for_save(
        self,
        state_dict: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, str]],
    ) -> tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        """把 SDXL 标准 LoRA 的“裸 key”转成 kohya/ComfyUI 兼容格式。

        molab native trainer 注入时用 prefix=unet/te1/te2，导出 key 形如
        ``unet_..._.lora_down.weight`` / ``te1_encoder_layers_..._.lora_up.weight``，
        缺少 sd-scripts 标准的 ``lora_`` 前缀与 ``text_model_`` 段，且无 per-module
        ``.alpha``、无 ``ss_network_module``。通用 ComfyUI / A1111 加载器按 ``lora_unet_`` /
        ``lora_te1_text_model_`` 前缀正则匹配，匹配不到就整份跳过，导致出图无效。
        这里在保存阶段补齐前缀 / alpha / 元数据，使产物可被通用工具加载。

        仅对 SDXL 架构生效；anima 等走各自的导出规则，不受影响。
        """
        model_arch = getattr(getattr(self.config, "model_type", ""), "value", getattr(self.config, "model_type", ""))
        if str(model_arch).lower() != "sdxl":
            return state_dict, metadata

        try:
            network_alpha = float(getattr(self.config, "network_alpha", 0.0) or 0.0)
        except (TypeError, ValueError):
            network_alpha = 0.0
        if network_alpha <= 0.0 and metadata:
            try:
                network_alpha = float(metadata.get("ss_network_alpha", 0.0) or 0.0)
            except (TypeError, ValueError):
                network_alpha = 0.0
        if network_alpha <= 0.0:
            network_alpha = 1.0

        new_state, new_meta, converted = export_sdxl_compatible_lora_keys(
            state_dict, metadata, network_alpha=network_alpha
        )
        if converted:
            self._log(
                f"SDXL LoRA key export: normalized {converted} modules to kohya/ComfyUI "
                f"format (lora_unet_/lora_te1_text_model_ prefixes + alpha)"
            )
        return new_state, new_meta

    def _save_state_dict_to_path(
        self,
        state_dict: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict[str, str]] = None,
    ):
        mem_efficient = bool(getattr(self.config, "mem_efficient_save", False))
        payload = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                t = value.detach()
                if mem_efficient:
                    # Move to CPU one at a time to avoid peak VRAM spike
                    t = t.cpu()
                payload[key] = t
            else:
                payload[key] = value

        if mem_efficient:
            # Free GPU memory before disk I/O
            self._maybe_release_tool_cuda_cache("mem_efficient_state_dict_save")

        if save_path.suffix.lower() == EXT_SAFETENSORS:
            try:
                from safetensors.torch import save_file
                save_file(payload, str(save_path), metadata=metadata)
                return
            except ImportError:
                self._log("safetensors not available, falling back to torch.save")

        torch.save({"state_dict": payload, "metadata": metadata or {}}, save_path)

    def _load_state_dict_from_path(self, path: Path) -> Dict[str, torch.Tensor]:
        if path.suffix.lower() == EXT_SAFETENSORS:
            from .safetensors_loader import load_safetensors

            disable_mmap = bool(getattr(self.config, "disable_mmap_load_safetensors", False))
            return load_safetensors(str(path), device="cpu", disable_mmap=disable_mmap)

        loaded = safe_torch_load(str(path), map_location="cpu")
        if isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict):
            return loaded["state_dict"]
        if isinstance(loaded, dict):
            return loaded
        raise TypeError(f"Unsupported state-dict payload in {path}")
    def _save_model(self, epoch: int, final: bool = False, step: Optional[int] = None):
        """保存模型 — only saves on rank 0 when DDP is active."""
        from .distributed import is_main_process
        if not is_main_process():
            return
        # Support save_to: alternate save directory
        save_dir = str(getattr(self.config, "save_to", "") or "").strip()
        output_dir = Path(save_dir) if save_dir else Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_ext = self._get_model_save_extension()

        if final:
            filename = f"{self.config.output_name}{model_ext}"
        elif step is not None:
            filename = f"{self.config.output_name}-step{step:06d}{model_ext}"
        else:
            filename = f"{FILENAME_MODEL_TEMPLATE.format(output_name=self.config.output_name, epoch=epoch)}{model_ext}"

        save_path = output_dir / filename

        if getattr(self, "_ti_trainer", None) is not None:
            # Textual Inversion needs token metadata alongside the tensor.
            # The dedicated TI trainer writes {placeholder_token, num_vectors, embeddings}.
            ti_path = save_path.with_suffix(EXT_PT)
            self._ti_trainer.save(str(ti_path))
            self._log(f"Textual Inversion embedding saved: {ti_path}")
            if not final and bool(getattr(self.config, "save_state", False)):
                self._save_state(epoch, step=step)
            if not final:
                self._prune_saved_artifacts(
                    state=False,
                    mode="step" if step is not None else "epoch",
                    current_value=int(step if step is not None else epoch),
                )
            return

        metadata = None
        if not bool(getattr(self.config, "no_metadata", False)):
            comment = str(getattr(self.config, "training_comment", "") or "").strip()
            metadata = {
                "ss_base_model_version": str(getattr(self.config.model_arch, "value", self.config.model_arch)),
                "ss_output_name": str(self.config.output_name),
                "ss_network_dim": str(self.config.network_dim),
                "ss_network_alpha": str(self.config.network_alpha),
                "ss_learning_rate": str(self.config.learning_rate),
                "ss_training_comment": comment or "Trained with Lulynx Native Trainer",
            }
            if step is not None:
                metadata["ss_training_step"] = str(step)
            if bool(getattr(self.config, "rs_lora_enabled", False)):
                metadata["ss_rs_lora"] = "true"
                metadata["ss_scaling_strategy"] = "alpha_over_sqrt_rank"

        optimizer_for_save = getattr(getattr(self, "training_loop", None), "optimizer", None)
        optimizer_eval_swapped = False
        if optimizer_for_save is not None and hasattr(optimizer_for_save, "eval") and hasattr(optimizer_for_save, "train"):
            try:
                optimizer_for_save.eval()
                optimizer_eval_swapped = True
            except Exception:
                optimizer_eval_swapped = False
        try:
            if getattr(self.config, "semantic_tuner_enabled", False):
                # Prepare metadata for V3.1 Loader
                ctx = self.te_manager.get_semantic_context()
                mode = ctx.get("mode", "hybrid") if ctx else "hybrid"
                if metadata is not None:
                    metadata["architecture_mode"] = mode
                    metadata["neuro_link_version"] = "v3.1.0"
                self._save_state_dict_to_path(self._build_semantic_tuner_state_dict(), save_path, metadata)
            else:
                adapter_state = self._get_current_adapter_state_dict(use_ema=bool(self._ema_tracker))
                if not adapter_state:
                    raise RuntimeError("No adapter state available to save.")
                adapter_state, metadata = self._prepare_adapter_init_export_for_save(adapter_state, metadata)
                adapter_state, metadata = self._prepare_anima_lokr_export_for_save(adapter_state, metadata)
                adapter_state, metadata = self._prepare_thin_svd_export_for_save(adapter_state, metadata)
                adapter_state, metadata = self._prepare_sdxl_compatible_lora_export_for_save(adapter_state, metadata)
                self._save_state_dict_to_path(adapter_state, save_path, metadata)
        finally:
            if optimizer_eval_swapped:
                try:
                    optimizer_for_save.train()
                except Exception:
                    pass

        self._log(f"Model saved: {save_path}")
        if final:
            try:
                mn = self._run_manifest_extra().get("mn_lora", {})
                gsp = mn.get("gsp", {}) if isinstance(mn, dict) else {}
                if gsp:
                    self._log(
                        "MN-LoRA telemetry: "
                        f"mode={gsp.get('mode', '')}, "
                        f"layers={gsp.get('layer_stats_count', 0)}, "
                        f"avg_residual={float(gsp.get('avg_residual_ratio', 0.0) or 0.0):.4f}, "
                        f"precond_ratio={float(gsp.get('precondition_norm_ratio_avg', 1.0) or 1.0):.4f}, "
                        f"clip_rate={float(gsp.get('precondition_clip_rate', 0.0) or 0.0):.4f}"
                    )
            except Exception:
                pass
        self._emit_runtime_event(
            {
                "event_type": "checkpoint",
                "step": int(step or 0),
                "epoch": int(epoch or 0),
                "severity": "info",
                "summary": f"saved: {save_path}",
                "data": {"path": str(save_path), "final": bool(final)},
            }
        )
        self._write_run_manifest(
            "checkpoint_saved" if not final else "final_checkpoint_saved",
            epoch=int(epoch or 0),
            checkpoint_path=str(save_path),
        )
        if not final:
            self._prune_saved_artifacts(
                state=False,
                mode="step" if step is not None else "epoch",
                current_value=int(step if step is not None else epoch),
            )

        if getattr(self.config, "save_state_to_huggingface", False):
            self._log("save_state_to_huggingface is requested but not implemented in the native trainer yet; skipping upload.")

        # ── Merged Checkpoint Export (#121) ──
        # When anima_merge_export or merge_export is enabled, produce a full
        # merged checkpoint (base + adapter weights baked in) alongside the
        # normal adapter-only save above.
        if bool(getattr(self.config, "anima_merge_export", False) or getattr(self.config, "merge_export", False)):
            self._export_merged_checkpoint(save_dir=output_dir, epoch=epoch, step=step, final=final)

        # Save State (Optimizer/Scheduler)
        if not final and bool(getattr(self.config, "save_state", False)):
            self._save_state(epoch, step=step)

    # ------------------------------------------------------------------
    # Merged Checkpoint Export (#121)
    # ------------------------------------------------------------------

    def _export_merged_checkpoint(
        self,
        save_dir: Path,
        epoch: int,
        step: Optional[int] = None,
        final: bool = False,
    ):
        """Export a full merged checkpoint (base weights + adapter deltas).

        Uses a deep-copy so the training model is not mutated.
        """
        from .distributed import is_main_process
        if not is_main_process():
            return

        from .merge_export import export_merged_model

        self._log("Exporting merged checkpoint (base + adapter) …")

        # Determine which sub-model to merge into.
        # For most archs the unet/transformer is the primary target.
        target = getattr(self.model, "unet", self.model) if self.model is not None else None
        if target is None:
            self._log("merge_export: no target model available, skipping")
            return

        # Build output path
        model_ext = self._get_model_save_extension()
        if final:
            merged_filename = f"{self.config.output_name}-merged{model_ext}"
        elif step is not None:
            merged_filename = f"{self.config.output_name}-merged-step{step:06d}{model_ext}"
        else:
            merged_filename = f"{FILENAME_MODEL_TEMPLATE.format(output_name=self.config.output_name, epoch=epoch)}-merged{model_ext}"

        merged_path = save_dir / merged_filename

        # Determine which injector to use
        lora_injector = None
        lycoris_injector = None
        from .lycoris_layers import LyCORISInjector
        if isinstance(self.lora_injector, LyCORISInjector):
            lycoris_injector = self.lora_injector
        elif self.lora_injector is not None and hasattr(self.lora_injector, "injected_layers"):
            lora_injector = self.lora_injector

        save_precision = str(getattr(self.config, "save_precision", "bf16"))

        try:
            result_path = export_merged_model(
                model=target,
                output_path=str(merged_path),
                save_precision=save_precision,
                lora_injector=lora_injector,
                lycoris_injector=lycoris_injector,
            )
            self._log(f"Merged checkpoint exported: {result_path}")
        except Exception as e:
            self._log(f"Merged checkpoint export failed: {e}")

    def _save_state(self, epoch: int, step: Optional[int] = None, final: bool = False):
        """保存训练状态 (Optimizer, Scheduler, RNG) — only saves on rank 0 when DDP is active."""
        from .distributed import is_main_process
        if not is_main_process():
            return
        sync_report = self._sync_turbocore_native_update_state("before_state_save")
        if self._turbocore_native_update_state_sync_failed(sync_report):
            self._log("Skipped state save: TurboCore native optimizer-state sync failed before state save.")
            return
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            state_filename = f"{self.config.output_name}-last-state{EXT_PT}"
        elif step is not None:
            state_filename = f"{self.config.output_name}-step{int(step):06d}-state{EXT_PT}"
        else:
            # Naming convention: output_name-000001-state.pt
            state_filename = f"{FILENAME_STATE_TEMPLATE.format(output_name=self.config.output_name, epoch=epoch)}{EXT_PT}"
        state_path = output_dir / state_filename

        state = {
            "epoch": epoch,
            "global_step": self.training_loop.global_step if self.training_loop else 0,
            "optimizer_state_dict": self.training_loop.optimizer.state_dict() if self.training_loop and self.training_loop.optimizer else {},
            "scheduler_state_dict": self.training_loop.lr_scheduler.state_dict() if self.training_loop and self.training_loop.lr_scheduler else {},
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "ema_state_dict": self._ema_tracker.state_dict() if self._ema_tracker else None,
            "turbocore_update_state": self.training_loop.get_turbocore_update_checkpoint_state()
            if self.training_loop and hasattr(self.training_loop, "get_turbocore_update_checkpoint_state")
            else None,
            "resource_manager_state": {
                "current_batch_size": self._resource_manager.current_batch_size,
                "current_accumulation": self._resource_manager.current_accumulation,
            } if self._resource_manager else None,
        }

        try:
            torch.save(state, state_path)
            self._log(f"State saved: {state_path}")
            self._write_run_manifest(
                "state_saved" if not final else "final_state_saved",
                epoch=int(epoch or 0),
                state_path=str(state_path),
            )
            if not final:
                self._prune_saved_artifacts(
                    state=True,
                    mode="step" if step is not None else "epoch",
                    current_value=int(step if step is not None else epoch),
                )
        except Exception as e:
            self._log(f"Failed to save state: {e}")

    def _load_state(self, resume_path: str) -> Optional[Dict]:
        """加载训练状态"""
        path = Path(resume_path)
        state_path = None

        if path.is_file():
            # Heuristic to find state file
            stem = path.stem # e.g. "my_lora-000001"
            parent = path.parent

            candidates = []

            # Pattern 0: final "last state" artifact
            candidates.append(parent / f"{self.config.output_name}-last-state.pt")

            # Pattern 1: append "-state.pt" (Matches new convention: my_lora-000001-state.pt)
            candidates.append(parent / (stem + "-state.pt"))

            # Pattern 1b: step save companion (my_lora-step000123.safetensors -> my_lora-step000123-state.pt)
            candidates.append(parent / (stem + "-state.pt"))

            # Pattern 2: Legacy _state_epochXX (Backward compatibility)
            if "_epoch" in stem:
                 candidates.append(parent / (stem.replace("_epoch", "_state_epoch") + ".pt"))

            # Pattern 3: direct match, but only for explicit state files
            if path.suffix == ".pt" and "state" in stem:
                 candidates.append(path)

            for c in candidates:
                if c.exists():
                    state_path = c
                    break

        if not state_path or not state_path.exists():
            self._log(f"No state file found for resume path: {resume_path}. Resuming weights only.")
            return None

        self._log(f"Loading state from {state_path}...")
        try:
            # Trainer state contains RNG tuples and numpy state, not only tensors.
            state = safe_torch_load(state_path, map_location=self.device, weights_only=False)
            if "rng_state" in state:
                rng_state = state["rng_state"]
                if isinstance(rng_state, torch.Tensor):
                    rng_state = rng_state.cpu()
                torch.set_rng_state(rng_state)
            if "cuda_rng_state" in state and torch.cuda.is_available():
                cuda_rng_state = state["cuda_rng_state"]
                if isinstance(cuda_rng_state, torch.Tensor):
                    cuda_rng_state = cuda_rng_state.cpu()
                torch.cuda.set_rng_state(cuda_rng_state)
            if "python_rng_state" in state:
                random.setstate(state["python_rng_state"])
            if "numpy_rng_state" in state:
                np.random.set_state(state["numpy_rng_state"])
            return state
        except Exception as e:
            self._log(f"Failed to load state: {e}")
            return None


__all__ = ["TrainerArtifactIoMixin"]

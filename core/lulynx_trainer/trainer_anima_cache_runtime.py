# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Anima cache-first runtime cluster extracted verbatim from ``trainer.py`` as a mixin.

These methods (cached-executable preparation, anima-faithful effective config /
text-cache 512 handling, component-path resolution, cache-encoder load/free,
faithful-adapter load, colorize cond-cache + anima-cache build, cached
dataset/dataloader + anima staged-resolution) run as bound methods of the
trainer instance — identical ``self`` semantics, identical call sites. Behaviour
is unchanged; this split only keeps ``trainer.py`` navigable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .anima_cache_runtime import build_anima_cache_encode_bundle
from .dataloader_policy import resolve_cached_dataloader_policy
from .staged_resolution import StagedResolutionStage, stages_to_summary
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


class TrainerAnimaCacheRuntimeMixin:
    def _prepare_anima_cached_executable(self, model_path: str) -> None:
        """Install the real native DiT module for cache-first Anima training."""
        online_cache_enabled = bool(getattr(self.config, "anima_online_cache", False))
        if self._anima_cache_pending and not online_cache_enabled:
            self._log("Anima cache build was deferred until model load; running now.")
            self._build_anima_cache_now()
        elif (
            self._should_auto_build_anima_cache_before_training()
            and not online_cache_enabled
            and not self._has_anima_cached_training_data()
        ):
            self._log(
                "Anima cache-first prepare: paired cache files were missing; attempting an upfront cache build."
            )
            self._build_anima_cache_now()

        # #133/#164: the faithful native forward conditions on max-padded
        # (512-token) cross-attn context (ref MAX_CROSSATTN_TOKENS). Auto-rebuild a
        # stale / un-padded text cache here — before the faithful decision and the
        # executable load — so the LoRA trains on the same 512-pad manifold the
        # inference path always uses (otherwise: flat/gray renders).
        self._ensure_anima_faithful_text_cache_512(model_path)

        if not self._has_anima_cached_training_data() and not online_cache_enabled:
            return
        if getattr(self.model, "anima_native_train_ready", False):
            return

        from .anima_native_dit import load_anima_native_executable_subset

        block_count = max(int(getattr(self.config, "anima_native_block_count", 28) or 28), 1)
        block_indices = tuple(range(block_count))
        # #147 faithful native forward (UI-default-on for native Anima; backend
        # config default still off). The resolver returns the EFFECTIVE flag and
        # degrades gracefully (timestep ① and adapter-context ③ turn off together)
        # + warns + records on the run when the cache/config can't support faithful.
        faithful = self._resolve_anima_faithful_effective()
        self._log(
            "Anima cache-first training: loading native DiT executable "
            f"({block_count} blocks, faithful={'on' if faithful else 'off'}) from {model_path}"
        )
        native_unet, native_report = load_anima_native_executable_subset(
            model_path,
            block_indices=block_indices,
            device=self.device,
            dtype=self.dtype,
            faithful=faithful,
        )
        for param in native_unet.parameters():
            param.requires_grad_(False)

        self.model.unet = native_unet
        if not online_cache_enabled:
            self.model.text_encoder_1 = None
            self.model.text_encoder_2 = None
            self.model.vae = None
        self.model.noise_scheduler = None
        self.model.anima_native_train_ready = True
        self.model.anima_cached_training_ready = True
        self.model.anima_native_executable_report = native_report
        # Fix ③: faithful forward conditions on the *frozen* llm_adapter output
        # (Qwen3 hidden + T5 ids -> cross-attn context), not raw Qwen3 hidden.
        # Load + freeze it once here and attach it to the unet module so it
        # travels into the training loop's forward-input handler; the handler
        # runs it under no_grad per prompt. Text side never trains (Anima guidance).
        native_unet.anima_llm_adapter = None
        if faithful:
            self._load_anima_faithful_adapter(model_path)
        self._log(
            "Anima native DiT executable ready: "
            f"loaded={native_report.loaded_key_count}/{native_report.module_key_count}, "
            f"device={native_report.device}, dtype={native_report.dtype_name or self.dtype}"
        )

    def _resolve_anima_faithful_effective(self) -> bool:
        """Resolve the EFFECTIVE faithful-forward flag at prepare time.

        ``anima_faithful_forward`` is the requested opt-in (UI-default-on for
        native Anima; backend config default still off). The faithful native
        forward (3D RoPE + frozen llm_adapter context + timestep ``t in [0,1]``)
        is mutually exclusive with two things:
          * the token-routing compute-reducer seams (TREAD/DiffCR) — they change
            the token count, mis-aligning per-position 3D RoPE; an explicit
            training feature, so the explicit choice wins: degrade faithful off
            and keep the reducer. **BlockSkip is exempt** — it skips whole blocks
            via token-preserving identity passthrough, threads rope_emb cleanly,
            and now runs UNDER faithful (and coexists with block-checkpointing),
            so it no longer degrades faithful.
          * a cache built without the T5 tokenizer (no ``t5_input_ids``) — the
            frozen llm_adapter needs T5 ids, so degrade to the legacy path.

        On either incompatibility we DEGRADE gracefully rather than crash: set
        ``config.anima_faithful_forward = False`` (so BOTH the timestep ① and the
        adapter-context ③ fixes turn off together — a half-faithful state is worse
        than the clean legacy path, see the multi-style A/B), emit a loud warning +
        a runtime event, and return False. Every downstream consumer (executable
        subset, training loop, timestep/forward handlers) then reads the same
        effective value.

        When compatible, keep native block checkpointing (now RoPE-compatible, the
        activation-memory lever) and force the generic HF gradient/cpu-offload
        checkpointing off; return True.

        The degrade *policy* lives in the cleanroom
        ``resolve_anima_faithful_decision`` helper (pure / CPU-testable); this method
        only gathers the config inputs and applies the side effects.
        """
        try:
            from .anima_faithful_train_context import resolve_anima_faithful_decision
        except ImportError:  # pragma: no cover - direct-file usage
            from core.lulynx_trainer.anima_faithful_train_context import (
                resolve_anima_faithful_decision,
            )

        decision = resolve_anima_faithful_decision(
            faithful_requested=bool(getattr(self.config, "anima_faithful_forward", False)),
            reducer_strategy=getattr(self.config, "dit_compute_reducer_strategy", "none"),
            has_t5_checker=self._anima_cache_has_t5,
        )

        if not decision.effective:
            if decision.degrade_reason is not None:
                # Requested faithful but had to degrade -> loud, non-silent fallback.
                self.config.anima_faithful_forward = False
                self._anima_faithful_active = False
                self._log(
                    "[anima-faithful][degrade] faithful forward was requested but "
                    f"{decision.degrade_detail} -> falling back to the legacy non-faithful "
                    "path for this run (timestep t*1000 + raw-Qwen3 context). This is NOT an "
                    "error; the run continues as the #132 baseline. Re-enable faithful by "
                    "removing the conflict above."
                )
                self._emit_runtime_event(
                    {
                        "event_type": "anima_faithful_degraded",
                        "reason": decision.degrade_reason,
                        "detail": decision.degrade_detail,
                    }
                )
            return False

        self._anima_faithful_active = True
        # #166: faithful 3D RoPE is now compatible with native block checkpointing
        # (anima_native_dit._checkpoint_block threads rope_emb into the recomputed
        # block). So we keep anima_block_checkpointing — it is the activation-memory
        # lever that lets the 4096-token 1024px DiT fit (25G -> ~5-6G). We still clear
        # the generic gradient/cpu-offload checkpointing flags: those drive the HF/SDXL
        # prepare_for_training path, not the native executable subset.
        if bool(getattr(self.config, "gradient_checkpointing", False)) or bool(
            getattr(self.config, "cpu_offload_checkpointing", False)
        ):
            self._log(
                "[anima-faithful] generic gradient/cpu-offload checkpointing disabled "
                "(HF path, inert for the native subset); native block checkpointing is "
                "kept and is now RoPE-compatible (the faithful activation-memory lever)."
            )
        self.config.gradient_checkpointing = False
        self.config.cpu_offload_checkpointing = False
        return True

    def _anima_cache_has_t5(self, *, sample_limit: int = 8) -> bool:
        """Return True when the cached Anima text-conditioning carries t5_input_ids.

        Peeks a few ``*_anima_te.*`` caches (the same files
        ``_infer_anima_cached_text_tokens`` inspects). Native Anima caches carry
        ``t5_input_ids`` in every file by default; old / clip-primary caches do not.
        Returns True on the first hit. Fail-open (returns True) when there is nothing
        to peek, leaving the per-batch forward resolver as the final guard; returns
        False only when it inspected caches and none carried the key.
        """
        cache_dirs = self._anima_cached_training_dirs()
        if not cache_dirs:
            return True
        inspected = 0
        for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
            for data_dir in cache_dirs:
                for path in sorted(data_dir.rglob(f"*{suffix}")):
                    if inspected >= sample_limit:
                        return False
                    inspected += 1
                    try:
                        if path.suffix.lower() == ".npz":
                            with np.load(str(path), mmap_mode="r") as data:
                                if "t5_input_ids" in data.files:
                                    return True
                        elif path.suffix.lower() == ".safetensors":
                            from safetensors import safe_open
                            with safe_open(str(path), framework="pt", device="cpu") as data:
                                if "t5_input_ids" in data.keys():
                                    return True
                        elif path.suffix.lower() == ".pt":
                            payload = safe_torch_load(str(path), map_location="cpu")
                            if isinstance(payload, dict) and "t5_input_ids" in payload:
                                return True
                    except Exception as exc:
                        logger.debug("Failed to inspect Anima text cache for t5_input_ids from %s: %s", path, exc)
        return False if inspected else True

    def _anima_text_cache_seq_lengths(
        self, *, sample_limit: int = 8
    ) -> tuple[Optional[int], Optional[int]]:
        """Peek cached ``*_anima_te.*`` and return (min qwen3 seq, min t5 seq).

        Mirrors :meth:`_anima_cache_has_t5`'s peek skeleton (same suffixes, same
        ``_anima_cached_training_dirs`` source, same fail-open contract). Reads the
        leading dim of ``prompt_embeds`` (the qwen3/source cross-attn length) and
        ``t5_input_ids`` (the T5 length) cheaply (mmap / slice shape, no full load).
        Returns the minimum length seen for each stream across the peeked samples,
        or ``None`` for a stream when nothing carried it — feeding the pure
        :func:`anima_text_cache_needs_padding_rebuild` policy to decide staleness.
        """
        cache_dirs = self._anima_cached_training_dirs()
        if not cache_dirs:
            return None, None
        qwen3_min: Optional[int] = None
        t5_min: Optional[int] = None

        def _track(current: Optional[int], value: Optional[int]) -> Optional[int]:
            if value is None:
                return current
            return value if current is None else min(current, value)

        inspected = 0
        for suffix in ("_anima_te.npz", "_anima_te.safetensors", "_anima_te.pt"):
            for data_dir in cache_dirs:
                for path in sorted(data_dir.rglob(f"*{suffix}")):
                    if inspected >= sample_limit:
                        return qwen3_min, t5_min
                    inspected += 1
                    try:
                        q_seq, t_seq = self._peek_anima_text_seq(path)
                    except Exception as exc:
                        logger.debug("Failed to peek Anima text cache seq from %s: %s", path, exc)
                        continue
                    qwen3_min = _track(qwen3_min, q_seq)
                    t5_min = _track(t5_min, t_seq)
        return qwen3_min, t5_min

    @staticmethod
    def _peek_anima_text_seq(path: Path) -> tuple[Optional[int], Optional[int]]:
        """Return (prompt_embeds seq, t5_input_ids seq) for one cache file, or None."""
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(str(path), mmap_mode="r") as data:
                q = data["prompt_embeds"].shape[0] if "prompt_embeds" in data.files else None
                t = data["t5_input_ids"].shape[0] if "t5_input_ids" in data.files else None
                return q, t
        if suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(path), framework="pt", device="cpu") as data:
                keys = set(data.keys())
                q = data.get_slice("prompt_embeds").get_shape()[0] if "prompt_embeds" in keys else None
                t = data.get_slice("t5_input_ids").get_shape()[0] if "t5_input_ids" in keys else None
                return q, t
        payload = safe_torch_load(str(path), map_location="cpu")
        if isinstance(payload, dict):
            q = payload["prompt_embeds"].shape[0] if "prompt_embeds" in payload else None
            t = payload["t5_input_ids"].shape[0] if "t5_input_ids" in payload else None
            return q, t
        return None, None

    def _ensure_anima_faithful_text_cache_512(self, model_path: str) -> None:
        """Force-rebuild the Anima text cache to full padding when faithful needs it.

        The faithful native forward conditions on the pretrained model's max-padded
        (512-token) cross-attn context (the Anima ``MAX_CROSSATTN_TOKENS`` invariant:
        text must be padded to max at *every* stage, never trimmed). A faithful LoRA
        trained on an un-padded cache (older builder leaves the natural ~66/74 length)
        learns the wrong text manifold and renders flat/gray against the inference
        path, which always pads to 512. So when faithful is requested we (1) pin both
        the build-side and load-side text trims off so the 512 padding survives, and
        (2) auto force-rebuild any prebuilt cache whose sequences are short.
        """
        if not bool(getattr(self.config, "anima_faithful_forward", False)):
            return
        if not bool(getattr(self.config, "anima_cached_training", False)):
            return
        # The 512-pad invariant must hold at BOTH the build (encode) and load
        # (dataset consume) stages, else either side silently trims it back off.
        self.config.anima_cached_text_token_limit = 0
        self.config.anima_text_token_limit = 0
        if bool(getattr(self.config, "anima_online_cache", False)):
            # Online cache encodes 512-padded samples live; no prebuilt cache to fix.
            return

        try:
            from .anima_faithful_train_context import anima_text_cache_needs_padding_rebuild
        except ImportError:  # pragma: no cover - direct-file usage
            from core.lulynx_trainer.anima_faithful_train_context import (
                anima_text_cache_needs_padding_rebuild,
            )

        expected_qwen3 = int(getattr(self.config, "anima_qwen3_max_token_length", 0) or 0) or 512
        expected_t5 = int(getattr(self.config, "anima_t5_max_token_length", 0) or 0) or 512
        qwen3_seq, t5_seq = self._anima_text_cache_seq_lengths()
        if not anima_text_cache_needs_padding_rebuild(
            qwen3_seq=qwen3_seq,
            t5_seq=t5_seq,
            expected_qwen3=expected_qwen3,
            expected_t5=expected_t5,
        ):
            return

        self._log(
            f"[anima-faithful] cached text is too short for the faithful path "
            f"(qwen3 seq={qwen3_seq}, t5 seq={t5_seq}; need {expected_qwen3}/{expected_t5}-pad "
            "cross-attn context per the Anima MAX_CROSSATTN_TOKENS invariant). A LoRA trained on "
            "un-padded text mismatches the always-512-padded inference path and renders flat/gray. "
            "Auto force-rebuilding the Anima cache at full padding now (re-encodes latents + text)."
        )
        # Cache-first deliberately skips loading the VAE + Qwen3/T5 encoders (it trains
        # straight off prebuilt tensors), but the force rebuild needs them to re-encode.
        # Load them from the model dir's standard subfolders just for the rebuild, then
        # free so the cache-first VRAM profile is restored before the faithful executable
        # (which disables checkpointing and is already VRAM-heavy) loads.
        loaded_for_rebuild = self._ensure_anima_cache_encoders_loaded(model_path)
        try:
            self._build_anima_cache_now(force=True)
        finally:
            if loaded_for_rebuild:
                self._free_anima_cache_encoders()

    def _resolve_anima_component_paths(self, model_path: str) -> dict:
        """Probe the standard Anima component subfolders under the model dir.

        Native Anima ships VAE / Qwen3 / T5 alongside the DiT checkpoint under
        ``vae/``, ``text_encoders/`` and ``tokenizer_t5/`` (the layout the family
        spec uses). Cache-first runs don't carry these paths in the config, so the
        512-pad rebuild resolves them here from ``model_path``. ``model_path`` is
        usually the single-file DiT (``<root>/diffusion_models/anima-base*.safetensors``)
        so we walk up a couple of levels to find the model root that actually holds
        the component subfolders. Returns a dict with any of ``vae`` / ``qwen3`` /
        ``t5`` that exist; missing keys are omitted (fail-open).
        """
        from pathlib import Path

        start = Path(str(model_path or ""))
        if start.is_file() or start.suffix:
            start = start.parent
        # The DiT lives under ``diffusion_models/`` while vae/ + text_encoders/ are
        # its siblings, so probe this dir and a couple of parents for the root that
        # actually carries the component subfolders.
        root = start
        for candidate in (start, start.parent, start.parent.parent):
            if (candidate / "vae").is_dir() or (candidate / "text_encoders").is_dir():
                root = candidate
                break
        out: dict = {}
        if not root.is_dir():
            return out

        def _pick_file(subdir: str, preferred: str) -> str:
            folder = root / subdir
            if not folder.is_dir():
                return ""
            pref = folder / preferred
            if pref.is_file():
                return str(pref)
            cands = sorted(folder.glob("*.safetensors"))
            return str(cands[0]) if cands else ""

        vae = _pick_file("vae", "qwen_image_vae.safetensors")
        qwen3 = _pick_file("text_encoders", "qwen_3_06b_base.safetensors")
        t5_dir = root / "tokenizer_t5"
        if vae:
            out["vae"] = vae
        if qwen3:
            out["qwen3"] = qwen3
        if t5_dir.is_dir():
            out["t5"] = str(t5_dir)
        return out

    def _ensure_anima_cache_encoders_loaded(self, model_path: str) -> bool:
        """Load the VAE + Qwen3/T5 encoders into ``self.model`` for a cache rebuild.

        Cache-first strips these (it trains off prebuilt tensors), but
        ``build_anima_cache_encode_bundle`` requires a live VAE + Qwen3 encoder +
        T5 tokenizer to re-encode. Resolve them from the model dir and copy them
        onto the existing model. Returns ``True`` only when this call loaded them
        (so the caller frees them afterward); ``False`` when the VAE is already
        present, the model is absent (CPU smoke), or the components can't be found.
        """
        model = getattr(self, "model", None)
        if model is None or getattr(model, "vae", None) is not None:
            return False
        paths = self._resolve_anima_component_paths(model_path)
        if not paths.get("vae"):
            self._log(
                "[anima-faithful] could not locate the VAE/encoders under the model dir; "
                "skipping the 512-pad rebuild (the short text cache stays as-is)."
            )
            return False
        try:
            from .anima_loader import load_anima_model
        except ImportError:  # pragma: no cover - direct-file usage
            from core.lulynx_trainer.anima_loader import load_anima_model
        helper_model, _report = load_anima_model(
            model_path=model_path,
            qwen3_path=paths.get("qwen3", ""),
            t5_tokenizer_path=paths.get("t5", ""),
            vae_path=paths.get("vae", ""),
            device=self.device,
            dtype=self.dtype,
            disable_mmap=bool(getattr(self.config, "disable_mmap_load_safetensors", False)),
        )
        for attr in (
            "vae",
            "anima_qwen3_encoder",
            "anima_qwen3_tokenizer",
            "anima_t5_tokenizer",
            "text_encoder_1",
            "text_encoder_2",
            "tokenizer_1",
        ):
            value = getattr(helper_model, attr, None)
            if value is not None:
                setattr(self.model, attr, value)
        from pathlib import Path as _Path

        self._log(
            "[anima-faithful] loaded VAE+Qwen3+T5 from the model dir for the 512-pad "
            f"rebuild (vae={_Path(paths['vae']).name})."
        )
        return True

    def _free_anima_cache_encoders(self) -> None:
        """Drop the rebuild-only encoders so cache-first VRAM is restored.

        The 2459-2462 path nulls ``vae`` + the CLIP encoders, but not the Qwen3
        encoder; null all of them here so the (checkpointing-disabled) faithful
        executable gets the lean cache-first memory profile it expects.
        """
        model = getattr(self, "model", None)
        if model is None:
            return
        for attr in (
            "vae",
            "anima_qwen3_encoder",
            "anima_qwen3_tokenizer",
            "anima_t5_tokenizer",
            "text_encoder_1",
            "text_encoder_2",
        ):
            if getattr(model, attr, None) is not None:
                setattr(model, attr, None)
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best-effort VRAM reclaim
            pass

    def _load_anima_faithful_adapter(self, model_path: str) -> None:
        """Load + freeze the native ``llm_adapter`` for faithful cross-attn context.

        The text side stays frozen throughout: the adapter is *run* (no_grad) to
        map Qwen3 hidden + T5 ids into cross-attn context, never trained."""
        try:
            from .anima_native_faithful import load_anima_llm_adapter
        except ImportError:  # pragma: no cover - direct-file usage
            from core.lulynx_trainer.anima_native_faithful import load_anima_llm_adapter
        adapter, report = load_anima_llm_adapter(model_path, device=self.device, dtype=self.dtype)
        adapter.requires_grad_(False)
        adapter.eval()
        self.model.unet.anima_llm_adapter = adapter
        self._log(
            "Anima faithful llm_adapter loaded (frozen): "
            f"layers={report.get('num_layers')}, heads={report.get('num_heads')}, "
            f"keys={report.get('loaded_keys')}/{report.get('total_keys')}"
        )
    def _maybe_build_colorize_cond_cache(self, encode_bundle, data_dir: str, force: bool) -> None:
        """EasyControl v2 colorize keystone: VAE-encode control images → cond sidecars.

        Default-off. Runs ONLY when ``easycontrol_v2_enabled`` + ``task_id=="colorize"``
        and both ``control_image_dir`` + ``cond_cache_dir`` are configured. This produces
        the per-target cond-latent sidecars the dataset loader expects (the one piece the
        colorize pipeline was missing) so training consumes a REAL control-image condition
        rather than the cache-first derived-from-target fallback. No-op otherwise; never
        fabricates a condition for targets whose control image is absent.
        """
        if not bool(getattr(self.config, "easycontrol_v2_enabled", False)):
            return
        try:
            from .colorize_cond_cache import build_colorize_cond_cache
            from .easycontrol_v2_contract import build_easycontrol_v2_task_spec_from_config
        except Exception as exc:  # pragma: no cover - import guard
            self._log(f"[colorize-cond-cache] unavailable: {exc}")
            return

        try:
            spec = build_easycontrol_v2_task_spec_from_config(self.config).normalized()
        except Exception as exc:
            self._log(f"[colorize-cond-cache] task spec invalid: {exc}")
            return
        if spec.task_id != "colorize":
            return
        if not spec.control_image_dir or not spec.cond_cache_dir:
            self._log(
                "[colorize-cond-cache] colorize enabled but control_image_dir / cond_cache_dir "
                "is unset; skipping cond-latent build (dataset loader will fall back)."
            )
            return

        from core.tools.image_utilities import IMAGE_EXTS

        # Enumerate target images under data_dir, excluding the control-image / cache
        # subtrees so we never treat a control image (or a produced sidecar) as a target.
        exclude_roots = []
        for raw in (spec.control_image_dir, spec.cond_cache_dir, spec.text_cache_dir):
            if not raw:
                continue
            try:
                exclude_roots.append(Path(raw).resolve())
            except Exception:
                continue
        targets: list[str] = []
        root = Path(data_dir)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            if any(resolved == r or r in resolved.parents for r in exclude_roots):
                continue
            targets.append(str(path))
        if not targets:
            self._log("[colorize-cond-cache] no target images found under data_dir; nothing to encode.")
            return

        report = build_colorize_cond_cache(
            target_image_paths=targets,
            spec=spec,
            vae_encode_fn=encode_bundle.vae_encode_fn,
            vae_chunk_size=int(getattr(self.config, "anima_vae_chunk_size", 0) or 0),
            disk_dtype=self._resolve_cache_disk_dtype(
                getattr(self.config, "latent_cache_disk_dtype", "float16")
            ),
            force=force,
            log=self._log,
        )
        self._anima_cache_builder_profile["colorize_cond_cache"] = report.to_dict()

    def _build_anima_cache_now(self, *, force: bool = False) -> None:
        """Build the Anima latent + text cache before training begins.

        Uses ``anima_cache_builder.build_anima_cache`` with VAE and text-encode
        callables sourced from the loaded model.  Skipped silently if the
        model has not been loaded yet — the caller should arrange ordering.

        ``force=True`` overwrites existing cache files (used by rebuild_cache mode
        and the faithful 512-pad auto-rebuild guard; without it the builder skips
        any sample whose cache already exists). The flag is remembered across the
        deferred build (model not yet loaded) so a pre-load rebuild request still
        forces on the real build.
        """
        from .anima_cache_builder import AnimaCacheBuilderConfig, build_anima_cache
        from .caption_source_mix import normalize_caption_source_mix_config

        force = bool(force) or bool(getattr(self, "_anima_cache_force_rebuild", False))

        data_dir = str(getattr(self.config, "train_data_dir", "") or "").strip()
        if not data_dir:
            self._log("Anima rebuild_cache requested but train_data_dir is empty; skipping.")
            return

        if self.model is None:
            self._log("Anima rebuild_cache requested but model not yet loaded; will build on first step.")
            self._anima_cache_pending = True
            self._anima_cache_force_rebuild = force
            return

        self._anima_cache_pending = False
        self._anima_cache_force_rebuild = False
        try:
            encode_bundle = build_anima_cache_encode_bundle(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                config=self.config,
            )
        except RuntimeError as exc:
            self._log(f"Anima cache builder: {exc}")
            return

        self._log(f"Anima cache builder inputs ready: {encode_bundle.summary}")

        # EasyControl v2 colorize: VAE-encode the control images into cond-latent
        # sidecars (the one keystone the colorize pipeline was missing). Default-off;
        # only runs when easycontrol_v2_enabled + task_id=colorize + dirs are set.
        self._maybe_build_colorize_cond_cache(encode_bundle, data_dir, force)

        cfg = AnimaCacheBuilderConfig(
            data_dir=data_dir,
            output_dir=data_dir,
            vae_chunk_size=int(getattr(self.config, "anima_vae_chunk_size", 0) or 0),
            text_token_limit=int(getattr(self.config, "anima_text_token_limit", 0) or 0),
            include_loss_mask=bool(getattr(self.config, "masked_loss", False)),
            disk_format=str(getattr(self.config, "latent_cache_disk_format", "npz") or "npz"),
            disk_dtype=self._resolve_cache_disk_dtype(getattr(self.config, "latent_cache_disk_dtype", "float16")),
            text_disk_format=str(getattr(self.config, "text_encoder_outputs_cache_disk_format", "npz") or "npz"),
            text_disk_dtype=self._resolve_cache_disk_dtype(
                getattr(self.config, "text_encoder_outputs_cache_disk_dtype", "float16")
            ),
            caption_source_mix=normalize_caption_source_mix_config(
                enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
                nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
                tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
                trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
                empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
                trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
            ),
        )

        staged_plan = self._build_anima_staged_resolution_plan()
        if staged_plan:
            self._log(f"Anima staged-resolution cache builder: {stages_to_summary(staged_plan)}")
            total_written = 0
            total_skipped = 0
            total_errors = 0
            for stage in staged_plan:
                stage_cfg = AnimaCacheBuilderConfig(
                    data_dir=cfg.data_dir,
                    output_dir=stage.cache_dir,
                    vae_chunk_size=cfg.vae_chunk_size,
                    text_token_limit=cfg.text_token_limit,
                    include_loss_mask=cfg.include_loss_mask,
                    disk_format=cfg.disk_format,
                    disk_dtype=cfg.disk_dtype,
                    text_disk_format=cfg.text_disk_format,
                    text_disk_dtype=cfg.text_disk_dtype,
                    target_resolution=stage.resolution,
                    caption_source_mix=cfg.caption_source_mix,
                )
                self._log(
                    "Anima staged cache builder: "
                    f"resolution={stage.resolution}, batch={stage.batch_size or getattr(self.config, 'batch_size', 1)}, "
                    f"output={stage.cache_dir}"
                )
                result = build_anima_cache(
                    vae_encode_fn=encode_bundle.vae_encode_fn,
                    text_encode_fn=encode_bundle.text_encode_fn,
                    config=stage_cfg,
                    force=force,
                    log=self._log,
                )
                total_written += result.written
                total_skipped += result.skipped
                total_errors += len(result.errors)
                self._anima_cache_builder_profile.setdefault("staged_cache_trust", []).append(
                    {
                        "resolution": int(stage.resolution),
                        "cache_dir": str(stage.cache_dir),
                        "written": int(result.written),
                        "skipped": int(result.skipped),
                        "manifest_path": str(getattr(result, "manifest_path", "") or ""),
                        "cache_trust": dict(getattr(result, "cache_trust", {}) or {}),
                    }
                )
                self._log(
                    f"Anima staged cache stage finished: resolution={stage.resolution}, "
                    f"written={result.written}, skipped={result.skipped}, errors={len(result.errors)}"
                )
            self._log(
                f"Anima staged cache builder finished: written={total_written}, "
                f"skipped={total_skipped}, errors={total_errors}"
            )
            return

        self._log(f"Anima cache builder: scanning {data_dir}...")
        result = build_anima_cache(
            vae_encode_fn=encode_bundle.vae_encode_fn,
            text_encode_fn=encode_bundle.text_encode_fn,
            config=cfg,
            force=force,
            log=self._log,
        )
        self._log(
            f"Anima cache builder finished: written={result.written}, skipped={result.skipped}, "
            f"errors={len(result.errors)}"
        )
        self._anima_cache_builder_profile = {
            "written": int(result.written),
            "skipped": int(result.skipped),
            "errors": len(result.errors),
            "manifest_path": str(getattr(result, "manifest_path", "") or ""),
            "cache_trust": dict(getattr(result, "cache_trust", {}) or {}),
        }
    def _create_anima_cached_dataset(self, data_dir: str | Path):
        from .anima_cached_dataset import AnimaCachedDataset

        easycontrol_v2_spec = None
        if bool(getattr(self.config, "easycontrol_v2_enabled", False)):
            try:
                from .easycontrol_v2_contract import build_easycontrol_v2_task_spec_from_config

                easycontrol_v2_spec = build_easycontrol_v2_task_spec_from_config(self.config).normalized()
            except Exception as exc:
                self._log(f"[easycontrol-v2] cond sidecar spec unavailable for cached dataset: {exc}")
                easycontrol_v2_spec = None

        return AnimaCachedDataset(
            data_dir=data_dir,
            latent_crop_size=int(getattr(self.config, "anima_cached_latent_crop_size", 0) or 0),
            text_token_limit=int(getattr(self.config, "anima_cached_text_token_limit", 0) or 0),
            fixed_text_tokens=int(getattr(self.config, "anima_fixed_text_tokens", 0) or 0),
            fixed_visual_tokens=int(getattr(self.config, "anima_fixed_visual_tokens", 0) or 0),
            caption_extension=getattr(self.config, "caption_extension", ".txt"),
            shuffle_caption=bool(getattr(self.config, "shuffle_caption", False)),
            shuffle_caption_tags_only=bool(getattr(self.config, "shuffle_caption_tags_only", False)),
            keep_tokens=int(getattr(self.config, "keep_tokens", 0) or 0),
            keep_tokens_separator=str(getattr(self.config, "keep_tokens_separator", "") or ""),
            weighted_captions=bool(getattr(self.config, "weighted_captions", False)),
            caption_source_mix_enabled=bool(getattr(self.config, "caption_source_mix_enabled", False)),
            caption_source_nl_ratio=getattr(self.config, "caption_source_nl_ratio", 65.0),
            caption_source_tag_ratio=getattr(self.config, "caption_source_tag_ratio", 20.0),
            caption_source_trigger_only_ratio=getattr(self.config, "caption_source_trigger_only_ratio", 10.0),
            caption_source_empty_ratio=getattr(self.config, "caption_source_empty_ratio", 5.0),
            caption_source_trigger_tokens=str(getattr(self.config, "caption_source_trigger_tokens", "") or ""),
            concept_geometry_enabled=bool(getattr(self.config, "concept_geometry_enabled", getattr(self.config, "h_lora_enabled", False))),
            concept_geometry_path=str(getattr(self.config, "concept_geometry_path", getattr(self.config, "h_lora_geometry_path", "")) or ""),
            concept_geometry_sampler_mode=str(getattr(self.config, "concept_geometry_sampler_mode", getattr(self.config, "h_lora_sampler_mode", "density_curriculum")) or "density_curriculum"),
            concept_geometry_loss_weighting=bool(getattr(self.config, "concept_geometry_loss_weighting", getattr(self.config, "h_lora_loss_weighting", False))),
            concept_geometry_density_power=float(getattr(self.config, "concept_geometry_density_power", getattr(self.config, "h_lora_density_power", 1.0)) or 1.0),
            concept_geometry_seed=int(getattr(self.config, "seed", 42) or 42),
            concept_geometry_total_epochs=int(getattr(self.config, "max_train_epochs", 1) or 1),
            concept_geometry_total_steps=int(getattr(self.config, "max_train_steps", 0) or 0),
            benchmark_data_wait_stall_ms=float(
                getattr(self.config, "bubble_controller_benchmark_data_wait_stall_ms", 0.0) or 0.0
            ),
            easycontrol_v2_spec=easycontrol_v2_spec,
        )

    def _create_anima_cached_dataloader(self, dataset, *, batch_size: int, drop_last: bool = False):
        from .anima_cached_dataset import create_anima_cached_dataloader

        policy = resolve_cached_dataloader_policy(
            self.config,
            route="anima",
            cached=True,
            cuda_available=str(self.device).startswith("cuda"),
        )
        replacement_mode = str(getattr(self.config, "lossless_cache_replacement_mode", "off") or "off").lower()
        if replacement_mode in {"anima_lynx_manifest_probe", "lynx_manifest_probe"}:
            from .lossless_anima_lynx_manifest_dataloader import (
                AnimaLynxManifestDataLoaderConfig,
                create_anima_lynx_manifest_dataloader,
                parse_focus_sample_ids,
            )
            from .lossless_anima_cache_replacement_dataloader import parse_lossless_cache_codecs

            sidecar_dir = str(getattr(self.config, "lossless_cache_replacement_sidecar_dir", "") or "")
            manifest_path = str(getattr(self.config, "lossless_cache_replacement_manifest_path", "") or "")
            self._log(
                "[lossless-cache] experimental Anima .lynx manifest replacement DataLoader enabled "
                "(explicit A/B probe only; default path remains raw cached DataLoader)"
            )
            return create_anima_lynx_manifest_dataloader(
                dataset,
                batch_size=max(int(batch_size or 1), 1),
                shuffle=True,
                drop_last=drop_last,
                config=AnimaLynxManifestDataLoaderConfig(
                    manifest_path=manifest_path or None,
                    container_dir=sidecar_dir or None,
                    shard_size=max(int(getattr(self.config, "lossless_cache_replacement_shard_size", 16) or 16), 1),
                    codecs=parse_lossless_cache_codecs(
                        getattr(self.config, "lossless_cache_replacement_codecs", "raw")
                    ),
                    min_saving=float(getattr(self.config, "lossless_cache_replacement_min_saving", 0.0) or 0.0),
                    prepare_manifest=bool(
                        getattr(self.config, "lossless_cache_replacement_prepare_sidecars", False)
                    ),
                    copy_arrays=bool(getattr(self.config, "lossless_cache_replacement_copy_arrays", True)),
                    verify_crc32=bool(getattr(self.config, "lossless_cache_replacement_strict", True)),
                    collate_mode=getattr(self.config, "cached_collate_mode", "auto"),
                    seed=int(getattr(self.config, "seed", 42) or 42),
                    focus_sample_ids=parse_focus_sample_ids(
                        getattr(self.config, "lossless_cache_replacement_focus_sample_ids", "")
                    ),
                ),
            )
        if replacement_mode in {"anima_lxfs_probe", "flat_lxfs_probe", "lxfs_probe"}:
            from .lossless_anima_cache_replacement_dataloader import (
                AnimaLosslessReplacementDataLoaderConfig,
                create_anima_lossless_cache_replacement_dataloader,
                parse_focus_sample_ids,
                parse_lossless_cache_codecs,
            )

            self._log(
                "[lossless-cache] experimental Anima flat LXFS replacement DataLoader enabled "
                "(explicit A/B probe only; default path remains raw cached DataLoader)"
            )
            return create_anima_lossless_cache_replacement_dataloader(
                dataset,
                batch_size=max(int(batch_size or 1), 1),
                shuffle=True,
                drop_last=drop_last,
                config=AnimaLosslessReplacementDataLoaderConfig(
                    prefetch_depth=max(
                        int(getattr(self.config, "lossless_cache_replacement_prefetch_depth", 2) or 2),
                        1,
                    ),
                    sidecar_dir=str(getattr(self.config, "lossless_cache_replacement_sidecar_dir", "") or "") or None,
                    sidecar_format="lxfs",
                    sidecar_suffix=str(
                        getattr(self.config, "lossless_cache_replacement_sidecar_suffix", ".lxfs") or ".lxfs"
                    ),
                    sidecar_strict=bool(getattr(self.config, "lossless_cache_replacement_strict", False)),
                    fallback_to_raw=bool(getattr(self.config, "lossless_cache_replacement_fallback_to_raw", True)),
                    prepare_sidecars=bool(
                        getattr(self.config, "lossless_cache_replacement_prepare_sidecars", False)
                    ),
                    min_saving=float(getattr(self.config, "lossless_cache_replacement_min_saving", 0.0) or 0.0),
                    codecs=parse_lossless_cache_codecs(
                        getattr(self.config, "lossless_cache_replacement_codecs", "lz4fast")
                    ),
                    collate_mode=getattr(self.config, "cached_collate_mode", "auto"),
                    seed=int(getattr(self.config, "seed", 42) or 42),
                    focus_sample_ids=parse_focus_sample_ids(
                        getattr(self.config, "lossless_cache_replacement_focus_sample_ids", "")
                    ),
                ),
            )
        return create_anima_cached_dataloader(
            dataset,
            batch_size=max(int(batch_size or 1), 1),
            shuffle=True,
            num_workers=policy.num_workers,
            persistent_workers=policy.persistent_workers,
            pin_memory=policy.pin_memory,
            prefetch_factor=policy.prefetch_factor,
            drop_last=drop_last,
            collate_mode=getattr(self.config, "cached_collate_mode", "auto"),
        )

    def _select_anima_staged_resolution_stage(self, epoch: int) -> tuple[int, StagedResolutionStage] | tuple[int, None]:
        stages = self._build_anima_staged_resolution_plan()
        if not stages:
            return -1, None
        selected = 0
        for index, stage in enumerate(stages):
            if epoch >= int(stage.start_epoch):
                selected = index
        return selected, stages[selected]

    def _maybe_switch_anima_staged_resolution_dataset(
        self,
        *,
        dataloader,
        epoch: int,
        drop_last: bool,
    ):
        index, stage = self._select_anima_staged_resolution_stage(epoch)
        if stage is None or index == self._anima_staged_resolution_active_index:
            return dataloader

        dataset = self._create_anima_cached_dataset(stage.cache_dir)
        batch_size = int(stage.batch_size or getattr(self.config, "batch_size", 1) or 1)
        new_dataloader = self._create_anima_cached_dataloader(dataset, batch_size=batch_size, drop_last=drop_last)
        self._capture_cache_reader_decode_sidecar_profile(
            new_dataloader,
            route="anima_cached",
            source=f"stage_{index}",
        )
        self._capture_cache_reader_training_gate_profile(
            new_dataloader,
            route="anima_cached",
            source=f"stage_{index}",
        )

        if self._ddp_wrapper is not None:
            try:
                from .distributed import wrap_dataloader_for_ddp

                new_dataloader = wrap_dataloader_for_ddp(new_dataloader, dataset, shuffle=True, seed=int(getattr(self.config, "seed", 42) or 42))
                self._ddp_wrapper._dataloader = new_dataloader
                self._ddp_wrapper._ddp_sampler = getattr(new_dataloader, "sampler", None)
            except Exception as exc:
                self._log(f"Anima staged resolution DDP dataloader refresh skipped: {exc}")

        self._dataset = dataset
        self._anima_staged_resolution_active_index = index
        self._apply_staged_resolution(stage.resolution)
        self._log(
            "Anima staged resolution: "
            f"epoch={epoch + 1}, resolution={stage.resolution}, "
            f"batch={batch_size}, samples={len(dataset)}, cache={stage.cache_dir}"
        )
        return new_dataloader


__all__ = ["TrainerAnimaCacheRuntimeMixin"]

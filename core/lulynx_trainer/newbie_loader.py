"""
Self-contained Newbie model loader.

Loads a Newbie checkpoint (diffusers-format directory) into a
``LoadedModel`` dataclass that is compatible with the native trainer's
existing contract.  This module is designed to be called from
``ModelLoader._load_newbie()`` once integration is wired up, but it can
also be used standalone for testing.

Design principles
-----------------
1. **No shared-file mutations.**  This module imports from
   ``model_loader.LoadedModel`` but does not modify it.
2. **Honest about assumptions.**  Every architectural guess is tagged via
   ``newbie_contract.NEWBIE_CONTRACT`` and logged at load time.
3. **Config-aware.**  Respects the Newbie-specific config hints:
   - ``newbie_diffusers_path``  – where to load from
   - ``newbie_lora_target``     – which LoRA preset to prefer
   - ``newbie_gemma3_prompt``   – stored as metadata for future use
   - ``newbie_use_flash_attn2`` – fed into the runtime optimization plan
4. **Graceful degradation.**  If the directory is SDXL-shaped (the Phase-1
   case), the loader still works -- it just logs that it is operating in
   scaffold mode.

Usage
-----
::

    from core.lulynx_trainer.newbie_loader import load_newbie

    model = load_newbie(
        diffusers_path="/path/to/newbie-diffusers",
        device="cuda",
        dtype=torch.bfloat16,
        lora_target="balanced",
        gemma3_prompt="",
        use_flash_attn2=True,
    )
"""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .model_loader import LoadedModel
from .newbie_contract import NEWBIE_CONTRACT, audit_newbie_contract
from .newbie_smoke import run_loaded_newbie_smoke
from .newbie_targets import get_newbie_targets

logger = logging.getLogger(__name__)


def _progress_value(value: Any) -> Any:
    """Normalize loader progress payload values for JSON/event sinks."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_progress_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _progress_value(item) for key, item in value.items()}
    return str(value)


def _emit_loader_progress(
    callback: Optional[Callable[[str, Dict[str, Any]], None]],
    stage: str,
    **data: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), {str(key): _progress_value(value) for key, value in data.items()})
    except Exception as exc:
        logger.debug("Newbie loader progress callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_JINA_CLIP_SUPPORT_FILES = (
    "__init__.py",
    "configuration_clip.py",
    "eva_model.py",
    "hf_model.py",
    "modeling_clip.py",
    "rope_embeddings.py",
    "transform.py",
)


def _repo_root() -> Path:
    """Return the repo root for runtime-only cache staging."""
    return Path(__file__).resolve().parents[3]


def _runtime_model_overlay_root() -> Path:
    """Return the ignored runtime overlay root used for dependency recovery."""
    root = _repo_root() / "backend" / "tmp" / "model_overlays"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _find_jina_clip_support_cache() -> Optional[Path]:
    """Locate a local Hugging Face module cache containing Jina CLIP support files."""
    env_candidates = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        env_candidates.append(Path(hf_home) / "modules" / "transformers_modules")
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        env_candidates.append(Path(hub_cache).parent / "modules" / "transformers_modules")

    default_candidates = [
        Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for base in [*env_candidates, *default_candidates]:
        if not base.exists():
            continue
        if base.name == "hub":
            matches = list(base.glob("models--jinaai--jina-clip-implementation/**/eva_model.py"))
            for match in matches:
                parent = match.parent
                if all((parent / name).exists() for name in ("eva_model.py", "hf_model.py")):
                    return parent
            continue

        matches = list(base.glob("jinaai/jina-clip-implementation/*/eva_model.py"))
        for match in matches:
            parent = match.parent
            if all((parent / name).exists() for name in ("eva_model.py", "hf_model.py")):
                return parent
    return None


def _best_effort_link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hard links for large immutable files; fall back to copy."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _file_readable(path: Path) -> bool:
    """Return True when a file can be opened for reading."""
    try:
        with path.open("rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _safe_json_dict(path: Path) -> Dict[str, Any]:
    """Best-effort JSON object read that treats unreadable files as empty."""
    if not path.is_file() or not _file_readable(path):
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _looks_like_local_jina_clip_signature(model_dir: Path) -> bool:
    """Infer a local Jina CLIP snapshot from its support files and weights."""
    return (
        (model_dir / "configuration_clip.py").is_file()
        and (model_dir / "modeling_clip.py").is_file()
        and _find_local_jina_weight_file(model_dir) is not None
    )


def _write_minimal_jina_clip_config(dst: Path, model_dir: Path) -> None:
    """Write a minimal local Jina CLIP config when the original config is unreadable."""
    tokenizer_config = _safe_json_dict(model_dir / "tokenizer_config.json")
    pad_token_id = int(tokenizer_config.get("pad_token_id", 0) or 0)
    config = {
        "architectures": ["JinaCLIPModel"],
        "model_type": "jina_clip",
        "projection_dim": int(getattr(NEWBIE_CONTRACT, "clip_projection_dim", 1024) or 1024),
        "pad_token_id": pad_token_id,
        "auto_map": {
            "AutoConfig": "configuration_clip.JinaCLIPConfig",
            "AutoModel": "modeling_clip.JinaCLIPModel",
        },
    }
    dst.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _looks_like_local_jina_clip(model_dir: Path) -> bool:
    """Return True when a local directory declares the custom Jina CLIP auto_map."""
    config_path = model_dir / "config.json"
    if not config_path.is_file() or not _file_readable(config_path):
        return _looks_like_local_jina_clip_signature(model_dir)
    config = _safe_json_dict(config_path)
    auto_map = config.get("auto_map") or {}
    return (
        auto_map.get("AutoConfig") == "configuration_clip.JinaCLIPConfig"
        and auto_map.get("AutoModel") == "modeling_clip.JinaCLIPModel"
    )


def _ensure_transformers_clip_compat() -> None:
    """Bridge small CLIP API drift needed by the local Jina CLIP snapshot."""
    from transformers.models.clip import modeling_clip as hf_clip_modeling

    if not hasattr(hf_clip_modeling, "clip_loss") and hasattr(
        hf_clip_modeling, "image_text_contrastive_loss"
    ):
        hf_clip_modeling.clip_loss = hf_clip_modeling.image_text_contrastive_loss


class _LocalJinaTextPooledEncoder(torch.nn.Module):
    """Stable local text-only pooled encoder reconstructed from Jina weights.

    The observed local ``clip_model`` snapshot contains only the text tower
    checkpoint under ``model.*`` keys. Loading it as a full ``JinaCLIPModel``
    leaves large parts of the module randomly initialized, which can surface as
    unstable pooled outputs. This wrapper uses the local tokenizer + text-tower
    weights directly to produce a deterministic 1024-d pooled text path.
    """

    def __init__(
        self,
        *,
        word_embeddings: torch.Tensor,
        emb_ln_weight: Optional[torch.Tensor],
        emb_ln_bias: Optional[torch.Tensor],
        token_type_embeddings: Optional[torch.Tensor],
        projection_dim: int,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.register_buffer("word_embeddings", word_embeddings.detach().contiguous())
        if emb_ln_weight is not None:
            self.register_buffer("emb_ln_weight", emb_ln_weight.detach().contiguous())
        else:
            self.emb_ln_weight = None
        if emb_ln_bias is not None:
            self.register_buffer("emb_ln_bias", emb_ln_bias.detach().contiguous())
        else:
            self.emb_ln_bias = None
        if token_type_embeddings is not None:
            self.register_buffer(
                "token_type_embeddings",
                token_type_embeddings.detach().contiguous(),
            )
        else:
            self.token_type_embeddings = None
        self.pad_token_id = int(pad_token_id)
        self.config = SimpleNamespace(
            projection_dim=int(projection_dim),
            hidden_size=int(word_embeddings.shape[-1]),
            model_type="jina_clip_text_only_pooler",
        )

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *_,
        **__,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("Local Jina text pooler requires input_ids")

        hidden = F.embedding(input_ids, self.word_embeddings)
        if self.token_type_embeddings is not None:
            hidden = hidden + self.token_type_embeddings[0].view(1, 1, -1)

        hidden = hidden.float()
        if self.emb_ln_weight is not None or self.emb_ln_bias is not None:
            ln_weight = None if self.emb_ln_weight is None else self.emb_ln_weight.float()
            ln_bias = None if self.emb_ln_bias is None else self.emb_ln_bias.float()
            hidden = F.layer_norm(hidden, (hidden.shape[-1],), ln_weight, ln_bias)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id)
        weights = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *_,
        **__,
    ) -> SimpleNamespace:
        if input_ids is None:
            raise RuntimeError("Local Jina text pooler requires input_ids")
        hidden = F.embedding(input_ids, self.word_embeddings).float()
        if self.token_type_embeddings is not None:
            hidden = hidden + self.token_type_embeddings[0].float().view(1, 1, -1)
        if self.emb_ln_weight is not None or self.emb_ln_bias is not None:
            ln_weight = None if self.emb_ln_weight is None else self.emb_ln_weight.float()
            ln_bias = None if self.emb_ln_bias is None else self.emb_ln_bias.float()
            hidden = F.layer_norm(hidden, (hidden.shape[-1],), ln_weight, ln_bias)
        pooled = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(
            text_embeds=pooled,
            pooler_output=pooled,
            last_hidden_state=hidden,
        )


class _EmergencyNewbieClipTokenizer:
    """Minimal tokenizer fallback for unreadable local Newbie CLIP assets."""

    def __init__(self, *, model_max_length: int = 128, vocab_size: int = 32768) -> None:
        self.model_max_length = int(max(model_max_length, 1))
        self.vocab_size = int(max(vocab_size, 1024))
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

    def _encode_text(self, text: str, max_length: int) -> list[int]:
        tokens = [self.bos_token_id]
        for piece in str(text or "").strip().split():
            bucket = abs(hash(piece)) % max(self.vocab_size - 4, 1)
            tokens.append(4 + bucket)
        tokens.append(self.eos_token_id)
        return tokens[:max_length]

    def __call__(
        self,
        text: str | list[str],
        *,
        return_tensors: str = "pt",
        padding: str | bool = False,
        truncation: bool = True,
        max_length: int | None = None,
        **_: Any,
    ) -> SimpleNamespace:
        del truncation
        if return_tensors != "pt":
            raise ValueError("_EmergencyNewbieClipTokenizer only supports return_tensors='pt'")
        texts = text if isinstance(text, list) else [text]
        target_length = int(max_length or self.model_max_length or 128)
        encoded = [self._encode_text(item, target_length) for item in texts]
        if padding in {"max_length", True}:
            padded = [seq + [self.pad_token_id] * max(target_length - len(seq), 0) for seq in encoded]
        else:
            width = max((len(seq) for seq in encoded), default=1)
            padded = [seq + [self.pad_token_id] * max(width - len(seq), 0) for seq in encoded]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token_id).long()
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        return type("EmergencyBatchEncoding", (dict,), {"__getattr__": lambda self, name: self[name]})(batch)


class _EmergencyNewbieClipEncoder(torch.nn.Module):
    """Parameter-light pooled text encoder fallback for unreadable CLIP bundles."""

    def __init__(self, *, projection_dim: int = 1024) -> None:
        super().__init__()
        self.projection_dim = int(max(projection_dim, 64))
        self.config = SimpleNamespace(
            projection_dim=self.projection_dim,
            hidden_size=self.projection_dim,
            pad_token_id=0,
        )

    def _token_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = input_ids.to(dtype=torch.float32)
        dims = torch.arange(self.projection_dim, device=base.device, dtype=torch.float32).view(1, 1, -1)
        phase = (base.unsqueeze(-1) % 251.0) / 251.0
        return torch.sin((dims + 1.0) * phase * math.pi)

    def get_text_features(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if input_ids is None:
            raise RuntimeError("Emergency Newbie CLIP encoder requires input_ids")
        hidden = self._token_features(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
        mask = attention_mask.to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        *_: Any,
        **__: Any,
    ) -> SimpleNamespace:
        if input_ids is None:
            raise RuntimeError("Emergency Newbie CLIP encoder requires input_ids")
        hidden = self._token_features(input_ids)
        pooled = self.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return SimpleNamespace(
            text_embeds=pooled,
            pooler_output=pooled,
            last_hidden_state=hidden,
        )


def _find_local_jina_weight_file(model_dir: Path) -> Optional[Path]:
    """Return the primary local safetensors file for the Jina CLIP snapshot."""
    for candidate_name in ("model.safetensors", "jina-clip-v2.safetensors"):
        candidate = model_dir / candidate_name
        if candidate.is_file():
            return candidate
    safetensors_files = sorted(model_dir.glob("*.safetensors"))
    return safetensors_files[0] if safetensors_files else None


def _looks_like_local_jina_text_only_checkpoint(model_dir: Path) -> bool:
    """Return True when the local Jina weights contain only the text tower."""
    weight_file = _find_local_jina_weight_file(model_dir)
    if weight_file is None:
        return False

    try:
        from safetensors import safe_open

        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception:
        return False

    has_text_tower = (
        "model.embeddings.word_embeddings.weight" in keys
        and any(key.startswith("model.encoder.layers.") for key in keys)
    )
    has_full_clip_keys = any(
        key.startswith(prefix)
        for key in keys
        for prefix in (
            "text_model.",
            "vision_model.",
            "text_projection",
            "visual_projection",
            "logit_scale",
        )
    )
    return has_text_tower and not has_full_clip_keys


def _load_local_jina_text_pooler(model_dir: Path) -> tuple[Optional[torch.nn.Module], list[str]]:
    """Reconstruct a stable pooled text encoder from local Jina tensors."""
    notes: list[str] = []
    if not _looks_like_local_jina_text_only_checkpoint(model_dir):
        return None, notes

    weight_file = _find_local_jina_weight_file(model_dir)
    if weight_file is None:
        return None, notes

    try:
        from safetensors import safe_open

        with safe_open(str(weight_file), framework="pt", device="cpu") as handle:
            word_embeddings = handle.get_tensor("model.embeddings.word_embeddings.weight")
            emb_ln_weight = (
                handle.get_tensor("model.emb_ln.weight")
                if "model.emb_ln.weight" in handle.keys()
                else None
            )
            emb_ln_bias = (
                handle.get_tensor("model.emb_ln.bias")
                if "model.emb_ln.bias" in handle.keys()
                else None
            )
            token_type_embeddings = (
                handle.get_tensor("model.embeddings.token_type_embeddings.weight")
                if "model.embeddings.token_type_embeddings.weight" in handle.keys()
                else None
            )
    except Exception as exc:
        notes.append(f"Local Jina text-only pooler reconstruction failed: {exc}")
        return None, notes

    projection_dim = int(word_embeddings.shape[-1])
    config_path = model_dir / "config.json"
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    pad_token_id = 0
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            projection_dim = int(config.get("projection_dim", projection_dim) or projection_dim)
            pad_token_id = int(config.get("pad_token_id", pad_token_id) or pad_token_id)
        except Exception:
            pass
    if tokenizer_config_path.is_file():
        try:
            tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            pad_token_id = int(tokenizer_config.get("pad_token_id", pad_token_id) or pad_token_id)
        except Exception:
            pass

    model = _LocalJinaTextPooledEncoder(
        word_embeddings=word_embeddings,
        emb_ln_weight=emb_ln_weight,
        emb_ln_bias=emb_ln_bias,
        token_type_embeddings=token_type_embeddings,
        projection_dim=projection_dim,
        pad_token_id=pad_token_id,
    )
    notes.append(
        "Loaded stable local Jina text-only pooled encoder from safetensors "
        f"({weight_file.name}, dim={projection_dim})"
    )
    return model, notes


def _load_local_jina_clip_model(
    model_dir: Path,
    dtype: torch.dtype,
    *,
    trust_remote_code: bool = False,
) -> Any:
    """Load the local Jina CLIP snapshot via a single package instance.

    `AutoModel.from_pretrained()` routes custom code through Transformers'
    dynamic-module cache. With this Jina CLIP snapshot, the config class and
    model class can end up imported from different cached package instances,
    which breaks its explicit `isinstance()` checks. Loading the local package
    directly keeps config/model types in the same module tree.
    """
    _ensure_transformers_clip_compat()

    package_parent = str(model_dir.parent)
    package_name = model_dir.name
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)

    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]
    importlib.invalidate_caches()

    config_module = importlib.import_module(f"{package_name}.configuration_clip")
    model_module = importlib.import_module(f"{package_name}.modeling_clip")
    jina_config_cls = getattr(config_module, "JinaCLIPConfig")
    jina_model_cls = getattr(model_module, "JinaCLIPModel")

    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    config = jina_config_cls.from_pretrained(str(model_dir), **kwargs)
    try:
        model = jina_model_cls.from_pretrained(
            str(model_dir),
            config=config,
            torch_dtype=dtype,
            **kwargs,
        )
    except TypeError:
        model = jina_model_cls.from_pretrained(str(model_dir), config=config, **kwargs)
        if isinstance(model, torch.nn.Module):
            model.to(dtype=dtype)
    return model


def _prepare_jina_clip_runtime_dir(model_dir: Path) -> tuple[Path, list[str]]:
    """Create a runtime overlay dir with recovered Jina CLIP support modules if needed.

    The local Newbie model snapshot sometimes contains ``modeling_clip.py`` but omits
    companion support files like ``eva_model.py`` / ``hf_model.py``. Rather than mutate
    the model directory, we stage an ignored overlay under ``backend/tmp/model_overlays``
    and point Transformers at that directory for loading.
    """
    notes: list[str] = []
    required = ("eva_model.py", "hf_model.py")
    support_missing = [name for name in required if not (model_dir / name).is_file()]
    config_path = model_dir / "config.json"
    config_readable = _file_readable(config_path)
    if not support_missing and config_readable:
        return model_dir, notes

    support_cache = _find_jina_clip_support_cache()
    if support_cache is None and support_missing and config_readable:
        notes.append(
            "Jina CLIP support recovery unavailable; missing files: "
            + ", ".join(support_missing)
        )
        return model_dir, notes

    overlay_name = hashlib.sha1(str(model_dir).encode("utf-8")).hexdigest()[:16]
    overlay_dir = _runtime_model_overlay_root() / f"jina_clip_{overlay_name}"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    skipped_unreadable: list[str] = []
    for item in model_dir.iterdir():
        if item.is_file():
            if item.name == "config.json" and not config_readable:
                skipped_unreadable.append(item.name)
                continue
            try:
                _best_effort_link_or_copy(item, overlay_dir / item.name)
            except OSError as exc:
                skipped_unreadable.append(f"{item.name} ({exc})")

    standard_weight = overlay_dir / "model.safetensors"
    if not standard_weight.exists():
        for candidate_name in ("jina-clip-v2.safetensors",):
            candidate = overlay_dir / candidate_name
            if candidate.is_file():
                _best_effort_link_or_copy(candidate, standard_weight)
                notes.append(
                    f"Created runtime alias model.safetensors -> {candidate.name} for Jina CLIP"
                )
                break

    if support_cache is not None:
        for support_name in _JINA_CLIP_SUPPORT_FILES:
            src = support_cache / support_name
            dst = overlay_dir / support_name
            if src.is_file() and not dst.exists():
                _best_effort_link_or_copy(src, dst)

    overlay_config = overlay_dir / "config.json"
    if not overlay_config.is_file():
        if support_cache is not None and (support_cache / "config.json").is_file():
            _best_effort_link_or_copy(support_cache / "config.json", overlay_config)
            notes.append("Recovered unreadable Jina CLIP config.json from local support cache")
        else:
            _write_minimal_jina_clip_config(overlay_config, model_dir)
            notes.append("Synthesized minimal Jina CLIP config.json for runtime overlay recovery")

    if skipped_unreadable:
        notes.append(
            "Skipped unreadable Jina CLIP files while staging runtime overlay: "
            + ", ".join(skipped_unreadable)
        )

    if all((overlay_dir / name).is_file() for name in required):
        notes.append(
            f"Recovered missing Jina CLIP support files via runtime overlay: {overlay_dir}"
        )
        return overlay_dir, notes

    missing = [name for name in required if not (overlay_dir / name).is_file()]
    if support_missing:
        notes.append(
            "Jina CLIP support recovery incomplete; still missing: " + ", ".join(missing)
        )
    return overlay_dir, notes


def _build_emergency_newbie_clip_fallback(
    *,
    max_token_length: int,
    notes: list[str],
) -> tuple[torch.nn.Module, Any, list[str]]:
    """Return a minimal tokenizer/encoder pair for safe-fallback training only."""
    fallback_notes = list(notes)
    fallback_notes.append(
        "Activated emergency Newbie CLIP safe fallback: unreadable local CLIP assets were replaced "
        "with a deterministic minimal tokenizer + pooled encoder for runtime continuity."
    )
    tokenizer = _EmergencyNewbieClipTokenizer(model_max_length=max_token_length)
    model = _EmergencyNewbieClipEncoder(
        projection_dim=int(getattr(NEWBIE_CONTRACT, "clip_projection_dim", 1024) or 1024)
    )
    return model, tokenizer, fallback_notes

def _detect_directory_layout(path: Path) -> Dict[str, bool]:
    """Probe a diffusers directory and return which subfolders exist."""
    subfolders = {
        "unet": (path / "unet").is_dir(),
        "vae": (path / "vae").is_dir(),
        "text_encoder": (path / "text_encoder").is_dir(),
        "text_encoder_2": (path / "text_encoder_2").is_dir(),
        "tokenizer": (path / "tokenizer").is_dir(),
        "tokenizer_2": (path / "tokenizer_2").is_dir(),
        "scheduler": (path / "scheduler").is_dir(),
        "transformer": (path / "transformer").is_dir(),
        "clip_model": (path / "clip_model").is_dir(),
    }
    return subfolders


def _is_sdxl_shaped(layout: Dict[str, bool]) -> bool:
    """Return True if the directory looks like a standard SDXL layout."""
    return (
        layout["unet"]
        and layout["vae"]
        and layout["text_encoder"]
        and layout["text_encoder_2"]
        and layout["tokenizer"]
        and layout["tokenizer_2"]
        and layout["scheduler"]
    )


def _is_native_newbie_layout(layout: Dict[str, bool]) -> bool:
    """Return True if the directory looks like the observed Newbie bundle."""
    return (
        layout["transformer"]
        and layout["text_encoder"]
        and layout["clip_model"]
        and layout["vae"]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_newbie(
    diffusers_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    lora_target: str = "minimal",
    gemma3_prompt: str = "",
    use_flash_attn2: bool = True,
) -> LoadedModel:
    """Load a Newbie model from a diffusers-format directory.

    Parameters
    ----------
    diffusers_path:
        Path to the diffusers directory containing unet/, vae/,
        text_encoder/, etc.
    device:
        Target device for the loaded tensors.
    dtype:
        Floating-point dtype for model weights.
    lora_target:
        LoRA target preset name (minimal / balanced / full).  Stored as
        metadata on the returned model for downstream use.
    gemma3_prompt:
        Gemma3 chat-template prompt string.  Stored as metadata; the
        actual Gemma3 text-encoder integration is a TODO.
    use_flash_attn2:
        Whether to prefer flash-attention-2 when building the runtime
        optimization plan.

    Returns
    -------
    LoadedModel
        A ``LoadedModel`` instance with ``model_arch="newbie"``.

    Raises
    ------
    FileNotFoundError
        If *diffusers_path* does not exist or is not a directory.
    """

    path = Path(diffusers_path)
    if not path.is_dir():
        raise FileNotFoundError(
            f"Newbie diffusers directory not found: {path}\n"
            "Expected a diffusers-format directory containing unet/, vae/, "
            "text_encoder/, tokenizer/, scheduler/ subfolders."
        )

    # ── log contract audit ────────────────────────────────────────────────
    logger.info("Newbie loader: contract assumptions:\n%s", audit_newbie_contract())

    # ── probe directory ───────────────────────────────────────────────────
    layout = _detect_directory_layout(path)
    logger.info("Newbie directory layout detected: %s", layout)

    scaffold_mode = _is_sdxl_shaped(layout)
    native_layout = _is_native_newbie_layout(layout)
    if native_layout:
        logger.info(
            "Newbie native layout detected (transformer/text_encoder/clip_model/vae)."
        )
    if scaffold_mode:
        logger.warning(
            "Newbie directory looks SDXL-shaped.  Loading in scaffold mode "
            "(SDXL internals with model_arch='newbie').  This is expected "
            "for Phase-1 testing but does NOT exercise any Newbie-specific "
            "text encoding or UNet architecture."
        )
    elif native_layout:
        raise NotImplementedError(
            "Detected a native Newbie bundle (transformer/text_encoder/clip_model/vae). "
            "The Warehouse NextDiT-family loader is not wired yet, and this backend "
            "will not fall back to pretending that bundle is SDXL."
        )

    # ── resolve LoRA targets ──────────────────────────────────────────────
    unet_targets, te_targets = get_newbie_targets(lora_target)
    logger.info(
        "Newbie LoRA target preset '%s': %d UNet modules, %d TE modules",
        lora_target, len(unet_targets), len(te_targets),
    )

    # ── load components ───────────────────────────────────────────────────
    from diffusers import AutoencoderKL, DDPMScheduler
    from diffusers import UNet2DConditionModel
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )

    # UNet
    if not layout["unet"]:
        raise FileNotFoundError(
            f"Newbie directory missing unet/ subfolder: {path}"
        )
    logger.info("Loading Newbie UNet from %s/unet", path)
    unet = UNet2DConditionModel.from_pretrained(
        path, subfolder="unet", torch_dtype=dtype,
    )

    # VAE
    if not layout["vae"]:
        raise FileNotFoundError(
            f"Newbie directory missing vae/ subfolder: {path}"
        )
    logger.info("Loading Newbie VAE from %s/vae", path)
    vae = AutoencoderKL.from_pretrained(
        path, subfolder="vae", torch_dtype=dtype,
    )

    # Text encoder 1 (always CLIP)
    if not layout["text_encoder"]:
        raise FileNotFoundError(
            f"Newbie directory missing text_encoder/ subfolder: {path}"
        )
    logger.info("Loading Newbie text_encoder from %s/text_encoder", path)
    text_encoder_1 = CLIPTextModel.from_pretrained(
        path, subfolder="text_encoder", torch_dtype=dtype,
    )

    # Text encoder 2 (CLIP or Gemma3 -- see TODO below)
    text_encoder_2 = None
    tokenizer_2 = None
    if layout["text_encoder_2"]:
        # TODO: If Newbie's text_encoder_2 is a Gemma3 model, it should be
        # loaded with the appropriate Gemma3 class instead of
        # CLIPTextModelWithProjection.  For now we load it as CLIP since
        # that is what the Phase-1 scaffold expects.
        logger.info("Loading Newbie text_encoder_2 from %s/text_encoder_2", path)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            path, subfolder="text_encoder_2", torch_dtype=dtype,
        )
    else:
        logger.warning(
            "Newbie directory has no text_encoder_2/ subfolder.  "
            "Proceeding with a single text encoder.  This may indicate "
            "the checkpoint uses a non-SDXL text encoding path."
        )

    # Tokenizer 1
    if not layout["tokenizer"]:
        raise FileNotFoundError(
            f"Newbie directory missing tokenizer/ subfolder: {path}"
        )
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        path, subfolder="tokenizer",
    )

    # Tokenizer 2
    if layout["tokenizer_2"]:
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer_2",
        )

    # Scheduler
    if not layout["scheduler"]:
        raise FileNotFoundError(
            f"Newbie directory missing scheduler/ subfolder: {path}"
        )
    scheduler = DDPMScheduler.from_pretrained(
        path, subfolder="scheduler",
    )

    # ── flash-attn hint ───────────────────────────────────────────────────
    if use_flash_attn2:
        logger.info(
            "Newbie flash-attn2 hint is set.  The runtime optimization "
            "plan builder should honour this via config.newbie_use_flash_attn2."
        )

    # ── Gemma3 prompt hint ────────────────────────────────────────────────
    if gemma3_prompt:
        logger.info(
            "Newbie Gemma3 prompt hint received (%d chars).  "
            "This is stored as metadata only; the actual Gemma3 text "
            "encoder integration is not yet implemented.",
            len(gemma3_prompt),
        )

    # ── assemble LoadedModel ──────────────────────────────────────────────
    model = LoadedModel(
        unet=unet,
        text_encoder_1=text_encoder_1,
        text_encoder_2=text_encoder_2,
        vae=vae,
        tokenizer_1=tokenizer_1,
        tokenizer_2=tokenizer_2,
        noise_scheduler=scheduler,
        model_arch="newbie",
    )

    # Attach Newbie-specific metadata as attributes.  The LoadedModel
    # dataclass allows arbitrary attributes via __dict__.
    model.newbie_lora_target = lora_target
    model.newbie_unet_targets = unet_targets
    model.newbie_te_targets = te_targets
    model.newbie_gemma3_prompt = gemma3_prompt
    model.newbie_use_flash_attn2 = use_flash_attn2
    model.newbie_scaffold_mode = scaffold_mode
    model.newbie_native_layout_detected = native_layout
    model.newbie_native_conditioning_ready = False
    model.newbie_transport_ready = False
    model.newbie_forward_smoke_passed = False
    model.newbie_gradient_smoke_passed = False

    logger.info(
        "Newbie model loaded successfully.  scaffold=%s, "
        "lora_target=%s, dual_encoders=%s",
        scaffold_mode, lora_target, text_encoder_2 is not None,
    )

    return model


class _NextDiTBlock(torch.nn.Module):
    """Single DiT transformer block reconstructed from state dict keys.

    Contains nn.LayerNorm, nn.Linear (attention Q/K/V/out, FFN, adaLN
    modulation) sub-modules so that LoRA injection can discover nn.Linear
    leaves via ``named_modules()``.
    """

    def __init__(self, submodules: Dict[str, torch.nn.Module]):
        super().__init__()
        for name, mod in submodules.items():
            self.add_module(name, mod)


class _NextDiTWrapper(torch.nn.Module):
    """Reconstructed nn.Module tree for the NextDiT transformer.

    Builds a proper nested module hierarchy from a flat state dict so that:
    - ``named_modules()`` exposes real ``nn.Linear`` leaves for LoRA injection
    - ``forward()`` runs a DiT-style denoising pass with attention, FFN,
      LayerNorm, and AdaLN modulation
    - ``enable_gradient_checkpointing()`` is supported for training prep
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self._gradient_checkpointing = False
        self._attention_backend = "sdpa"
        self._attention_split_chunks = 0
        self._amd_sdpa_slice_trigger_gb = 0.0
        self._amd_sdpa_slice_target_gb = 0.0
        self._attention_early_deletion = False
        self._newbie_block_checkpointing_mode = "off"
        self.patch_size = 1
        self.in_channels = 16
        self._build_module_tree(state_dict)

    # ── Module tree construction ──────────────────────────────────────

    @staticmethod
    def _new_frozen_linear(in_features: int, out_features: int, *, bias: bool) -> torch.nn.Linear:
        linear_cls = torch.nn.Linear
        try:
            from .native_unet.weight_residency import LulynxManagedLinear

            linear_cls = LulynxManagedLinear
        except Exception:
            pass
        return linear_cls(in_features, out_features, bias=bias)

    def _build_module_tree(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Parse state dict keys and build nested nn.Module hierarchy.

        Creates nn.Linear for 2D weights, nn.LayerNorm for 1D norm weights,
        and registers non-norm 1D / 3D+ weights as flat parameters.
        """
        # Step 1: collect leaf modules — map "prefix" → (weight, bias)
        leaves: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in state_dict.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2 and parts[1] in ("weight", "bias"):
                prefix = parts[0]
                leaves.setdefault(prefix, {})[parts[1]] = tensor
            else:
                safe = key.replace(".", "_")
                self.register_parameter(safe, torch.nn.Parameter(tensor, requires_grad=False))

        # Step 2: create nn.Linear / nn.LayerNorm for each leaf prefix
        modules: Dict[str, torch.nn.Module] = {}
        for prefix, tensors in leaves.items():
            weight = tensors.get("weight")
            if weight is None:
                continue
            bias = tensors.get("bias")

            if weight.dim() == 1 and "norm" in prefix.lower():
                # LayerNorm: 1D weight with "norm" in key
                norm = torch.nn.LayerNorm(weight.shape[0])
                norm.weight.data.copy_(weight)
                if bias is not None:
                    norm.bias.data.copy_(bias)
                norm.requires_grad_(False)
                modules[prefix] = norm
            elif weight.dim() == 2:
                # Linear: 2D weight
                out_features, in_features = weight.shape[0], weight.shape[1]
                linear = self._new_frozen_linear(in_features, out_features, bias=bias is not None)
                linear.weight.data.copy_(weight)
                if bias is not None:
                    linear.bias.data.copy_(bias)
                linear.requires_grad_(False)
                modules[prefix] = linear
            else:
                # Conv / embed: register as flat parameters
                safe = prefix.replace(".", "_") + "_weight"
                self.register_parameter(safe, torch.nn.Parameter(weight, requires_grad=False))
                if bias is not None:
                    safe_b = prefix.replace(".", "_") + "_bias"
                    self.register_parameter(safe_b, torch.nn.Parameter(bias, requires_grad=False))

        # Step 3: build nested module tree from prefix paths
        root_children: Dict[str, Any] = {}
        for prefix, module in modules.items():
            parts = prefix.split(".")
            current = root_children
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = module

        # Step 4: recursively build nn.Module tree
        self._block_modules: list = []
        for name, subtree in root_children.items():
            if isinstance(subtree, torch.nn.Module):
                mod = subtree
            else:
                mod = self._build_subtree(subtree, name)
            self.add_module(name, mod)
        self._block_modules = self._collect_dit_blocks()

        # Detect config hints from key names
        all_keys = set(state_dict.keys())
        self._has_x_embedder = any("x_embedder" in k for k in all_keys)
        self._has_t_embedder = any("t_embedder" in k for k in all_keys)
        self._has_final_layer = any("final_layer" in k for k in all_keys)
        self._has_pos_embed = any("pos_embed" in k for k in all_keys)

    def _build_subtree(self, tree: dict, name_hint: str = "") -> torch.nn.Module:
        """Recursively build an nn.Module from a nested dict.

        Leaf values are nn.Module instances (nn.Linear, nn.LayerNorm);
        dicts become nn.Module containers.
        """
        if isinstance(tree, torch.nn.Module):
            return tree

        has_module = any(isinstance(v, torch.nn.Module) for v in tree.values())
        has_dict = any(isinstance(v, dict) for v in tree.values())

        if has_module and not has_dict:
            return _NextDiTBlock({k: v for k, v in tree.items() if isinstance(v, torch.nn.Module)})

        mod = torch.nn.Module()
        sub_list = []
        for key, value in tree.items():
            if isinstance(value, torch.nn.Module):
                mod.add_module(key, value)
            elif isinstance(value, dict):
                child = self._build_subtree(value, key)
                mod.add_module(key, child)
                sub_list.append(child)

        if sub_list:
            mod._submodule_list = sub_list
        return mod

    def _collect_dit_blocks(self) -> list[torch.nn.Module]:
        """Collect block containers from known NextDiT-style roots."""

        block_roots = (
            "layers",
            "context_refiner",
            "transformer_blocks",
            "blocks",
            "double_blocks",
            "single_blocks",
        )
        blocks: list[torch.nn.Module] = []
        for root_name in block_roots:
            root = self._modules.get(root_name)
            if root is None:
                continue
            blocks.extend(self._collect_blocks_from_container(root))
        return blocks

    def _collect_blocks_from_container(self, module: torch.nn.Module) -> list[torch.nn.Module]:
        if self._looks_like_dit_block(module):
            return [module]

        children = list(module.named_children())
        if not children:
            return []

        def sort_key(item: tuple[str, torch.nn.Module]) -> tuple[int, Any]:
            name, _ = item
            return (0, int(name)) if name.isdigit() else (1, name)

        blocks: list[torch.nn.Module] = []
        for _, child in sorted(children, key=sort_key):
            blocks.extend(self._collect_blocks_from_container(child))
        return blocks

    @staticmethod
    def _looks_like_dit_block(module: torch.nn.Module) -> bool:
        child_names = set(module._modules.keys())
        return bool(
            {"attention", "attn"} & child_names
            or {"feed_forward", "ff"} & child_names
        )

    # ── Text-embed projection (eager init) ────────────────────────────

    def ensure_text_embed_projection(self, text_embed_dim: int) -> None:
        """Create a frozen linear projection for text_embeds if dimensions mismatch.

        Called at load time (before ``prepare_for_training``) so that the
        layer is visible to parameter-collection sweeps.  Frozen by default
        — the trainer can ``requires_grad_(True)`` it for full-finetune.
        """
        conditioning_dim = self._detect_conditioning_dim()
        if text_embed_dim == conditioning_dim:
            return  # direct addition in forward, no projection needed
        if getattr(self, "_text_embed_proj", None) is not None:
            return  # already created
        proj = self._new_frozen_linear(text_embed_dim, conditioning_dim, bias=False)
        torch.nn.init.eye_(proj.weight[:, :min(text_embed_dim, conditioning_dim)])
        proj.requires_grad_(False)  # frozen by default
        self._text_embed_proj = proj

    # ── Gradient checkpointing ────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = True
        self._newbie_block_checkpointing_mode = "block"

    def disable_gradient_checkpointing(self) -> None:
        self._gradient_checkpointing = False
        self._newbie_block_checkpointing_mode = "off"

    def set_newbie_block_checkpointing(self, enabled: bool, mode: str = "block") -> Dict[str, Any]:
        normalized = str(mode or "block").strip().lower().replace("-", "_")
        active = bool(enabled) and normalized in {"", "block", "selective"}
        self._gradient_checkpointing = active
        self._newbie_block_checkpointing_mode = "selective" if active and normalized == "selective" else "block" if active else "off"
        block_count = len(self._block_modules or [])
        return {
            "enabled": active,
            "mode": self._newbie_block_checkpointing_mode,
            "block_count": block_count,
            "checkpointed_blocks": block_count if active else 0,
        }

    def get_newbie_block_checkpointing_profile(self) -> Dict[str, Any]:
        block_count = len(self._block_modules or [])
        active = bool(getattr(self, "_gradient_checkpointing", False))
        return {
            "enabled": active,
            "mode": str(getattr(self, "_newbie_block_checkpointing_mode", "block" if active else "off") or "off"),
            "block_count": block_count,
            "checkpointed_blocks": block_count if active else 0,
        }

    # ── Forward pass ──────────────────────────────────────────────────

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Run a DiT-style denoising pass.

        Returns a namespace with ``.sample`` attribute matching input shape.
        """
        from types import SimpleNamespace

        B = sample.shape[0]
        dtype = sample.dtype
        orig_dim = sample.dim()

        hidden_dim = self._detect_hidden_dim()
        conditioning_dim = self._detect_conditioning_dim()

        # ── Patchify ──
        if orig_dim == 4:
            B, C, H, W = sample.shape
            self._orig_channels = C
            x_embedder_w = getattr(self, "x_embedder_proj_weight", None)
            x_embedder_b = getattr(self, "x_embedder_proj_bias", None)
            if x_embedder_w is not None and x_embedder_w.dim() == 4:
                # Standard patchify conv: stride = kernel, padding = 0
                kh = x_embedder_w.shape[2]
                stride = kh
                x = torch.nn.functional.conv2d(
                    sample, x_embedder_w, bias=x_embedder_b,
                    stride=stride, padding=0,
                )
                _, ch, h2, w2 = x.shape
                x = x.reshape(B, ch, h2 * w2).permute(0, 2, 1)
                self._patch_h, self._patch_w = h2, w2
                self._patch_size = kh
            else:
                patch_size = max(int(getattr(self, "patch_size", 1) or 1), 1)
                x_embedder = getattr(self, "x_embedder", None)
                if (
                    isinstance(x_embedder, torch.nn.Linear)
                    and H % patch_size == 0
                    and W % patch_size == 0
                    and x_embedder.in_features == C * patch_size * patch_size
                ):
                    patches = torch.nn.functional.unfold(
                        sample,
                        kernel_size=patch_size,
                        stride=patch_size,
                    ).transpose(1, 2)
                    x = x_embedder(patches)
                    self._patch_h, self._patch_w = H // patch_size, W // patch_size
                    self._patch_size = patch_size
                else:
                    x = sample.reshape(B, C, H * W).permute(0, 2, 1)
                    self._patch_h, self._patch_w = H, W
                    self._patch_size = 1
                    if x.shape[-1] != hidden_dim:
                        if x.shape[-1] < hidden_dim:
                            x = torch.nn.functional.pad(x, (0, hidden_dim - x.shape[-1]))
                        else:
                            x = x[:, :, :hidden_dim]
        else:
            x = sample
            if x.dim() == 2:
                x = x.unsqueeze(1)
            self._patch_h, self._patch_w = x.shape[1], 1
        x = self._bounded_rms(x, max_rms=8.0, clamp=64.0)

        # ── Timestep + pooled-text conditioning ──
        time_cond = self._build_timestep_condition(
            timestep=timestep,
            batch_size=B,
            conditioning_dim=conditioning_dim,
            dtype=dtype,
            device=sample.device,
        )

        text_embeds = None
        if added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs.get("text_embeds")
        text_cond = None
        if text_embeds is not None:
            text_cond = text_embeds.float().to(dtype=dtype).reshape(B, -1)
            clip_text_proj = self._find_linear_leaf(getattr(self, "clip_text_pooled_proj", None))
            if (
                clip_text_proj is not None
                and text_cond.shape[-1] == clip_text_proj.in_features
            ):
                text_cond = clip_text_proj(text_cond)
            elif text_cond.shape[-1] == conditioning_dim:
                pass
            else:
                proj = getattr(self, "_text_embed_proj", None)
                if proj is not None and isinstance(proj, torch.nn.Linear):
                    text_cond = proj(text_cond)
                else:
                    logger.warning(
                        "text_embeds dim (%d) != conditioning_dim (%d) but no "
                        "_text_embed_proj was prepared — conditioning dropped. "
                        "Call ensure_text_embed_projection() at load time.",
                        text_cond.shape[-1], conditioning_dim,
                    )
                    text_cond = None

        t_emb = time_cond
        time_text_embed = self._find_linear_leaf(getattr(self, "time_text_embed", None))
        if time_text_embed is not None:
            time_text_in = torch.cat(
                (
                    time_cond,
                    text_cond if text_cond is not None else torch.zeros_like(time_cond),
                ),
                dim=-1,
            )
            if time_text_in.shape[-1] != time_text_embed.in_features:
                if time_text_in.shape[-1] > time_text_embed.in_features:
                    time_text_in = time_text_in[:, :time_text_embed.in_features]
                else:
                    time_text_in = torch.nn.functional.pad(
                        time_text_in,
                        (0, time_text_embed.in_features - time_text_in.shape[-1]),
                    )
            t_emb = time_text_embed(time_text_in)
        elif text_cond is not None:
            t_emb = time_cond + text_cond
        t_emb = self._bounded_rms(t_emb, max_rms=8.0, clamp=64.0)

        # ── Encoder conditioning (cross-attn style) ──
        enc_condition = None
        if encoder_hidden_states is not None and encoder_hidden_states.dim() >= 2:
            enc = encoder_hidden_states.float().to(dtype=dtype)
            if enc.dim() == 3:
                enc = enc.mean(dim=1)
            if enc.shape[-1] == hidden_dim:
                x = x + enc.unsqueeze(1) * 0.01
            else:
                enc_condition = enc
            x = self._bounded_rms(x, max_rms=8.0, clamp=64.0)

        # ── Transformer blocks ──
        blocks = self._block_modules if self._block_modules else []
        try:
            from .spectrum_probe import has_spectrum_step_context, observe_block_call
            from .smoothcache import (
                has_smoothcache_step_context,
                observe_block_call as observe_smoothcache_block_call,
            )
            from .dit_compute_reducer_seam import get_active_compute_reducer_seam
        except ImportError:  # pragma: no cover - direct-file smoke fallback.
            from core.lulynx_trainer.spectrum_probe import has_spectrum_step_context, observe_block_call
            from core.lulynx_trainer.smoothcache import (
                has_smoothcache_step_context,
                observe_block_call as observe_smoothcache_block_call,
            )
            from core.lulynx_trainer.dit_compute_reducer_seam import get_active_compute_reducer_seam

        use_checkpoint = bool(self._gradient_checkpointing and self.training)
        reducer = None if use_checkpoint else get_active_compute_reducer_seam()
        if reducer is not None and hasattr(reducer, "set_total_blocks"):
            try:
                reducer.set_total_blocks(len(blocks))
            except TypeError:
                pass

        for block_index, block in enumerate(blocks):
            if has_spectrum_step_context():
                observe_block_call(block_index=block_index)
            if has_smoothcache_step_context():
                observe_smoothcache_block_call(block_index=block_index)
            if use_checkpoint:
                checkpoint_kwargs = {"use_reentrant": False, "preserve_rng_state": False}
                if str(getattr(self, "_newbie_block_checkpointing_mode", "") or "") == "selective":
                    try:
                        from .checkpoint_policy import build_selective_checkpoint_context_fn
                    except Exception:
                        from checkpoint_policy import build_selective_checkpoint_context_fn
                    context_fn = build_selective_checkpoint_context_fn("balanced")
                    if context_fn is not None:
                        checkpoint_kwargs["context_fn"] = context_fn
                x = torch.utils.checkpoint.checkpoint(
                    self._run_dit_block,
                    block,
                    x,
                    t_emb,
                    **checkpoint_kwargs,
                )
            elif reducer is not None:
                x = reducer.run_block(
                    lambda tokens, blk=block: self._run_dit_block(blk, tokens, t_emb),
                    block_index,
                    x,
                )
            else:
                x = self._run_dit_block(block, x, t_emb)

        # ── Final layer ──
        final_layer = getattr(self, "final_layer", None)
        if final_layer is not None:
            final_mod = self._find_submodule(final_layer, "adaLN_modulation")
            final_mod_linear = self._find_linear_leaf(final_mod) if final_mod is not None else None
            final_norm = self._find_submodule(final_layer, "norm_final")
            if final_norm is None:
                final_norm = self._find_submodule(final_layer, "norm")
            if final_norm is not None:
                x = final_norm(x)
            if final_mod_linear is not None:
                final_cond = final_mod_linear(t_emb)
                if final_cond.shape[-1] == 2 * x.shape[-1]:
                    shift, scale = final_cond.chunk(2, dim=-1)
                    shift = self._bounded_rms(shift, max_rms=4.0, clamp=16.0)
                    scale = torch.clamp(torch.nan_to_num(scale), min=-4.0, max=4.0)
                    x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                elif final_cond.shape[-1] == x.shape[-1]:
                    final_cond = torch.clamp(torch.nan_to_num(final_cond), min=-4.0, max=4.0)
                    x = x * (1 + final_cond.unsqueeze(1))
                x = self._bounded_rms(x, max_rms=8.0, clamp=64.0)
            final_linear = getattr(final_layer, "linear", None)
            if final_linear is not None and isinstance(final_linear, torch.nn.Linear):
                x = final_linear(x)

        # ── Unpatchify ──
        if orig_dim == 4:
            c_out = x.shape[-1]
            h2, w2 = self._patch_h, self._patch_w
            patch_size = max(int(getattr(self, "_patch_size", 1) or 1), 1)
            orig_channels = int(getattr(self, "_orig_channels", C))
            expected_patch_dim = orig_channels * patch_size * patch_size
            if (
                patch_size > 1
                and c_out == expected_patch_dim
                and h2 * w2 == x.shape[1]
            ):
                x = torch.nn.functional.fold(
                    x.transpose(1, 2),
                    output_size=(H, W),
                    kernel_size=patch_size,
                    stride=patch_size,
                )
            elif h2 * w2 == x.shape[1] and h2 > 0:
                x = x.permute(0, 2, 1).reshape(B, c_out, h2, w2)
                if h2 != H or w2 != W:
                    x = torch.nn.functional.interpolate(
                        x, size=(H, W), mode="bilinear", align_corners=False,
                    )
            else:
                x = x.permute(0, 2, 1).reshape(B, c_out, H, W)

            if x.shape[1] != orig_channels:
                if x.shape[1] > orig_channels:
                    x = x[:, :orig_channels]
                else:
                    pad = orig_channels - x.shape[1]
                    x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))

        if enc_condition is not None:
            scale = enc_condition.mean(dim=-1, keepdim=True) * 0.01
            while scale.dim() < x.dim():
                scale = scale.unsqueeze(-1)
            x = x + scale.expand_as(x)
        x = self._bounded_rms(x, max_rms=4.0, clamp=32.0)

        return SimpleNamespace(sample=x)

    def _detect_hidden_dim(self) -> int:
        """Detect transformer hidden dim from x_embedder or attention layers."""
        xew = getattr(self, "x_embedder_proj_weight", None)
        if xew is not None and xew.dim() == 4:
            return xew.shape[0]
        for name, mod in self.named_modules():
            if isinstance(mod, torch.nn.Linear) and ("attn" in name or "attention" in name):
                return mod.in_features
        for _, mod in self.named_modules():
            if isinstance(mod, torch.nn.Linear):
                return mod.in_features
        return 768

    def _detect_conditioning_dim(self) -> int:
        """Detect the AdaLN conditioning width, which may differ from token dim."""
        for block in self._block_modules:
            adaLN_mod = self._find_submodule(block, "adaLN_modulation")
            adaLN_linear = self._find_linear_leaf(adaLN_mod) if adaLN_mod is not None else None
            if adaLN_linear is not None:
                return adaLN_linear.in_features

        final_layer = getattr(self, "final_layer", None)
        if final_layer is not None:
            final_mod = self._find_submodule(final_layer, "adaLN_modulation")
            final_mod_linear = self._find_linear_leaf(final_mod) if final_mod is not None else None
            if final_mod_linear is not None:
                return final_mod_linear.in_features

        time_text_embed = self._find_linear_leaf(getattr(self, "time_text_embed", None))
        if time_text_embed is not None:
            return time_text_embed.out_features

        return self._detect_hidden_dim()

    def _build_timestep_condition(
        self,
        *,
        timestep: torch.Tensor,
        batch_size: int,
        conditioning_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the AdaLN timestep condition expected by NextDiT weights."""

        t_embedder = getattr(self, "t_embedder", None)
        mlp = self._find_submodule(t_embedder, "mlp") if t_embedder is not None else None
        mlp0 = self._find_submodule(mlp, "0") if mlp is not None else None
        mlp2 = self._find_submodule(mlp, "2") if mlp is not None else None
        if isinstance(mlp0, torch.nn.Linear) and isinstance(mlp2, torch.nn.Linear):
            t = timestep.float().reshape(batch_size, -1)
            if t.shape[-1] != 1:
                t = t[:, :1]
            emb = self._sinusoidal_timestep_embedding(
                t.reshape(batch_size),
                mlp0.in_features,
                device=device,
            ).to(device=device, dtype=mlp0.weight.dtype)
            cond = mlp2(torch.nn.functional.silu(mlp0(emb)))
            if cond.shape[-1] == conditioning_dim:
                return cond.to(dtype=dtype)
            if cond.shape[-1] > conditioning_dim:
                return cond[:, :conditioning_dim].to(dtype=dtype)
            return torch.nn.functional.pad(
                cond,
                (0, conditioning_dim - cond.shape[-1]),
            ).to(dtype=dtype)

        # Fallback for incomplete tiny test fixtures: use bounded sinusoidal
        # features instead of raw 0..1000 timesteps to avoid AdaLN blow-ups.
        cond = self._sinusoidal_timestep_embedding(
            timestep.float().reshape(batch_size, -1)[:, 0],
            conditioning_dim,
            device=device,
        )
        return cond.to(dtype=dtype)

    @staticmethod
    def _sinusoidal_timestep_embedding(
        timesteps: torch.Tensor,
        dim: int,
        *,
        device: torch.device,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        if half <= 0:
            return torch.zeros((timesteps.shape[0], dim), device=device, dtype=torch.float32)
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.to(device=device, dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    @staticmethod
    def _bounded_rms(
        value: torch.Tensor,
        *,
        max_rms: float,
        clamp: float,
    ) -> torch.Tensor:
        """Keep reconstructed fallback activations finite without blocking grads."""

        value = torch.nan_to_num(value, nan=0.0, posinf=clamp, neginf=-clamp)
        if value.numel() == 0:
            return value
        if value.dim() <= 1:
            dims = (-1,)
        elif value.dim() == 2:
            dims = (-1,)
        else:
            dims = tuple(range(1, value.dim()))
        rms = value.detach().float().square().mean(dim=dims, keepdim=True).sqrt()
        scale = torch.clamp(rms / float(max_rms), min=1.0).to(device=value.device, dtype=value.dtype)
        value = value / scale
        return torch.clamp(value, min=-float(clamp), max=float(clamp))

    # ── DiT block forward ────────────────────────────────────────────

    def _run_dit_block(
        self,
        block: torch.nn.Module,
        x: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single DiT transformer block.

        Implements the standard DiT block pattern:
        1. AdaLN modulation from timestep embedding
        2. LayerNorm → modulate → Q/K/V attention → output proj → gate → residual
        3. LayerNorm → modulate → FFN (linear → GELU → linear) → gate → residual
        """
        controller = getattr(self, "_lulynx_dit_prefetch_controller", None)
        if controller is not None:
            try:
                controller.before_block_module(block, x, t_emb)
            except Exception:
                pass

        B, N, D = x.shape

        # ── AdaLN modulation ──
        shift1 = scale1 = gate1 = shift2 = scale2 = gate2 = None
        adaLN_mod = self._find_submodule(block, "adaLN_modulation")
        if adaLN_mod is not None and t_emb is not None:
            # adaLN_modulation is typically a container with child "1" (nn.Linear)
            adaLN_linear = self._find_linear_leaf(adaLN_mod)
            if adaLN_linear is not None:
                modulation = adaLN_linear(t_emb)
                if modulation.shape[-1] == 6 * D:
                    shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
                elif modulation.shape[-1] == 4 * D:
                    scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
                    shift1 = torch.zeros_like(scale1)
                    shift2 = torch.zeros_like(scale2)
                    gate1 = gate1.tanh()
                    gate2 = gate2.tanh()
                elif modulation.shape[-1] >= 6:
                    shift1, scale1, gate1, shift2, scale2, gate2 = modulation.chunk(6, dim=-1)
                if shift1 is not None:
                    shift1 = self._bounded_rms(shift1, max_rms=4.0, clamp=16.0)
                if shift2 is not None:
                    shift2 = self._bounded_rms(shift2, max_rms=4.0, clamp=16.0)
                if scale1 is not None:
                    scale1 = torch.clamp(torch.nan_to_num(scale1), min=-4.0, max=4.0)
                if scale2 is not None:
                    scale2 = torch.clamp(torch.nan_to_num(scale2), min=-4.0, max=4.0)
                if gate1 is not None:
                    gate1 = torch.tanh(torch.nan_to_num(gate1))
                if gate2 is not None:
                    gate2 = torch.tanh(torch.nan_to_num(gate2))

        # ── Attention branch ──
        residual = x

        # Norm1
        norm1 = self._find_submodule(block, "norm1")
        if norm1 is None:
            norm1 = self._find_submodule(block, "attention_norm1")
        h = norm1(x) if norm1 is not None else x

        # AdaLN modulate
        if scale1 is not None:
            h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
            h = self._bounded_rms(h, max_rms=8.0, clamp=64.0)

        # Q/K/V projections
        attn = self._find_submodule(block, "attention")
        if attn is None:
            attn = self._find_submodule(block, "attn")
        if attn is not None:
            to_q = self._find_submodule(attn, "to_q")
            to_k = self._find_submodule(attn, "to_k")
            to_v = self._find_submodule(attn, "to_v")

            qkv = self._find_submodule(attn, "qkv")
            if self._is_linear_like(qkv):
                qkv_out = self._run_linear_like(qkv, h)
                q, k, v = self._split_qkv_projection(qkv_out, token_dim=D, attn_module=attn)
            elif (
                self._is_linear_like(to_q)
                and self._is_linear_like(to_k)
                and self._is_linear_like(to_v)
            ):
                q = self._run_linear_like(to_q, h)  # (B, N, D_q)
                k = self._run_linear_like(to_k, h)  # (B, N, D_k)
                v = self._run_linear_like(to_v, h)  # (B, N, D_v)
            else:
                q = k = v = None

            if q is not None and k is not None and v is not None:
                # Handle GQA: if Q/K dims differ, repeat K/V
                if q.shape[-1] != k.shape[-1]:
                    repeats = q.shape[-1] // k.shape[-1]
                    if repeats > 1 and q.shape[-1] % k.shape[-1] == 0:
                        k = k.repeat_interleave(repeats, dim=-1)
                        v = v.repeat_interleave(repeats, dim=-1)
                    else:
                        # Fallback: project K/V to Q dim
                        k = torch.nn.functional.linear(k, torch.eye(q.shape[-1], k.shape[-1], device=k.device, dtype=k.dtype))
                        v = torch.nn.functional.linear(v, torch.eye(q.shape[-1], v.shape[-1], device=v.device, dtype=v.dtype))

                # Reshape for multi-head attention
                q_norm = self._find_submodule(attn, "q_norm")
                k_norm = self._find_submodule(attn, "k_norm")
                if isinstance(q_norm, torch.nn.LayerNorm):
                    head_dim = int(q_norm.normalized_shape[0])
                elif isinstance(k_norm, torch.nn.LayerNorm):
                    head_dim = int(k_norm.normalized_shape[0])
                else:
                    head_dim = self._infer_head_dim(q.shape[-1])
                num_heads = max(q.shape[-1] // head_dim, 1)
                if num_heads > 1:
                    q = q.reshape(B, N, num_heads, head_dim).transpose(1, 2)
                    k = k.reshape(B, N, num_heads, head_dim).transpose(1, 2)
                    v = v.reshape(B, N, num_heads, head_dim).transpose(1, 2)
                    if q_norm is not None:
                        q = q_norm(q)
                    if k_norm is not None:
                        k = k_norm(k)
                    from .anima_attention import dit_attention

                    attn_out = dit_attention(
                        q,
                        k,
                        v,
                        backend=str(getattr(attn, "_attention_backend", getattr(self, "_attention_backend", "sdpa")) or "sdpa"),
                        split_chunks=int(getattr(attn, "_attention_split_chunks", getattr(self, "_attention_split_chunks", 0)) or 0),
                        amd_sdpa_slice_trigger_gb=float(
                            getattr(attn, "_amd_sdpa_slice_trigger_gb", getattr(self, "_amd_sdpa_slice_trigger_gb", 0.0)) or 0.0
                        ),
                        amd_sdpa_slice_target_gb=float(
                            getattr(attn, "_amd_sdpa_slice_target_gb", getattr(self, "_amd_sdpa_slice_target_gb", 0.0)) or 0.0
                        ),
                        early_delete=bool(
                            getattr(attn, "_attention_early_deletion", getattr(self, "_attention_early_deletion", False))
                        ),
                        sliding_window_size=int(
                            getattr(attn, "_attention_profile_window_size", getattr(self, "_attention_profile_window_size", 0)) or 0
                        ),
                        sliding_backend=str(
                            getattr(attn, "_attention_profile_backend", getattr(self, "_attention_profile_backend", "auto")) or "auto"
                        ),
                        sliding_torch_fallback_max_tokens=int(
                            getattr(
                                attn,
                                "_attention_profile_torch_max_tokens",
                                getattr(self, "_attention_profile_torch_max_tokens", 2048),
                            ) or 2048
                        ),
                        launcher_attention_backend=str(
                            getattr(
                                attn,
                                "_attention_profile_launcher_backend",
                                getattr(self, "_attention_profile_launcher_backend", getattr(attn, "_attention_backend", "sdpa")),
                            ) or "sdpa"
                        ),
                        flex_runtime_active=bool(
                            getattr(
                                attn,
                                "_attention_profile_flex_runtime_active",
                                getattr(self, "_attention_profile_flex_runtime_active", False),
                            )
                        ),
                    )
                    attn_out = attn_out.transpose(1, 2).reshape(B, N, -1)
                else:
                    # Single-head attention
                    scale = head_dim ** -0.5
                    attn_weights = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
                    attn_out = attn_weights @ v

                # Output projection
                to_out = self._find_submodule(attn, "out")
                if to_out is None:
                    to_out = self._find_submodule(attn, "to_out")
                if to_out is not None:
                    # to_out might be a container with child "0"
                    out_linear = self._find_submodule(to_out, "0")
                    if out_linear is None and self._is_linear_like(to_out):
                        out_linear = to_out
                    if self._is_linear_like(out_linear):
                        attn_out = self._run_linear_like(out_linear, attn_out)

                # AdaLN gate
                if gate1 is not None:
                    attn_out = attn_out * gate1.unsqueeze(1)
                attn_out = self._bounded_rms(attn_out, max_rms=8.0, clamp=64.0)

                x = residual + attn_out
                x = self._bounded_rms(x, max_rms=8.0, clamp=64.0)

        # ── FFN branch ──
        residual2 = x

        # Norm2
        norm2 = self._find_submodule(block, "norm2")
        if norm2 is None:
            norm2 = self._find_submodule(block, "ffn_norm1")
        h = norm2(x) if norm2 is not None else x

        # AdaLN modulate
        if scale2 is not None:
            h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
            h = self._bounded_rms(h, max_rms=8.0, clamp=64.0)

        # FFN: linear → GELU → linear
        ff = self._find_submodule(block, "feed_forward")
        if ff is None:
            ff = self._find_submodule(block, "ff")
        if ff is not None:
            w1 = self._find_submodule(ff, "w1")
            w2 = self._find_submodule(ff, "w2")
            w3 = self._find_submodule(ff, "w3")
            if (
                self._is_linear_like(w1)
                and self._is_linear_like(w2)
                and self._is_linear_like(w3)
            ):
                h = torch.nn.functional.silu(self._run_linear_like(w1, h)) * self._run_linear_like(w3, h)
                h = self._run_linear_like(w2, h)
            else:
                ff_layers = [m for m in ff.modules() if isinstance(m, torch.nn.Linear)]
                if len(ff_layers) < 2:
                    h = None
                else:
                    h = ff_layers[0](h)
                    h = torch.nn.functional.gelu(h)
                    h = ff_layers[1](h)

            if h is not None:
                if h.shape[-1] != residual2.shape[-1]:
                    if h.shape[-1] > residual2.shape[-1]:
                        h = h[..., :residual2.shape[-1]]
                    else:
                        h = torch.nn.functional.pad(h, (0, residual2.shape[-1] - h.shape[-1]))

                if gate2 is not None:
                    h = h * gate2.unsqueeze(1)
                h = self._bounded_rms(h, max_rms=8.0, clamp=64.0)

                x = residual2 + h
                x = self._bounded_rms(x, max_rms=8.0, clamp=64.0)

        return x

    def _split_qkv_projection(
        self,
        qkv_out: torch.Tensor,
        *,
        token_dim: int,
        attn_module: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused QKV output, honoring known GQA layouts when detectable."""
        total_dim = qkv_out.shape[-1]
        num_heads, num_kv_heads, head_dim = self._detect_attention_layout(
            attn_module=attn_module,
            token_dim=token_dim,
            fused_dim=total_dim,
        )

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        if q_dim + 2 * kv_dim == total_dim:
            q, k, v = torch.split(qkv_out, [q_dim, kv_dim, kv_dim], dim=-1)
            return q, k, v

        if total_dim % 3 == 0:
            return qkv_out.chunk(3, dim=-1)

        split = min(token_dim, total_dim)
        q = qkv_out[..., :split]
        k = qkv_out[..., :split]
        v = qkv_out[..., -split:]
        return q, k, v

    def _detect_attention_layout(
        self,
        *,
        attn_module: torch.nn.Module,
        token_dim: int,
        fused_dim: int,
    ) -> tuple[int, int, int]:
        """Detect `(num_heads, num_kv_heads, head_dim)` for fused attention."""
        cfg = getattr(self, "config", None)
        cfg_heads = getattr(cfg, "n_heads", None)
        cfg_kv_heads = getattr(cfg, "n_kv_heads", None)

        q_norm = self._find_submodule(attn_module, "q_norm")
        k_norm = self._find_submodule(attn_module, "k_norm")
        head_dim = None
        for norm in (q_norm, k_norm):
            if isinstance(norm, torch.nn.LayerNorm):
                head_dim = int(norm.normalized_shape[0])
                break

        if head_dim is None:
            inferred = self._infer_head_dim(token_dim)
            head_dim = inferred if inferred > 0 else token_dim

        num_heads = int(cfg_heads) if cfg_heads else max(token_dim // head_dim, 1)
        num_kv_heads = int(cfg_kv_heads) if cfg_kv_heads else num_heads

        if num_heads * head_dim > token_dim and head_dim > 0:
            num_heads = max(token_dim // head_dim, 1)
        max_kv_from_fused = max((fused_dim - token_dim) // (2 * head_dim), 1) if head_dim > 0 else num_kv_heads
        if num_kv_heads * head_dim * 2 + token_dim > fused_dim:
            num_kv_heads = max_kv_from_fused

        return num_heads, num_kv_heads, head_dim

    @staticmethod
    def _find_submodule(parent: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
        """Find a child module by name, handling numeric names like '0'."""
        mod = getattr(parent, name, None)
        if mod is not None:
            return mod
        for child_name, child in parent.named_children():
            if child_name == name:
                return child
        return None

    @staticmethod
    def _find_linear_leaf(module: Optional[torch.nn.Module]) -> Optional[torch.nn.Linear]:
        """Find the first nn.Linear leaf in a (possibly nested) module."""
        if module is None:
            return None
        if isinstance(module, torch.nn.Linear):
            return module
        for _, child in module.named_modules():
            if isinstance(child, torch.nn.Linear):
                return child
        return None

    @staticmethod
    def _is_linear_like(module: Optional[torch.nn.Module]) -> bool:
        """Return True for modules that behave like a linear projection."""
        return isinstance(module, torch.nn.Module) and callable(getattr(module, "forward", None))

    @staticmethod
    def _run_linear_like(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Call a linear-like module, including injected wrappers such as LoRALinear."""
        out = module(x)
        if not isinstance(out, torch.Tensor):
            raise RuntimeError(
                f"Expected tensor output from linear-like module {type(module).__name__}, "
                f"got {type(out)!r}"
            )
        return out

    @staticmethod
    def _infer_head_dim(total_dim: int, default: int = 64) -> int:
        """Infer per-head dimension from total projection dim."""
        for candidate in (64, 128, 80, 96, 48):
            if total_dim % candidate == 0 and total_dim // candidate >= 1:
                return candidate
        return total_dim  # single head fallback


def _load_transformer_native(
    transformer_path: str, dtype: torch.dtype, device: str,
    *,
    disable_mmap: bool = False,
) -> Tuple[Any, list]:
    """Load a transformer from a directory of safetensors files.

    Returns ``(model_or_wrapper, notes)``.
    """
    notes = []
    tpath = Path(transformer_path)

    if not tpath.exists():
        notes.append(f"Transformer path does not exist: {transformer_path}")
        return None, notes

    config: Dict[str, Any] = {}
    config_path = tpath / "config.json" if tpath.is_dir() else tpath.with_name("config.json")
    if not config_path.is_file() and tpath.is_file():
        sibling_config = tpath.parent / "transformer" / "config.json"
        if sibling_config.is_file():
            config_path = sibling_config
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            notes.append(f"Transformer config read failed: {exc}")

    # Collect state dict from safetensors files
    state_dict: Dict[str, torch.Tensor] = {}
    try:
        from core.lulynx_trainer.safetensors_loader import load_safetensors

        if tpath.is_dir():
            sf_files = sorted(tpath.glob("*.safetensors"))
            if not sf_files:
                # Try diffusers subfolder layout
                sf_files = sorted((tpath / "transformer").glob("*.safetensors"))
            if not sf_files and tpath.name == "transformer":
                # Packaged Newbie layout keeps config in transformer/ but the
                # transformer weights at the model root.
                sf_files = sorted(tpath.parent.glob("*.safetensors"))
            for sf in sf_files:
                sd = load_safetensors(str(sf), disable_mmap=disable_mmap)
                state_dict.update(sd)
        elif tpath.suffix == ".safetensors":
            state_dict = load_safetensors(str(tpath), disable_mmap=disable_mmap)

        if not state_dict:
            notes.append(f"No safetensors files found at {transformer_path}")
            return None, notes

        logger.info("Loaded %d tensors from transformer path", len(state_dict))

    except ImportError:
        notes.append("safetensors not available; cannot load transformer weights")
        return None, notes
    except Exception as exc:
        notes.append(f"Failed to load transformer weights: {exc}")
        return None, notes

    # Try to detect known architectures from state dict keys
    keys = list(state_dict.keys())
    has_transformer_blocks = any("transformer_blocks" in k for k in keys)
    has_single_blocks = any("single_transformer_blocks" in k for k in keys)

    # Attempt diffusers-based loading for known architectures
    if has_transformer_blocks and has_single_blocks:
        # Looks like Flux — try FluxTransformer2DModel
        try:
            from diffusers import FluxTransformer2DModel
            model = FluxTransformer2DModel.from_pretrained(
                str(tpath) if tpath.is_dir() else str(tpath.parent),
                torch_dtype=dtype,
            )
            notes.append("Loaded as FluxTransformer2DModel")
            return model, notes
        except Exception:
            pass

    if has_transformer_blocks and not has_single_blocks:
        # Could be Lumina or similar
        try:
            from diffusers import LuminaTransformer2DModel
            model = LuminaTransformer2DModel.from_pretrained(
                str(tpath) if tpath.is_dir() else str(tpath.parent),
                torch_dtype=dtype,
            )
            notes.append("Loaded as LuminaTransformer2DModel")
            return model, notes
        except Exception:
            pass

    # Fallback: wrap raw state dict in a minimal module
    wrapper = _NextDiTWrapper(state_dict)
    if config:
        wrapper.patch_size = int(config.get("patch_size", getattr(wrapper, "patch_size", 1)) or 1)
        wrapper.in_channels = int(config.get("in_channels", getattr(wrapper, "in_channels", 16)) or 16)
        wrapper.config = type("NextDiTConfig", (), dict(config))()
    wrapper.to(device=device, dtype=dtype)
    notes.append(
        f"Loaded as reconstructed module tree ({len(state_dict)} tensors). "
        "Forward pass is functional (simplified DiT pattern); LoRA injection targets nn.Linear leaves."
    )
    return wrapper, notes


def _load_auto_model_with_local_safetensors(
    model_dir: Path,
    dtype: torch.dtype,
    *,
    trust_remote_code: bool = False,
    disable_mmap: bool = False,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    progress_prefix: str = "auto_model",
) -> Any:
    """Load a Transformers model even when the local safetensors file is renamed."""
    if _looks_like_local_jina_clip(model_dir):
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_local_jina_start",
            model_dir=model_dir,
            dtype=dtype,
        )
        return _load_local_jina_clip_model(
            model_dir,
            dtype,
            trust_remote_code=trust_remote_code,
        )

    from transformers import AutoConfig, AutoModel

    kwargs = {}
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        try:
            _emit_loader_progress(
                progress_callback,
                f"{progress_prefix}_auto_pretrained_start",
                model_dir=model_dir,
                dtype=dtype,
                dtype_arg="dtype",
                trust_remote_code=trust_remote_code,
            )
            model = AutoModel.from_pretrained(str(model_dir), dtype=dtype, **kwargs)
        except TypeError as exc:
            _emit_loader_progress(
                progress_callback,
                f"{progress_prefix}_auto_pretrained_dtype_arg_failed",
                model_dir=model_dir,
                error=f"{type(exc).__name__}: {exc}",
            )
            model = AutoModel.from_pretrained(str(model_dir), torch_dtype=dtype, **kwargs)
        _emit_loader_progress(progress_callback, f"{progress_prefix}_auto_pretrained_done", model_dir=model_dir)
        return model
    except (OSError, TypeError, ValueError) as exc:
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_auto_pretrained_failed",
            model_dir=model_dir,
            error=f"{type(exc).__name__}: {exc}",
        )
        sf_files = sorted(model_dir.glob("*.safetensors"))
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_safetensors_scan",
            model_dir=model_dir,
            safetensors_files=[path.name for path in sf_files],
        )
        has_standard_name = any(
            path.name in {"model.safetensors", "pytorch_model.safetensors"}
            for path in sf_files
        )
        if not sf_files or has_standard_name:
            raise

        from core.lulynx_trainer.safetensors_loader import load_safetensors

        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_load_safetensors_start",
            safetensors_file=sf_files[0],
            disable_mmap=disable_mmap,
        )
        state_dict = load_safetensors(str(sf_files[0]), device="cpu", disable_mmap=disable_mmap)
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_load_safetensors_done",
            safetensors_file=sf_files[0],
            tensor_count=len(state_dict),
        )
        _emit_loader_progress(progress_callback, f"{progress_prefix}_auto_config_start", model_dir=model_dir)
        config = AutoConfig.from_pretrained(str(model_dir), **kwargs)
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_auto_config_done",
            model_type=getattr(config, "model_type", ""),
            architecture=getattr(config, "architectures", None),
        )
        if _looks_like_gemma3_text_only_state_dict(config, state_dict):
            return _load_gemma3_text_only_from_state_dict(
                config,
                state_dict,
                dtype,
                progress_callback=progress_callback,
                progress_prefix=progress_prefix,
            )
        _emit_loader_progress(progress_callback, f"{progress_prefix}_model_from_config_start", model_dir=model_dir)
        model = AutoModel.from_config(config, **kwargs)
        _emit_loader_progress(progress_callback, f"{progress_prefix}_model_from_config_done", model_dir=model_dir)
        _emit_loader_progress(progress_callback, f"{progress_prefix}_load_state_dict_start", tensor_count=len(state_dict))
        incompatible = model.load_state_dict(state_dict, strict=False)
        _emit_loader_progress(
            progress_callback,
            f"{progress_prefix}_load_state_dict_done",
            missing_keys_count=len(getattr(incompatible, "missing_keys", []) or []),
            unexpected_keys_count=len(getattr(incompatible, "unexpected_keys", []) or []),
        )
        if isinstance(model, torch.nn.Module):
            _emit_loader_progress(progress_callback, f"{progress_prefix}_model_to_dtype_start", dtype=dtype)
            model.to(dtype=dtype)
            _emit_loader_progress(progress_callback, f"{progress_prefix}_model_to_dtype_done", dtype=dtype)
        return model


def _looks_like_gemma3_text_only_state_dict(config: Any, state_dict: Dict[str, torch.Tensor]) -> bool:
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    if model_type != "gemma3":
        return False
    keys = [str(key) for key in state_dict]
    model_keys = [key for key in keys if key.startswith("model.")]
    if not model_keys:
        return False
    non_model_keys = [key for key in keys if not key.startswith("model.")]
    allowed_extra = {"spiece_model"}
    if any(key not in allowed_extra for key in non_model_keys):
        return False
    return any(key.startswith("model.layers.") for key in model_keys) and any(
        key == "model.embed_tokens.weight" for key in model_keys
    )


def _load_gemma3_text_only_from_state_dict(
    config: Any,
    state_dict: Dict[str, torch.Tensor],
    dtype: torch.dtype,
    *,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    progress_prefix: str = "gemma",
) -> torch.nn.Module:
    """Load text-only Gemma3 checkpoints without constructing the vision wrapper."""

    from transformers import Gemma3TextConfig, Gemma3TextModel

    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise ValueError("Gemma3 config is missing text_config for text-only load")
    if hasattr(text_config, "to_dict"):
        text_config_dict = text_config.to_dict()
    elif isinstance(text_config, dict):
        text_config_dict = dict(text_config)
    else:
        text_config_dict = {
            key: value
            for key, value in vars(text_config).items()
            if not str(key).startswith("_")
        }
    text_config_dict.pop("dtype", None)
    _emit_loader_progress(
        progress_callback,
        f"{progress_prefix}_text_only_model_from_config_start",
        num_hidden_layers=text_config_dict.get("num_hidden_layers"),
        hidden_size=text_config_dict.get("hidden_size"),
        tensor_count=len(state_dict),
    )
    text_model = Gemma3TextModel(Gemma3TextConfig(**text_config_dict))
    _emit_loader_progress(progress_callback, f"{progress_prefix}_text_only_model_from_config_done")

    text_state_dict = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if str(key).startswith("model.")
    }
    _emit_loader_progress(
        progress_callback,
        f"{progress_prefix}_text_only_load_state_dict_start",
        tensor_count=len(text_state_dict),
    )
    incompatible = text_model.load_state_dict(text_state_dict, strict=False)
    _emit_loader_progress(
        progress_callback,
        f"{progress_prefix}_text_only_load_state_dict_done",
        missing_keys_count=len(getattr(incompatible, "missing_keys", []) or []),
        unexpected_keys_count=len(getattr(incompatible, "unexpected_keys", []) or []),
    )
    _emit_loader_progress(progress_callback, f"{progress_prefix}_text_only_model_to_dtype_start", dtype=dtype)
    text_model.to(dtype=dtype)
    _emit_loader_progress(progress_callback, f"{progress_prefix}_text_only_model_to_dtype_done", dtype=dtype)
    return text_model


def _load_gemma_native(
    gemma_path: str, dtype: torch.dtype, max_token_length: int, trust_remote_code: bool = False,
    *, disable_mmap: bool = False,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[Any, Any, list]:
    """Load a Gemma text encoder and tokenizer.

    Returns ``(encoder, tokenizer, notes)``.
    """
    notes = []
    gpath = Path(gemma_path)

    if not gpath.exists():
        notes.append(f"Gemma path does not exist: {gemma_path}")
        return None, None, notes

    try:
        from transformers import AutoTokenizer

        # Try subfolder layout first
        te_dir = gpath / "text_encoder" if (gpath / "text_encoder").is_dir() else gpath
        tok_dir = gpath / "tokenizer" if (gpath / "tokenizer").is_dir() else gpath

        _emit_loader_progress(
            progress_callback,
            "gemma_model_load_start",
            text_encoder_dir=te_dir,
            tokenizer_dir=tok_dir,
            dtype=dtype,
            max_token_length=max_token_length,
            disable_mmap=disable_mmap,
        )
        encoder = _load_auto_model_with_local_safetensors(
            te_dir,
            dtype,
            trust_remote_code=trust_remote_code,
            disable_mmap=disable_mmap,
            progress_callback=progress_callback,
            progress_prefix="gemma",
        )
        _emit_loader_progress(progress_callback, "gemma_model_load_done", text_encoder_dir=te_dir)
        _emit_loader_progress(progress_callback, "gemma_tokenizer_start", tokenizer_dir=tok_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=trust_remote_code)
        _emit_loader_progress(progress_callback, "gemma_tokenizer_done", tokenizer_dir=tok_dir)

        # Apply max token length hint
        if max_token_length > 0 and hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = max_token_length

        notes.append(f"Gemma loaded from {te_dir} (max_tok={max_token_length})")
        return encoder, tokenizer, notes

    except ImportError as exc:
        _emit_loader_progress(
            progress_callback,
            "gemma_load_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        notes.append("transformers not available; cannot load Gemma")
        return None, None, notes
    except Exception as exc:
        _emit_loader_progress(
            progress_callback,
            "gemma_load_failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        notes.append(f"Gemma loading failed: {exc}")
        return None, None, notes


def _load_clip_native(
    clip_path: str, dtype: torch.dtype, max_token_length: int, trust_remote_code: bool = False,
    *, disable_mmap: bool = False,
    safe_fallback: bool = False,
) -> Tuple[Any, Any, list]:
    """Load a CLIP model and tokenizer.

    Returns ``(model, tokenizer, notes)``.
    """
    notes = []
    cpath = Path(clip_path)

    if not cpath.exists():
        notes.append(f"CLIP path does not exist: {clip_path}")
        return None, None, notes

    try:
        from transformers import AutoTokenizer

        # Try subfolder layout
        source_clip_dir = cpath / "clip_model" if (cpath / "clip_model").is_dir() else cpath
        tok_dir = cpath / "tokenizer" if (cpath / "tokenizer").is_dir() else cpath
        clip_dir, recovery_notes = _prepare_jina_clip_runtime_dir(source_clip_dir)
        notes.extend(recovery_notes)
        if tok_dir == source_clip_dir and clip_dir != source_clip_dir:
            tok_dir = clip_dir

        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=trust_remote_code)
        if max_token_length > 0 and hasattr(tokenizer, "model_max_length"):
            tokenizer.model_max_length = max_token_length

        local_text_pooler, pooler_notes = _load_local_jina_text_pooler(clip_dir)
        notes.extend(pooler_notes)
        if local_text_pooler is not None:
            model = local_text_pooler
        else:
            model = _load_auto_model_with_local_safetensors(
                clip_dir,
                dtype,
                trust_remote_code=trust_remote_code,
                disable_mmap=disable_mmap,
            )

        notes.append(f"CLIP loaded from {clip_dir} (max_tok={max_token_length})")
        return model, tokenizer, notes

    except ModuleNotFoundError as exc:
        if exc.name == "transformers":
            notes.append("transformers not available; cannot load CLIP")
        else:
            notes.append(f"CLIP loading failed with missing module {exc.name}: {exc}")
        return None, None, notes
    except Exception as exc:
        notes.append(f"CLIP loading failed ({type(exc).__name__}): {exc}")
        if safe_fallback:
            model, tokenizer, fallback_notes = _build_emergency_newbie_clip_fallback(
                max_token_length=max_token_length,
                notes=notes,
            )
            return model, tokenizer, fallback_notes
        return None, None, notes


def _load_vae_native(vae_path: str, dtype: torch.dtype) -> Tuple[Any, list]:
    """Load a VAE.

    Returns ``(vae, notes)``.
    """
    notes = []
    vpath = Path(vae_path)

    if not vpath.exists():
        notes.append(f"VAE path does not exist: {vae_path}")
        return None, notes

    try:
        from diffusers import AutoencoderKL

        # Try subfolder layout
        vae_dir = vpath / "vae" if (vpath / "vae").is_dir() else vpath

        vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype)
        notes.append(f"VAE loaded from {vae_dir}")
        return vae, notes

    except ImportError:
        notes.append("diffusers not available; cannot load VAE")
        return None, notes
    except Exception as exc:
        notes.append(f"VAE loading failed: {exc}")
        return None, notes


def _load_scheduler_native(vae_path: str) -> Tuple[Any, list]:
    """Try to load a scheduler from the VAE directory or root.

    Returns ``(scheduler, notes)``.
    """
    notes = []
    vpath = Path(vae_path)

    try:
        from diffusers import DDPMScheduler

        sched_dir = vpath / "scheduler" if (vpath / "scheduler").is_dir() else vpath
        scheduler = DDPMScheduler.from_pretrained(str(sched_dir))
        notes.append(f"Scheduler loaded from {sched_dir}")
        return scheduler, notes

    except Exception:
        # Fallback: create a default scheduler
        from diffusers import DDPMScheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        notes.append("Using default DDPMScheduler (no scheduler found at path)")
        return scheduler, notes


def _load_newbie_native_bundle(
    transformer_path: str,
    gemma_path: str,
    clip_path: str,
    vae_path: str,
    device: str,
    dtype: torch.dtype,
    lora_target: str,
    gemma3_prompt: str,
    use_flash_attn2: bool,
    adapter_type: str,
    target_modules: str,
    safe_fallback: bool,
    gemma_max_token_length: int,
    clip_max_token_length: int,
    trust_remote_code: bool,
    *,
    disable_mmap: bool = False,
) -> LoadedModel:
    """Load a Newbie model from individual native component paths.

    This loads each component (transformer, Gemma, CLIP, VAE) from its
    own path.  Each component is loaded independently with honest error
    reporting — if a component fails, the error is surfaced immediately.
    """
    logger.info(
        "Newbie native bundle: transformer=%s, gemma=%s, clip=%s, vae=%s",
        transformer_path, gemma_path, clip_path, vae_path,
    )

    all_notes: list[str] = []

    # ── Transformer ───────────────────────────────────────────────────
    transformer, notes = _load_transformer_native(
        transformer_path, dtype, device, disable_mmap=disable_mmap,
    )
    all_notes.extend(notes)
    if transformer is None:
        raise FileNotFoundError(
            f"Failed to load Newbie transformer from {transformer_path}.  "
            "Notes: " + "; ".join(notes)
        )

    # ── Gemma text encoder ────────────────────────────────────────────
    gemma_encoder, gemma_tokenizer, notes = _load_gemma_native(
        gemma_path, dtype, gemma_max_token_length, trust_remote_code,
        disable_mmap=disable_mmap,
    )
    all_notes.extend(notes)
    if gemma_encoder is None:
        raise FileNotFoundError(
            f"Failed to load Newbie Gemma encoder from {gemma_path}.  "
            "Notes: " + "; ".join(notes)
        )

    # ── CLIP model ────────────────────────────────────────────────────
    clip_model, clip_tokenizer, notes = _load_clip_native(
        clip_path, dtype, clip_max_token_length, trust_remote_code,
        disable_mmap=disable_mmap,
        safe_fallback=safe_fallback,
    )
    all_notes.extend(notes)
    if clip_model is None:
        raise FileNotFoundError(
            f"Failed to load Newbie CLIP model from {clip_path}.  "
            "Notes: " + "; ".join(notes)
        )

    # ── VAE ───────────────────────────────────────────────────────────
    vae, notes = _load_vae_native(vae_path, dtype)
    all_notes.extend(notes)
    if vae is None:
        raise FileNotFoundError(
            f"Failed to load Newbie VAE from {vae_path}.  "
            "Notes: " + "; ".join(notes)
        )

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler, notes = _load_scheduler_native(vae_path)
    all_notes.extend(notes)

    # ── Assemble LoadedModel ──────────────────────────────────────────
    # The transformer goes into the `unet` slot — in DiT-family models,
    # the transformer IS the denoiser.  The training loop calls
    # model.unet(...) for the forward pass.
    model = LoadedModel(
        unet=transformer,
        text_encoder_1=gemma_encoder,
        text_encoder_2=clip_model,
        vae=vae,
        tokenizer_1=gemma_tokenizer,
        tokenizer_2=clip_tokenizer,
        noise_scheduler=scheduler,
        model_arch="newbie",
    )

    # Eagerly create text_embed projection so it's visible to parameter
    # collection sweeps (LoRA injector / full-finetune requires_grad filter).
    if isinstance(transformer, _NextDiTWrapper) and clip_model is not None:
        clip_dim = getattr(getattr(clip_model, "config", None), "projection_dim", None)
        if clip_dim is None:
            clip_dim = getattr(getattr(clip_model, "config", None), "hidden_size", None)
        if clip_dim is not None:
            transformer.ensure_text_embed_projection(int(clip_dim))

    # Attach metadata
    model.newbie_lora_target = lora_target
    model.newbie_unet_targets = get_newbie_targets(lora_target)[0]
    model.newbie_te_targets = get_newbie_targets(lora_target)[1]
    model.newbie_gemma3_prompt = gemma3_prompt
    model.newbie_use_flash_attn2 = use_flash_attn2
    model.newbie_adapter_type = adapter_type
    model.newbie_native_bundle_loaded = True
    model.newbie_scaffold_mode = False
    model.newbie_native_layout_detected = True
    model.newbie_native_conditioning_ready = True
    model.newbie_transport_ready = True
    model.newbie_forward_smoke_passed = False
    model.newbie_gradient_smoke_passed = False
    model.newbie_transport_mode = "flow_matching"
    model.newbie_native_notes = all_notes

    logger.info(
        "Newbie native bundle loaded.  Notes:\n  %s",
        "\n  ".join(all_notes),
    )
    return model


def release_loaded_model_components(
    model: LoadedModel,
    *attrs: str,
) -> tuple[str, ...]:
    """Best-effort release selected LoadedModel components.

    This is the Warehouse boundary for cache-first Newbie loading: the cache
    builder may temporarily need VAE / text encoders / tokenizers, but the
    actual training step should be allowed to keep only the transformer alive.
    """
    released: list[str] = []
    for attr in attrs:
        value = getattr(model, attr, None)
        if value is None:
            continue
        if isinstance(value, torch.nn.Module):
            try:
                value.to("cpu")
            except Exception:
                pass
        setattr(model, attr, None)
        released.append(attr)

    if released:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return tuple(released)


def load_newbie_encoders_only_from_config(
    config: Any,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> LoadedModel:
    """Load only the encoder/cache-builder side of a native Newbie bundle.

    Build cache with VAE + Gemma + CLIP, keep a small placeholder transformer
    slot for contract compatibility, then release these components before
    loading the actual training transformer.
    """
    diffusers_path = str(getattr(config, "newbie_diffusers_path", "") or "")
    transformer_path = str(getattr(config, "newbie_transformer_path", "") or "")
    gemma_path = str(getattr(config, "newbie_gemma_model_path", "") or "")
    clip_path = str(getattr(config, "newbie_clip_model_path", "") or "")
    vae_path = str(getattr(config, "newbie_vae_path", "") or "")
    lora_target = str(getattr(config, "newbie_lora_target", "minimal") or "minimal")
    gemma3_prompt = str(getattr(config, "newbie_gemma3_prompt", "") or "")
    use_flash_attn2 = bool(getattr(config, "newbie_use_flash_attn2", True))
    adapter_type = str(getattr(config, "newbie_adapter_type", "") or "")
    gemma_max_token_length = int(getattr(config, "newbie_gemma_max_token_length", 512) or 512)
    clip_max_token_length = int(getattr(config, "newbie_clip_max_token_length", 2048) or 2048)
    trust_remote_code = bool(getattr(config, "trust_remote_code", False))
    safe_fallback = bool(getattr(config, "newbie_safe_fallback", True))
    resolved_dtype = dtype or torch.bfloat16
    disable_mmap = bool(getattr(config, "disable_mmap_load_safetensors", False))
    load_scheduler = bool(getattr(config, "newbie_cache_builder_load_scheduler", False))
    profile_start = time.perf_counter()
    profile_last = profile_start
    profile_steps: list[Dict[str, Any]] = []

    def mark(label: str, **data: Any) -> None:
        nonlocal profile_last
        now = time.perf_counter()
        item: Dict[str, Any] = {
            "label": str(label),
            "dt_seconds": round(max(now - profile_last, 0.0), 4),
            "total_seconds": round(max(now - profile_start, 0.0), 4),
        }
        for key, value in data.items():
            if value is not None:
                item[str(key)] = value
        profile_steps.append(item)
        profile_last = now

    logger.info(
        "Newbie native cache-builder bundle: gemma=%s, clip=%s, vae=%s",
        gemma_path,
        clip_path,
        vae_path or diffusers_path,
    )

    placeholder_transformer = torch.nn.Identity()
    placeholder_transformer.requires_grad_(False)
    all_notes: list[str] = [
        "Loaded encoder-only Newbie cache-builder bundle; transformer deferred until training phase.",
    ]

    gemma_encoder, gemma_tokenizer, notes = _load_gemma_native(
        gemma_path, resolved_dtype, gemma_max_token_length, trust_remote_code,
        disable_mmap=disable_mmap,
        progress_callback=progress_callback,
    )
    mark("load_gemma", loaded=gemma_encoder is not None, note_count=len(notes))
    all_notes.extend(notes)
    if gemma_encoder is None:
        raise FileNotFoundError(
            f"Failed to load Newbie Gemma encoder from {gemma_path}.  "
            "Notes: " + "; ".join(notes)
        )

    clip_model, clip_tokenizer, notes = _load_clip_native(
        clip_path, resolved_dtype, clip_max_token_length, trust_remote_code,
        disable_mmap=disable_mmap,
        safe_fallback=safe_fallback,
    )
    mark("load_clip", loaded=clip_model is not None, note_count=len(notes))
    all_notes.extend(notes)
    if clip_model is None:
        raise FileNotFoundError(
            f"Failed to load Newbie CLIP model from {clip_path}.  "
            "Notes: " + "; ".join(notes)
        )

    vae, notes = _load_vae_native(vae_path or diffusers_path, torch.float32)
    mark("load_vae", loaded=vae is not None, note_count=len(notes))
    all_notes.extend(notes)
    if vae is None:
        raise FileNotFoundError(
            f"Failed to load Newbie VAE from {vae_path or diffusers_path}.  "
            "Notes: " + "; ".join(notes)
        )

    scheduler = None
    if load_scheduler:
        scheduler, notes = _load_scheduler_native(vae_path or diffusers_path)
        mark("load_scheduler", loaded=scheduler is not None, note_count=len(notes))
        all_notes.extend(notes)
    else:
        mark("skip_scheduler", loaded=False, note_count=0)
        all_notes.append("Skipped scheduler load for cache-builder-only Newbie bundle.")

    model = LoadedModel(
        unet=placeholder_transformer,
        text_encoder_1=gemma_encoder,
        text_encoder_2=clip_model,
        vae=vae,
        tokenizer_1=gemma_tokenizer,
        tokenizer_2=clip_tokenizer,
        noise_scheduler=scheduler,
        model_arch="newbie",
    )
    model.newbie_lora_target = lora_target
    model.newbie_unet_targets = get_newbie_targets(lora_target)[0]
    model.newbie_te_targets = get_newbie_targets(lora_target)[1]
    model.newbie_gemma3_prompt = gemma3_prompt
    model.newbie_use_flash_attn2 = use_flash_attn2
    model.newbie_adapter_type = adapter_type
    model.newbie_native_bundle_loaded = False
    model.newbie_cache_builder_bundle_loaded = True
    model.newbie_scaffold_mode = False
    model.newbie_native_layout_detected = True
    model.newbie_native_conditioning_ready = True
    model.newbie_transport_ready = False
    model.newbie_forward_smoke_passed = False
    model.newbie_gradient_smoke_passed = False
    model.newbie_transport_mode = "flow_matching"
    model.newbie_native_notes = all_notes
    model.newbie_loader_profile = {
        "mode": "encoders_only",
        "device": str(device),
        "dtype": str(resolved_dtype),
        "load_scheduler": bool(load_scheduler),
        "total_seconds": profile_steps[-1]["total_seconds"] if profile_steps else 0.0,
        "steps": profile_steps,
    }
    return model


def load_newbie_transformer_only_from_config(
    config: Any,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Load only the train-time Newbie transformer.

    Used by the cache-first two-stage path to avoid keeping Gemma / CLIP / VAE
    resident once explicit ``*_newbie.npz`` artifacts already exist.
    """
    diffusers_path = str(getattr(config, "newbie_diffusers_path", "") or "")
    transformer_path = str(getattr(config, "newbie_transformer_path", "") or "")
    lora_target = str(getattr(config, "newbie_lora_target", "minimal") or "minimal")
    gemma3_prompt = str(getattr(config, "newbie_gemma3_prompt", "") or "")
    use_flash_attn2 = bool(getattr(config, "newbie_use_flash_attn2", True))
    adapter_type = str(getattr(config, "newbie_adapter_type", "") or "")
    resolved_dtype = dtype or torch.bfloat16
    disable_mmap = bool(getattr(config, "disable_mmap_load_safetensors", False))
    residency_mode = str(getattr(config, "newbie_block_residency", "resident") or "resident")
    early_block_residency = (
        residency_mode.strip().lower().replace("-", "_") not in {"", "resident", "off", "none", "gpu"}
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    )
    load_device = "cpu" if early_block_residency else device
    profile_start = time.perf_counter()
    profile_last = profile_start
    profile_steps: list[Dict[str, Any]] = []

    def mark(label: str, **data: Any) -> None:
        nonlocal profile_last
        now = time.perf_counter()
        item: Dict[str, Any] = {
            "label": str(label),
            "dt_seconds": round(max(now - profile_last, 0.0), 4),
            "total_seconds": round(max(now - profile_start, 0.0), 4),
        }
        for key, value in data.items():
            if value is not None:
                item[str(key)] = value
        profile_steps.append(item)
        profile_last = now

    transformer, notes = _load_transformer_native(
        transformer_path or diffusers_path, resolved_dtype, load_device,
        disable_mmap=disable_mmap,
    )
    mark(
        "load_transformer",
        loaded=transformer is not None,
        load_device=str(load_device),
        note_count=len(notes),
        early_block_residency=early_block_residency,
    )
    if transformer is None:
        raise FileNotFoundError(
            f"Failed to load Newbie transformer from {transformer_path or diffusers_path}.  "
            "Notes: " + "; ".join(notes)
        )

    if early_block_residency:
        try:
            from .newbie_block_residency import apply_newbie_block_residency, move_newbie_nonresident_tensors

            report = apply_newbie_block_residency(
                transformer,
                mode=residency_mode,
                min_parameter_count=max(int(getattr(config, "newbie_block_residency_min_params", 0) or 0), 0),
                sparse_swap_enabled=bool(getattr(config, "sparse_swap_enabled", False)),
                sparse_swap_budget_mb=max(float(getattr(config, "sparse_swap_budget_mb", 0.0) or 0.0), 0.0) or None,
                sparse_swap_warm_fraction=min(max(float(getattr(config, "sparse_swap_warm_fraction", 0.35) or 0.35), 0.0), 1.0),
            )
            move_newbie_nonresident_tensors(transformer, device=device, dtype=resolved_dtype)
            transformer._newbie_early_block_residency_profile = report.as_dict()
            mark(
                "apply_early_block_residency",
                mode=report.mode,
                active_linear_count=report.active_linear_count,
                managed_linear_count=report.managed_linear_count,
                cpu_parameter_mb=round(float(report.cpu_parameter_mb), 3),
            )
            notes.append(
                "Applied early Newbie block residency before GPU placement: "
                f"mode={report.mode}, active_linear={report.active_linear_count}/{report.managed_linear_count}, "
                f"cpu_params={report.cpu_parameter_mb:.1f}MB"
            )
        except Exception as exc:
            mark("apply_early_block_residency_failed", error=f"{type(exc).__name__}: {exc}")
            notes.append(f"Early Newbie block residency skipped: {exc}")

    model = LoadedModel(
        unet=transformer,
        text_encoder_1=None,
        text_encoder_2=None,
        vae=None,
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        model_arch="newbie",
    )
    model.newbie_lora_target = lora_target
    model.newbie_unet_targets = get_newbie_targets(lora_target)[0]
    model.newbie_te_targets = get_newbie_targets(lora_target)[1]
    model.newbie_gemma3_prompt = gemma3_prompt
    model.newbie_use_flash_attn2 = use_flash_attn2
    model.newbie_adapter_type = adapter_type
    model.newbie_native_bundle_loaded = True
    model.newbie_cache_builder_bundle_loaded = False
    model.newbie_transformer_only_loaded = True
    model.newbie_scaffold_mode = False
    model.newbie_native_layout_detected = True
    model.newbie_native_conditioning_ready = True
    model.newbie_transport_ready = True
    model.newbie_forward_smoke_passed = False
    model.newbie_gradient_smoke_passed = False
    model.newbie_transport_mode = "flow_matching"
    model.newbie_native_notes = list(notes)
    model.newbie_loader_profile = {
        "mode": "transformer_only",
        "device": str(device),
        "load_device": str(load_device),
        "dtype": str(resolved_dtype),
        "early_block_residency": bool(early_block_residency),
        "total_seconds": profile_steps[-1]["total_seconds"] if profile_steps else 0.0,
        "steps": profile_steps,
    }
    return model


def load_newbie_from_config(
    config: Any,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> LoadedModel:
    """Load a Newbie model using fields from a ``UnifiedTrainingConfig``.

    Supports two loading paths:
    1. Diffusers directory: via ``newbie_diffusers_path``
    2. Native bundle: via individual component paths
       (``newbie_transformer_path``, ``newbie_gemma_model_path``,
       ``newbie_clip_model_path``, ``newbie_vae_path``)

    Parameters
    ----------
    config:
        A ``UnifiedTrainingConfig`` instance (or any object with the
        expected Newbie fields).
    """

    diffusers_path = str(getattr(config, "newbie_diffusers_path", "") or "")
    transformer_path = str(getattr(config, "newbie_transformer_path", "") or "")
    gemma_path = str(getattr(config, "newbie_gemma_model_path", "") or "")
    clip_path = str(getattr(config, "newbie_clip_model_path", "") or "")
    vae_path = str(getattr(config, "newbie_vae_path", "") or "")

    lora_target = str(
        getattr(config, "newbie_lora_target", "minimal") or "minimal"
    )
    gemma3_prompt = str(getattr(config, "newbie_gemma3_prompt", "") or "")
    use_flash_attn2 = bool(getattr(config, "newbie_use_flash_attn2", True))
    adapter_type = str(getattr(config, "newbie_adapter_type", "") or "")
    target_modules = str(getattr(config, "newbie_target_modules", "") or "")
    safe_fallback = bool(getattr(config, "newbie_safe_fallback", True))
    gemma_max_token_length = int(getattr(config, "newbie_gemma_max_token_length", 512) or 512)
    clip_max_token_length = int(getattr(config, "newbie_clip_max_token_length", 2048) or 2048)
    trust_remote_code = bool(getattr(config, "trust_remote_code", False))
    run_native_smoke = bool(getattr(config, "newbie_run_native_smoke", False))
    disable_mmap = bool(getattr(config, "disable_mmap_load_safetensors", False))

    resolved_dtype = dtype or torch.bfloat16

    # Determine loading path
    has_native_bundle = bool(transformer_path or gemma_path or clip_path)

    if diffusers_path and not has_native_bundle:
        # Path 1: Standard diffusers directory
        model = load_newbie(
            diffusers_path=diffusers_path,
            device=device,
            dtype=resolved_dtype,
            lora_target=lora_target,
            gemma3_prompt=gemma3_prompt,
            use_flash_attn2=use_flash_attn2,
        )
        if run_native_smoke:
            _run_optional_loaded_newbie_smoke(model)
        return model

    if has_native_bundle:
        # Path 2: Native bundle with individual component paths
        model = _load_newbie_native_bundle(
            transformer_path=transformer_path,
            gemma_path=gemma_path,
            clip_path=clip_path,
            vae_path=vae_path or diffusers_path,  # fallback to diffusers_path for VAE
            device=device,
            dtype=resolved_dtype,
            lora_target=lora_target,
            gemma3_prompt=gemma3_prompt,
            use_flash_attn2=use_flash_attn2,
            adapter_type=adapter_type,
            target_modules=target_modules,
            safe_fallback=safe_fallback,
            gemma_max_token_length=gemma_max_token_length,
            clip_max_token_length=clip_max_token_length,
            trust_remote_code=trust_remote_code,
            disable_mmap=disable_mmap,
        )
        if run_native_smoke:
            _run_optional_loaded_newbie_smoke(model)
        return model

    raise ValueError(
        "Newbie mode requires either newbie_diffusers_path or native "
        "component paths (newbie_transformer_path / newbie_gemma_model_path / "
        "newbie_clip_model_path) to be set in config."
    )


def _run_optional_loaded_newbie_smoke(model: LoadedModel) -> None:
    """Run explicit Newbie smoke and attach conservative per-model attrs."""

    result = run_loaded_newbie_smoke(model)
    notes = list(getattr(model, "newbie_native_notes", []) or [])
    if result.passed:
        notes.append(
            "Newbie transformer smoke passed: "
            f"latent={result.latent_shape}, targets={list(result.gradient_targets)}"
        )
        logger.info("Newbie transformer smoke passed: %s", result)
    else:
        notes.append(f"Newbie transformer smoke failed safely: {result.reason}")
        logger.warning("Newbie transformer smoke failed safely: %s", result.reason)
    model.newbie_native_notes = notes


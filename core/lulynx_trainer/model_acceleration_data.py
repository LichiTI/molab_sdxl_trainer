"""Data and cache predicates for model-aware acceleration policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping


_WEBDATASET_SHARD_SUFFIXES = (".tar", ".tar.gz", ".tgz")


def _str(value: Any) -> str:
    return str(value or "").strip()


def _flag(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stable_caption_conditioning(config: Mapping[str, Any], *, default_shuffle: bool = False) -> bool:
    if _flag(config.get("shuffle_caption"), default=default_shuffle) and not _flag(config.get("shuffle_caption_tags_only")):
        return False
    if (_float_or_none(config.get("caption_dropout_rate")) or 0.0) > 0.0:
        return False
    if (_float_or_none(config.get("tag_dropout_rate")) or 0.0) > 0.0:
        return False
    return not _str(config.get("caption_tag_dropout_targets"))


def can_cache_text_encoder_outputs(config: Mapping[str, Any], family: str) -> bool:
    if not stable_caption_conditioning(config, default_shuffle=family == "flux"):
        return False
    if _flag(config.get("train_text_encoder")) or _flag(config.get("network_train_text_encoder_only")):
        return False
    if family == "flux":
        return True
    if _flag(config.get("network_train_unet_only")):
        return True
    return "train_text_encoder" in config and not _flag(config.get("train_text_encoder"))


def _data_dir(config: Mapping[str, Any], data_dir: str | Path | None) -> Path | None:
    raw = data_dir or config.get("train_data_dir") or config.get("data_dir") or config.get("trainDataDir")
    if not raw:
        return None
    try:
        return Path(str(raw)).expanduser()
    except (TypeError, ValueError):
        return None


def _looks_like_webdataset_shard(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _WEBDATASET_SHARD_SUFFIXES)


def _has_webdataset_shards(path: Path | None) -> bool:
    if path is None:
        return False
    try:
        if path.is_file():
            return _looks_like_webdataset_shard(path)
        if not path.is_dir():
            return False
        return any(child.is_file() and _looks_like_webdataset_shard(child) for child in path.iterdir())
    except OSError:
        return False


def apply_data_backend_policy(
    decision: Any,
    config: Mapping[str, Any],
    *,
    data_dir: str | Path | None,
    patch_text: Callable[[str, str, str], None],
) -> None:
    if decision.effective_profile != "aggressive":
        return
    path = _data_dir(config, data_dir)
    if not _has_webdataset_shards(path):
        decision.notes.append("Aggressive data-backend acceleration can use WebDataset once .tar/.tar.gz/.tgz shards are materialized.")
        return
    if decision.model_family in {"anima", "newbie"}:
        decision.notes.append(
            "WebDataset materialization is kept off for native cached Anima/Newbie routes; "
            "their current training path does not consume data_backend yet."
        )
        decision.add_skip("data_backend", "data_backend", config.get("data_backend", "auto"), "native cache route")
        return
    patch_text(
        "data_backend",
        "webdataset",
        "Use the materialized CaptionDataset bridge when the dataset already contains WebDataset shards.",
    )


__all__ = ["apply_data_backend_policy", "can_cache_text_encoder_outputs", "stable_caption_conditioning"]

"""
数据集加载器

支持 caption/tag 格式，分桶策略
"""

import os
import random
import hashlib
import logging
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np

try:
    import pillow_jxl  # noqa: F401 — registers .jxl codec with Pillow
except ImportError:
    pass

try:
    from .caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from .caption_source_mix import (
        compose_caption_from_source,
        merge_trigger_tokens,
        normalize_caption_source_mix_config,
        remove_trigger_tokens,
        select_caption_source,
    )
    from .dataset_discovery import resolve_caption_path
    from .easycontrol_v2_contract import (
        EasyControlV2TaskSpec,
        build_easycontrol_v2_task_spec,
        sidecar_plan_for_target,
    )
    from .model_to_condition import SDXLModelToCondition
except ImportError:  # pragma: no cover - direct script usage
    from caption_sidecar import json_caption_to_training_parts, json_caption_to_training_text
    from caption_source_mix import (
        compose_caption_from_source,
        merge_trigger_tokens,
        normalize_caption_source_mix_config,
        remove_trigger_tokens,
        select_caption_source,
    )
    from dataset_discovery import resolve_caption_path
    from easycontrol_v2_contract import (
        EasyControlV2TaskSpec,
        build_easycontrol_v2_task_spec,
        sidecar_plan_for_target,
    )
    from model_to_condition import SDXLModelToCondition

logger = logging.getLogger(__name__)


def _attach_native_shadow_adapter_if_enabled(
    dataloader: DataLoader,
    dataset: "CaptionDataset",
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> DataLoader:
    try:
        from core.turbocore_dataset_shadow_adapter import maybe_attach_caption_dataset_shadow_adapter
    except Exception:
        return dataloader
    return maybe_attach_caption_dataset_shadow_adapter(
        dataloader,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


@dataclass
class ImageSample:
    """单个训练样本"""
    image_path: str
    caption_path: Optional[str]  # Memory Optimization: Store path, load on demand
    caption_variant_paths: Optional[List[Optional[str]]] = None  # Multiple caption paths for variants
    original_size: Tuple[int, int] = (0, 0)
    target_size: Tuple[int, int] = (0, 0)
    crop_coords: Tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom
    caption_token_length: int = 0  # Pre-computed token count for length bucketing


class BucketManager:
    """分桶管理器"""
    
    def __init__(
        self,
        base_resolution: int = 1024,
        min_resolution: int = 512,
        max_resolution: int = 2048,
        resolution_step: int = 64,
        max_aspect_ratio: float = 2.0,
        selection_mode: str = "aspect",
        custom_resos: Optional[Any] = None,
    ):
        self.base_resolution = base_resolution
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.resolution_step = resolution_step
        self.max_aspect_ratio = max_aspect_ratio
        self.selection_mode = (selection_mode or "aspect").strip().lower()
        self.buckets = self._parse_custom_buckets(custom_resos) or self._generate_buckets()

    @staticmethod
    def _parse_custom_buckets(value: Optional[Any]) -> List[Tuple[int, int]]:
        """Parse UI custom bucket values such as ``512x768,768x512``."""
        if not value:
            return []
        raw_items: List[Any]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                raw_items = parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                raw_items = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
        elif isinstance(value, (list, tuple)):
            raw_items = list(value)
        else:
            raw_items = [value]

        buckets: List[Tuple[int, int]] = []
        for item in raw_items:
            try:
                if isinstance(item, dict):
                    width = int(item.get("width") or item.get("w"))
                    height = int(item.get("height") or item.get("h"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    width, height = int(item[0]), int(item[1])
                else:
                    text = str(item).lower().replace("*", "x").replace(" ", "")
                    if "x" not in text:
                        continue
                    left, right = text.split("x", 1)
                    width, height = int(left), int(right)
                if width > 0 and height > 0:
                    buckets.append((width, height))
            except Exception:
                logger.debug("Ignoring invalid custom bucket resolution: %r", item)
        return list(dict.fromkeys(buckets))
        
    def _generate_buckets(self) -> List[Tuple[int, int]]:
        """生成所有可用的桶尺寸"""
        buckets = []
        base_area = self.base_resolution ** 2
        
        for w in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
            for h in range(self.min_resolution, self.max_resolution + 1, self.resolution_step):
                # 检查面积接近目标
                area = w * h
                if base_area > 0 and abs(area - base_area) / base_area > 0.1:
                    continue
                elif base_area == 0:
                     continue
                    
                # 检查宽高比
                aspect = max(w / h, h / w)
                if aspect > self.max_aspect_ratio:
                    continue
                    
                buckets.append((w, h))
        
        return buckets
    
    def get_bucket(self, width: int, height: int) -> Tuple[int, int]:
        """为给定尺寸选择最佳桶"""
        aspect = width / height
        source_area = width * height
        
        best_bucket = None
        best_diff = float('inf')
        
        for bw, bh in self.buckets:
            bucket_aspect = bw / bh
            bucket_area = bw * bh
            if self.selection_mode in {"area", "pixel", "pixels"}:
                diff = abs(source_area - bucket_area)
            elif self.selection_mode in {"larger", "ceil", "no_downscale"}:
                if bw < width or bh < height:
                    continue
                diff = bucket_area - source_area
            elif self.selection_mode in {"smaller", "floor", "no_upscale"}:
                if bw > width or bh > height:
                    continue
                diff = source_area - bucket_area
            else:
                diff = abs(aspect - bucket_aspect)
            
            if diff < best_diff:
                best_diff = diff
                best_bucket = (bw, bh)
        
        return best_bucket or (self.base_resolution, self.base_resolution)


class CaptionDataset(Dataset):
    """Caption 数据集"""
    
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".jxl"}
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 1024,
        caption_extension: str = ".txt",
        enable_bucket: bool = True,
        min_bucket_reso: int = 512,
        max_bucket_reso: int = 2048,
        bucket_reso_steps: int = 64,
        bucket_selection_mode: str = "aspect",
        bucket_custom_resos: Optional[Any] = None,
        shuffle_caption: bool = True,
        shuffle_caption_tags_only: bool = False,
        keep_tokens: int = 0,
        keep_tokens_separator: str = "",
        flip_augment: bool = False,
        color_augment: bool = False,
        transform: Optional[Callable] = None,
        clip_transform: Optional[Callable] = None,
        conditioning_data_dir: str = "",
        # ── Advanced caption options ──
        caption_dropout_rate: float = 0.0,
        caption_dropout_every_n_epochs: int = 0,
        tag_dropout_rate: float = 0.0,
        caption_tag_dropout_targets: str = "",
        caption_tag_dropout_target_mode: str = "drop_all",
        caption_tag_dropout_target_count: int = 1,
        token_warmup_min: int = 0,
        token_warmup_max: int = 0,
        token_warmup_steps: int = 0,
        weighted_captions: bool = False,
        masked_loss: bool = False,
        alpha_mask: bool = False,
        # ── Caption-length bucketing ──
        caption_length_bucket_size: int = 0,
        # ── Caption Variants (Multi-Caption Training) ──
        caption_variants_enabled: bool = False,
        caption_variants: str = "",
        caption_variant_schedule: str = "alternate",
        caption_variant_ratio: str = "",
        caption_variant_curriculum_start_epoch: int = 0,
        caption_variant_curriculum_end_epoch: int = 10,
        caption_variant_custom_sequence: str = "",
        caption_variant_loss_adaptive: bool = False,
        albumentations_enabled: bool = False,
        albumentations_pipeline: str = "",
        albumentations_mask_replay: bool = True,
        dual_caption_enabled: bool = False,
        dual_caption_short_key: str = "short",
        dual_caption_long_key: str = "long",
        caption_source_mix_enabled: bool = False,
        caption_source_nl_ratio: float = 65.0,
        caption_source_tag_ratio: float = 20.0,
        caption_source_trigger_only_ratio: float = 10.0,
        caption_source_empty_ratio: float = 5.0,
        caption_source_trigger_tokens: str = "",
        image_decode_backend: str = "pil",
        image_decode_cache_size: int = 0,
        easycontrol_v2_enabled: bool = False,
        easycontrol_v2_task_id: str = "generic",
        easycontrol_v2_control_kind: str = "reference_latent",
        easycontrol_v2_target_family: str = "anima",
        easycontrol_v2_cond_cache_dir: str = "",
        easycontrol_v2_text_cache_dir: str = "",
        easycontrol_v2_control_image_dir: str = "",
        easycontrol_v2_control_suffix: str = "",
        easycontrol_v2_drop_p: float = 0.1,
        easycontrol_v2_cond_noise_max: float = 0.0,
        easycontrol_v2_scale: float = 1.0,
        easycontrol_v2_match_target_bucket: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.caption_extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
        self.shuffle_caption = shuffle_caption
        self.shuffle_caption_tags_only = bool(shuffle_caption_tags_only)
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator or ""
        self.flip_augment = flip_augment
        self.caption_length_bucket_size = caption_length_bucket_size
        self.color_augment = color_augment

        self._album_pipeline = None
        if albumentations_enabled and albumentations_pipeline:
            from .album_augment import AlbumentationsPipeline
            if AlbumentationsPipeline.available():
                self._album_pipeline = AlbumentationsPipeline(
                    albumentations_pipeline,
                    mask_replay=albumentations_mask_replay,
                )

        self._dual_caption_enabled = bool(dual_caption_enabled)
        self._dual_caption_short_key = str(dual_caption_short_key or "short")
        self._dual_caption_long_key = str(dual_caption_long_key or "long")
        self.caption_source_mix = normalize_caption_source_mix_config(
            enabled=caption_source_mix_enabled,
            nl_ratio=caption_source_nl_ratio,
            tag_ratio=caption_source_tag_ratio,
            trigger_only_ratio=caption_source_trigger_only_ratio,
            empty_ratio=caption_source_empty_ratio,
            trigger_tokens=caption_source_trigger_tokens,
        )
        self.image_decode_backend_requested = str(image_decode_backend or "pil").strip().lower().replace("-", "_")
        self.image_decode_cache_size = max(int(image_decode_cache_size or 0), 0)
        self.image_decode_backend = self._resolve_image_decode_backend(self.image_decode_backend_requested)
        self._image_decode_cache: "OrderedDict[Tuple[str, int, int, bool], Tuple[Image.Image, Optional[Image.Image]]]" = OrderedDict()
        self._image_decode_cache_hits = 0
        self._image_decode_cache_misses = 0
        self._image_decode_fallback_logged = False
        self.easycontrol_v2_enabled = bool(easycontrol_v2_enabled)
        self.easycontrol_v2_spec: Optional[EasyControlV2TaskSpec] = None
        if self.easycontrol_v2_enabled:
            self.easycontrol_v2_spec = build_easycontrol_v2_task_spec(
                {
                    "task_id": easycontrol_v2_task_id,
                    "control_kind": easycontrol_v2_control_kind,
                    "target_family": easycontrol_v2_target_family,
                    "cond_cache_dir": easycontrol_v2_cond_cache_dir,
                    "text_cache_dir": easycontrol_v2_text_cache_dir,
                    "control_image_dir": easycontrol_v2_control_image_dir,
                    "control_suffix": easycontrol_v2_control_suffix,
                    "drop_p": easycontrol_v2_drop_p,
                    "cond_noise_max": easycontrol_v2_cond_noise_max,
                    "scale": easycontrol_v2_scale,
                    "match_target_bucket": easycontrol_v2_match_target_bucket,
                }
            )

        # Parse caption variants
        self.caption_variants_enabled = caption_variants_enabled
        self.caption_variant_configs = self._parse_caption_variants(caption_variants) if caption_variants_enabled else []
        self.caption_variant_schedule = caption_variant_schedule
        self.caption_variant_ratio = self._parse_json_array(caption_variant_ratio)
        self.caption_variant_curriculum_start_epoch = caption_variant_curriculum_start_epoch
        self.caption_variant_curriculum_end_epoch = caption_variant_curriculum_end_epoch
        self.caption_variant_custom_sequence = self._parse_json_array(caption_variant_custom_sequence)
        self.caption_variant_loss_adaptive = caption_variant_loss_adaptive
        self._variant_epoch_counter = 0
        self.transform = transform
        self.clip_transform = clip_transform
        self.conditioning_data_dir = Path(conditioning_data_dir) if conditioning_data_dir else None
        # Advanced caption options
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = max(int(caption_dropout_every_n_epochs or 0), 0)
        self.tag_dropout_rate = tag_dropout_rate
        self.caption_tag_dropout_targets = self._parse_tag_targets(caption_tag_dropout_targets)
        self.caption_tag_dropout_target_mode = str(caption_tag_dropout_target_mode or "drop_all").strip().lower()
        self.caption_tag_dropout_target_count = max(int(caption_tag_dropout_target_count or 1), 1)
        self.token_warmup_min = token_warmup_min
        self.token_warmup_max = token_warmup_max
        self.token_warmup_steps = token_warmup_steps
        self.weighted_captions = weighted_captions
        self.masked_loss = masked_loss
        self.alpha_mask = alpha_mask
        self._global_step = 0
        self._current_epoch = 0

        # 分桶管理
        if enable_bucket:
            self.bucket_manager = BucketManager(
                base_resolution=resolution,
                min_resolution=min_bucket_reso,
                max_resolution=max_bucket_reso,
                resolution_step=bucket_reso_steps,
                selection_mode=bucket_selection_mode,
                custom_resos=bucket_custom_resos,
            )
        else:
            self.bucket_manager = None

        # 扫描数据集
        self.samples = self._scan_dataset()
        logger.info(f"Found {len(self.samples)} samples in {data_dir}")
        if self.image_decode_backend != self.image_decode_backend_requested:
            logger.info(
                "Image decode backend resolved: requested=%s, resolved=%s, cache_size=%s",
                self.image_decode_backend_requested,
                self.image_decode_backend,
                self.image_decode_cache_size,
            )

    @staticmethod
    def _parse_json_array(value: str) -> List:
        """Parse JSON array string"""
        if not value or not value.strip():
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _parse_caption_variants(value: str) -> List[Dict]:
        """Parse caption variants configuration JSON"""
        if not value or not value.strip():
            return []
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                return []
            # Validate each variant config
            variants = []
            for item in parsed:
                if isinstance(item, dict) and "suffix" in item:
                    variants.append({
                        "suffix": str(item["suffix"]),
                        "shuffle": bool(item.get("shuffle", True)),
                        "dropout": float(item.get("dropout", 0.0)),
                        "keep_tokens": int(item.get("keep_tokens", 0)),
                    })
            return variants
        except Exception as e:
            logger.warning(f"Failed to parse caption_variants: {e}")
            return []

    def _scan_dataset(self) -> List[ImageSample]:
        """扫描数据集目录"""
        samples = []
        need_token_length = self.caption_length_bucket_size > 0

        try:
            for img_path in self.data_dir.rglob("*"):
                if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                # ControlNet sidecar images should condition another sample,
                # not become separate training samples themselves.
                if img_path.stem.endswith("_control"):
                    continue

                # 查找对应的 caption 文件
                caption_path = resolve_caption_path(img_path.parent, img_path.stem, self.caption_extension)

                # Memory Optimization: Don't read caption here unless we need token length
                stored_caption_path = str(caption_path) if caption_path is not None and caption_path.exists() else None

                # Find caption variant files if enabled
                caption_variant_paths = None
                if self.caption_variants_enabled and self.caption_variant_configs:
                    caption_variant_paths = []
                    for variant_config in self.caption_variant_configs:
                        variant_suffix = variant_config["suffix"]
                        if not variant_suffix.startswith("."):
                            variant_suffix = f".{variant_suffix}"
                        variant_path = img_path.with_suffix(variant_suffix)
                        if variant_path.exists():
                            caption_variant_paths.append(str(variant_path))
                        else:
                            caption_variant_paths.append(None)
                            logger.debug(f"Variant caption not found: {variant_path}")

                # Pre-compute token length for caption-length bucketing
                token_length = 0
                if need_token_length and stored_caption_path:
                    try:
                        raw = json_caption_to_training_text(Path(stored_caption_path).read_text(encoding="utf-8").strip())
                        if "," in raw:
                            token_length = len([t for t in raw.split(",") if t.strip()])
                        else:
                            token_length = len(raw.split())
                    except Exception:
                        token_length = 0

                # 获取图片尺寸
                try:
                    # 使用 lazy loading 只读取 header
                    with Image.open(img_path) as img:
                        original_size = img.size
                except Exception as e:
                    logger.warning(f"Failed to open {img_path}: {e}")
                    continue

                # 计算目标尺寸
                if self.bucket_manager:
                    target_size = self.bucket_manager.get_bucket(*original_size)
                else:
                    target_size = (self.resolution, self.resolution)

                # 计算裁剪坐标 (中心裁剪)
                crop_coords = self._calculate_crop(original_size, target_size)

                samples.append(ImageSample(
                    image_path=str(img_path),
                    caption_path=stored_caption_path,
                    caption_variant_paths=caption_variant_paths,
                    original_size=original_size,
                    target_size=target_size,
                    crop_coords=crop_coords,
                    caption_token_length=token_length,
                ))
        except Exception as e:
            logger.error(f"Error scanning dataset: {e}")
            # If scan fails, we return whatever we found so far or empty

        return samples
    
    def _calculate_crop(
        self,
        original_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """计算中心裁剪坐标"""
        ow, oh = original_size
        tw, th = target_size
        
        # 计算缩放比例
        scale = max(tw / ow, th / oh)
        scaled_w = int(ow * scale)
        scaled_h = int(oh * scale)
        
        # 中心裁剪
        left = (scaled_w - tw) // 2
        top = (scaled_h - th) // 2
        
        return (left, top, left + tw, top + th)

    def _select_caption_variant(self, idx: int) -> int:
        """Select which caption variant to use based on schedule strategy"""
        if not self.caption_variants_enabled or not self.caption_variant_configs:
            return -1  # Use default caption_path

        num_variants = len(self.caption_variant_configs)

        if self.caption_variant_schedule == "alternate":
            # Round-robin by epoch
            return self._variant_epoch_counter % num_variants

        elif self.caption_variant_schedule == "ratio":
            # Weighted random sampling
            if not self.caption_variant_ratio or len(self.caption_variant_ratio) != num_variants:
                return idx % num_variants  # Fallback to round-robin
            weights = self.caption_variant_ratio
            total = sum(weights)
            if total <= 0:
                return 0
            rand_val = random.random() * total
            cumsum = 0.0
            for i, w in enumerate(weights):
                cumsum += w
                if rand_val < cumsum:
                    return i
            return num_variants - 1

        elif self.caption_variant_schedule == "curriculum":
            # Progressive: start with variant 0, gradually shift to later variants
            start_epoch = self.caption_variant_curriculum_start_epoch
            end_epoch = self.caption_variant_curriculum_end_epoch
            if end_epoch <= start_epoch:
                end_epoch = start_epoch + 10  # Fallback to 10 epochs

            if self._current_epoch < start_epoch:
                return 0  # Use first variant before curriculum starts
            elif self._current_epoch >= end_epoch:
                return num_variants - 1  # Use last variant after curriculum ends
            else:
                # Linear progression from variant 0 to variant N-1
                progress = (self._current_epoch - start_epoch) / (end_epoch - start_epoch)
                variant_idx = int(progress * (num_variants - 1))
                return min(variant_idx, num_variants - 1)

        elif self.caption_variant_schedule == "custom":
            # User-defined sequence
            if not self.caption_variant_custom_sequence:
                return idx % num_variants
            seq_len = len(self.caption_variant_custom_sequence)
            if seq_len == 0:
                return 0
            seq_idx = self._current_epoch % seq_len
            variant_idx = int(self.caption_variant_custom_sequence[seq_idx])
            return min(max(variant_idx, 0), num_variants - 1)

        return 0  # Default to first variant

    def _process_caption(
        self,
        caption: str,
        variant_idx: int = -1,
        *,
        structured_tags: Optional[List[str]] = None,
        structured_nl: Optional[List[str]] = None,
        structured_triggers: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """处理 caption，返回 (caption, weight)"""
        # Get variant-specific settings if applicable
        variant_config = None
        if variant_idx >= 0 and variant_idx < len(self.caption_variant_configs):
            variant_config = self.caption_variant_configs[variant_idx]

        # Use variant-specific dropout or global dropout
        tag_dropout = variant_config["dropout"] if variant_config else self.tag_dropout_rate
        should_shuffle = variant_config["shuffle"] if variant_config else self.shuffle_caption

        # Caption Dropout: 以一定概率返回空 caption
        if self.caption_dropout_rate > 0 and random.random() < self.caption_dropout_rate:
            return "", 1.0
        if self.caption_dropout_every_n_epochs > 0:
            if (self._current_epoch + 1) % self.caption_dropout_every_n_epochs == 0:
                return "", 1.0

        if not caption:
            return "", 1.0

        # 分割 tags。结构化 JSON sidecar 可保留 tag/NL 边界，便于只打乱 tag。
        tags = list(structured_tags) if structured_tags is not None else self._split_caption_tags(caption)
        nl_parts = list(structured_nl) if structured_nl else []

        # Tag Dropout: 随机丢弃部分 tag (use variant-specific rate)
        if tag_dropout > 0:
            tags = [t for t in tags if random.random() >= tag_dropout]

        tags = self._apply_targeted_tag_dropout(tags)

        # Weighted Captions: 解析末尾 "weight:X" 格式
        weight = 1.0
        if self.weighted_captions and tags:
            last_tag = tags[-1]
            if ":" in last_tag:
                parts = last_tag.rsplit(":", 1)
                try:
                    weight = float(parts[1])
                    tags = tags[:-1]
                except (ValueError, IndexError):
                    pass

        # Token Warmup: 在 token_warmup_steps 步内，把保留的 tag 数从 token_warmup_min
        # 线性升到 token_warmup_max。只有真正配置了 warmup *日程*（steps > 0）才启用：
        # steps==0 时没有任何渐变意义，且 max 在 trainer 侧被接到 keep_tokens，一旦误触发
        # 就会把 caption 截断到 keep_tokens、静默丢弃 keep_tokens 之后的所有 tag，破坏下方
        # "保留前 N 个再 shuffle" 的策略（该 bug 仅在真实训练 global_step>0 时显形，step=0
        # 的 smoke 夹具照不出来）。
        if self.token_warmup_max > 0 and self.token_warmup_steps > 0 and self._global_step > 0:
            ratio = min(self._global_step / max(self.token_warmup_steps, 1), 1.0)
            keep = int(self.token_warmup_min + (self.token_warmup_max - self.token_warmup_min) * ratio)
            if keep > 0:
                tags = tags[:keep]

        # 保留前 N 个 token（固定位置）
        if self.keep_tokens > 0:
            kept = tags[:self.keep_tokens]
            rest = tags[self.keep_tokens:]

            # 打乱剩余 tags (use variant-specific shuffle)
            if should_shuffle:
                random.shuffle(rest)
            if kept and self.keep_tokens_separator:
                tags = kept + [self.keep_tokens_separator] + rest
            else:
                tags = kept + rest
        elif should_shuffle:
            random.shuffle(tags)

        if self.caption_source_mix.enabled and (structured_tags is not None or structured_nl is not None):
            mix_triggers = merge_trigger_tokens(
                self.caption_source_mix.trigger_tokens,
                structured_triggers or (),
            )
            mix_source = select_caption_source(
                self.caption_source_mix,
                has_tags=bool(tags),
                has_nl=bool(nl_parts),
                has_triggers=bool(mix_triggers),
            )
            if mix_source is not None:
                mixed_tags = remove_trigger_tokens(tags, mix_triggers)
                if self.keep_tokens_separator and mixed_tags:
                    mixed_tags = [
                        token
                        for idx, token in enumerate(mixed_tags)
                        if token != self.keep_tokens_separator or (0 < idx < len(mixed_tags) - 1)
                    ]
                mixed_caption = compose_caption_from_source(
                    mix_source,
                    trigger_tokens=mix_triggers,
                    tags=mixed_tags,
                    nl_parts=nl_parts,
                )
                return mixed_caption, weight

        if self.shuffle_caption_tags_only and nl_parts:
            final_parts = list(tags)
            if final_parts and self.keep_tokens_separator:
                final_parts.append(self.keep_tokens_separator)
            final_parts.extend(nl_parts)
            return ", ".join(part for part in final_parts if str(part or "").strip()), weight

        return ", ".join(tags), weight

    @staticmethod
    def _parse_tag_targets(raw_targets: Any) -> set[str]:
        if not raw_targets:
            return set()
        if isinstance(raw_targets, (list, tuple)):
            items = [str(item).strip() for item in raw_targets]
        else:
            items = [part.strip() for part in re.split(r"[\n,]+", str(raw_targets)) if part.strip()]
        return {item.lower() for item in items if item}

    @staticmethod
    def _split_caption_tags(caption: str) -> List[str]:
        if "," in caption:
            return [t.strip() for t in caption.split(",") if t.strip()]
        return [t.strip() for t in caption.split() if t.strip()]

    def _apply_targeted_tag_dropout(self, tags: List[str]) -> List[str]:
        if not tags or not self.caption_tag_dropout_targets:
            return tags

        matched_indices = [
            index for index, tag in enumerate(tags)
            if tag.strip().lower() in self.caption_tag_dropout_targets
        ]
        if not matched_indices:
            return tags

        mode = self.caption_tag_dropout_target_mode
        if mode == "random_n":
            drop_count = min(len(matched_indices), self.caption_tag_dropout_target_count)
            selected = set(random.sample(matched_indices, drop_count))
            return [tag for index, tag in enumerate(tags) if index not in selected]

        return [tag for index, tag in enumerate(tags) if index not in set(matched_indices)]

    def set_global_step(self, step: int):
        """设置当前训练步数（供 token warmup 使用）"""
        self._global_step = step

    def set_current_epoch(self, epoch: int):
        """Set current epoch for caption policies that operate per epoch."""
        self._current_epoch = max(int(epoch), 0)

    def increment_variant_epoch(self):
        """Increment variant epoch counter for alternate schedule mode."""
        self._variant_epoch_counter += 1
        if self.caption_variants_enabled and self.caption_variant_schedule == "alternate":
            logger.info(f"Caption variant epoch incremented to {self._variant_epoch_counter}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_decode_backend(self, requested: str) -> str:
        aliases = {
            "": "pil",
            "default": "pil",
            "none": "pil",
            "off": "pil",
            "lru": "pil_lru",
            "pil_cache": "pil_lru",
            "cached_pil": "pil_lru",
            "torchvision": "torchvision_cpu",
            "torchvision_io": "torchvision_cpu",
            "torchvision_cpu_decode": "torchvision_cpu",
        }
        backend = aliases.get(str(requested or "pil").replace(" ", ""), requested)
        if backend == "auto":
            return "pil_lru" if self.image_decode_cache_size > 0 else "pil"
        if backend == "pil_lru" and self.image_decode_cache_size <= 0:
            return "pil"
        if backend == "torchvision_cpu":
            return "torchvision_cpu"
        if backend not in {"pil", "pil_lru"}:
            logger.warning("Unknown image_decode_backend=%r; using pil", requested)
            return "pil"
        return backend

    def _image_decode_cache_key(self, image_path: str, need_alpha: bool) -> Tuple[str, int, int, bool]:
        path = Path(image_path)
        stat = path.stat()
        return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size), bool(need_alpha))

    def _load_image_rgb_alpha(self, image_path: str, need_alpha: bool = False) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Load an image as RGB and optionally preserve alpha, with an opt-in PIL LRU."""
        use_cache = self.image_decode_backend == "pil_lru" and self.image_decode_cache_size > 0
        cache_key = self._image_decode_cache_key(image_path, need_alpha) if use_cache else None
        if cache_key is not None:
            cached = self._image_decode_cache.get(cache_key)
            if cached is not None:
                self._image_decode_cache_hits += 1
                self._image_decode_cache.move_to_end(cache_key)
                cached_rgb, cached_alpha = cached
                return cached_rgb.copy(), cached_alpha.copy() if cached_alpha is not None else None

        self._image_decode_cache_misses += 1
        if self.image_decode_backend == "torchvision_cpu":
            try:
                image, alpha_channel = self._load_image_rgb_alpha_torchvision_cpu(image_path, need_alpha=need_alpha)
            except Exception as exc:
                if not self._image_decode_fallback_logged:
                    logger.warning(
                        "image_decode_backend=torchvision_cpu failed (%s); falling back to pil for this worker",
                        exc,
                    )
                    self._image_decode_fallback_logged = True
                with Image.open(image_path) as img:
                    alpha_channel = img.getchannel("A").copy() if need_alpha and "A" in img.getbands() else None
                    image = img.convert("RGB")
        else:
            with Image.open(image_path) as img:
                alpha_channel = img.getchannel("A").copy() if need_alpha and "A" in img.getbands() else None
                image = img.convert("RGB")

        if cache_key is not None:
            self._image_decode_cache[cache_key] = (image.copy(), alpha_channel.copy() if alpha_channel is not None else None)
            self._image_decode_cache.move_to_end(cache_key)
            while len(self._image_decode_cache) > self.image_decode_cache_size:
                self._image_decode_cache.popitem(last=False)

        return image, alpha_channel

    def _load_image_rgb_alpha_torchvision_cpu(
        self,
        image_path: str,
        need_alpha: bool = False,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        """Decode through torchvision on CPU, then return PIL images for existing augments."""
        from torchvision.io import ImageReadMode, read_image
        from torchvision.transforms.functional import to_pil_image

        mode = ImageReadMode.UNCHANGED if need_alpha else ImageReadMode.RGB
        tensor = read_image(str(image_path), mode=mode)
        if tensor.dim() != 3:
            raise ValueError(f"torchvision decoded image must be CHW, got shape={tuple(tensor.shape)}")
        channels = int(tensor.shape[0])
        alpha_channel = None
        if need_alpha and channels == 4:
            alpha_channel = to_pil_image(tensor[3:4]).copy()
            image = to_pil_image(tensor[:3]).convert("RGB")
        elif need_alpha and channels == 2:
            alpha_channel = to_pil_image(tensor[1:2]).copy()
            image = to_pil_image(tensor[:1]).convert("RGB")
        else:
            image = to_pil_image(tensor).convert("RGB")
        return image, alpha_channel
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 加载图片
        image, alpha_channel = self._load_image_rgb_alpha(sample.image_path, need_alpha=self.alpha_mask)

        # 缩放并裁剪
        tw, th = sample.target_size
        scale = max(tw / image.width, th / image.height)
        new_w = int(image.width * scale)
        new_h = int(image.height * scale)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 中心裁剪
        left = (new_w - tw) // 2
        top = (new_h - th) // 2
        image = image.crop((left, top, left + tw, top + th))
        loss_mask = self._load_loss_mask(sample.image_path, alpha_channel, (new_w, new_h), (left, top, left + tw, top + th))

        # Decide the horizontal flip once so the target image, its loss mask and
        # any external control image all flip together (an unflipped mask/control
        # would mirror-misalign the supervision). do_flip stays False when
        # flip_augment is off, keeping the non-augmented path byte-for-byte.
        do_flip = self.flip_augment and random.random() > 0.5
        if do_flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if loss_mask is not None:
                loss_mask = torch.flip(loss_mask, dims=[-1])

        if self._album_pipeline is not None:
            img_np = np.array(image)
            mask_np = loss_mask.numpy() if loss_mask is not None else None
            img_np, mask_np = self._album_pipeline(img_np, mask_np)
            image = Image.fromarray(img_np)
            if mask_np is not None and loss_mask is not None:
                loss_mask = torch.from_numpy(mask_np)

        # Select caption variant and load caption (Lazy Load)
        variant_idx = self._select_caption_variant(idx)
        raw_caption = ""
        structured_tags: Optional[List[str]] = None
        structured_nl: Optional[List[str]] = None
        structured_triggers: Optional[List[str]] = None
        needs_structured_parts = self.shuffle_caption_tags_only or self.caption_source_mix.enabled

        if variant_idx >= 0 and sample.caption_variant_paths:
            # Use variant caption
            if variant_idx < len(sample.caption_variant_paths):
                variant_path = sample.caption_variant_paths[variant_idx]
                if variant_path:
                    try:
                        raw_caption_text = Path(variant_path).read_text(encoding="utf-8").strip()
                        structured = json_caption_to_training_parts(
                            raw_caption_text,
                            dual_caption_enabled=self._dual_caption_enabled,
                            dual_caption_short_key=self._dual_caption_short_key,
                            dual_caption_long_key=self._dual_caption_long_key,
                        )
                        raw_caption = str(structured.get("text") or "")
                        if needs_structured_parts and structured.get("structured"):
                            structured_tags = list(structured.get("tags") or [])
                            structured_nl = list(structured.get("nl") or [])
                            structured_triggers = list(structured.get("triggers") or [])
                    except Exception as e:
                        logger.debug(f"Failed to read variant caption {variant_path}: {e}")
                        raw_caption = ""
        elif sample.caption_path:
            # Use default caption
            try:
                raw_caption_text = Path(sample.caption_path).read_text(encoding="utf-8").strip()
                structured = json_caption_to_training_parts(
                    raw_caption_text,
                    dual_caption_enabled=self._dual_caption_enabled,
                    dual_caption_short_key=self._dual_caption_short_key,
                    dual_caption_long_key=self._dual_caption_long_key,
                )
                raw_caption = str(structured.get("text") or "")
                if needs_structured_parts and structured.get("structured"):
                    structured_tags = list(structured.get("tags") or [])
                    structured_nl = list(structured.get("nl") or [])
                    structured_triggers = list(structured.get("triggers") or [])
            except Exception as e:
                logger.debug(f"Failed to read caption {sample.caption_path}: {e}")
                raw_caption = ""

        caption, caption_weight = self._process_caption(
            raw_caption,
            variant_idx,
            structured_tags=structured_tags,
            structured_nl=structured_nl,
            structured_triggers=structured_triggers,
        )

        # 转换为张量 (Target Image for UNet)
        if self.transform:
            target_image = self.transform(image)
        else:
            target_image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            # 归一化到 [-1, 1]
            target_image = target_image * 2.0 - 1.0
        
        # Prepare guidance image (for IP-Adapter)
        # By default, use the same image. In the future, could support separate files.
        guidance_image = image.copy()
        if self.clip_transform:
            guidance_image = self.clip_transform(guidance_image)
        else:
            # Default CLIP-like transform (224x224) if not provided
            guidance_image = guidance_image.resize((224, 224), Image.LANCZOS)
            guidance_image = torch.from_numpy(np.array(guidance_image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
            # Common CLIP normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            guidance_image = (guidance_image - mean) / std

        # Prepare control image (for ControlNet). Prefer a dedicated
        # conditioning directory, then sibling *_control files, then fallback
        # to the target image for workflows that generate controls in-model.
        control_image = image.copy()
        p = Path(sample.image_path)
        control_candidates = []
        if self.conditioning_data_dir is not None:
            control_candidates.append(self.conditioning_data_dir / p.name)
            control_candidates.append(self.conditioning_data_dir / f"{p.stem}_control{p.suffix}")
            for ext in sorted(self.SUPPORTED_EXTENSIONS):
                control_candidates.append(self.conditioning_data_dir / f"{p.stem}{ext}")
                control_candidates.append(self.conditioning_data_dir / f"{p.stem}_control{ext}")
        control_candidates.append(p.parent / f"{p.stem}_control{p.suffix}")
        control_candidates.append(p.parent / f"{p.stem}_control.png")

        control_path = next((candidate for candidate in control_candidates if candidate.exists()), None)

        if control_path is not None:
            c_img, _ = self._load_image_rgb_alpha(str(control_path), need_alpha=False)
            # Resize/Crop the same way
            c_img = c_img.resize((new_w, new_h), Image.LANCZOS)
            c_img = c_img.crop((left, top, left + tw, top + th))
            if do_flip:
                c_img = c_img.transpose(Image.FLIP_LEFT_RIGHT)
            control_image = c_img
        
        # Convert control image to tensor (usually normalized to [0, 1])
        control_tensor = torch.from_numpy(np.array(control_image.convert("RGB"))).permute(2, 0, 1).float() / 255.0

        item = {
            "image": target_image,
            "guidance_image": guidance_image,
            "control_images": control_tensor,
            "caption": caption,
            "caption_weight": caption_weight,
            "loss_mask": loss_mask,
            "original_size": sample.original_size,
            "target_size": sample.target_size,
            "crop_coords": sample.crop_coords,
            "filenames": sample.image_path,
        }
        if self.easycontrol_v2_enabled:
            item.update(self._load_easycontrol_v2_sidecars(sample.image_path))
        return item

    def _load_easycontrol_v2_sidecars(self, image_path: str) -> Dict[str, Any]:
        if self.easycontrol_v2_spec is None:
            return {}

        plan = sidecar_plan_for_target(image_path, self.easycontrol_v2_spec)
        missing: List[str] = []
        payload: Dict[str, Any] = {
            "easycontrol_v2_plan": plan.to_dict(),
            "easycontrol_v2_missing": missing,
        }

        if not plan.cond_latent_path:
            missing.append(f"{plan.stem}: cond_cache_dir is required")
        else:
            cond = _load_sidecar_tensor(plan.cond_latent_path)
            if cond is None:
                missing.append(f"{plan.stem}: missing cond latent {plan.cond_latent_path}")
            else:
                payload["cond_latents"] = cond
                payload["control_latents"] = cond

        if plan.requires_text_cache:
            if not plan.text_cache_path:
                missing.append(f"{plan.stem}: text_cache_dir is required for colorize")
            else:
                text = _load_sidecar_tensor_or_dict(plan.text_cache_path)
                if text is None:
                    missing.append(f"{plan.stem}: missing text cache {plan.text_cache_path}")
                else:
                    payload["color_text_embeds"] = text

        if plan.control_image_path and not Path(plan.control_image_path).is_file():
            missing.append(f"{plan.stem}: missing control image {plan.control_image_path}")

        return payload

    def _load_loss_mask(
        self,
        image_path: str,
        alpha_channel: Optional[Image.Image],
        resized_size: Tuple[int, int],
        crop_box: Tuple[int, int, int, int],
    ) -> Optional[torch.Tensor]:
        """Load an optional alpha/sidecar mask aligned with the training image."""
        if not (self.masked_loss or self.alpha_mask):
            return None

        mask_img = alpha_channel
        if mask_img is None:
            p = Path(image_path)
            candidates = [
                p.parent / f"{p.stem}_mask{ext}"
                for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
            ]
            mask_path = next((candidate for candidate in candidates if candidate.exists()), None)
            if mask_path is not None:
                try:
                    with Image.open(mask_path) as loaded:
                        mask_img = loaded.convert("L")
                except Exception as e:
                    logger.debug(f"Failed to load mask {mask_path}: {e}")
                    mask_img = None

        if mask_img is None:
            return None

        mask_img = mask_img.resize(resized_size, Image.LANCZOS).crop(crop_box)
        mask = torch.from_numpy(np.array(mask_img.convert("L"))).float().unsqueeze(0) / 255.0
        return mask


def _load_sidecar_tensor(path: str) -> Optional[torch.Tensor]:
    data = _load_sidecar_tensor_or_dict(path)
    if torch.is_tensor(data):
        return data
    if isinstance(data, dict):
        for key in ("cond_latents", "control_latents", "latents", "latent", "samples", "tensor"):
            value = data.get(key)
            if torch.is_tensor(value):
                return value
        tensors = [value for value in data.values() if torch.is_tensor(value)]
        if len(tensors) == 1:
            return tensors[0]
    return None


def _load_sidecar_tensor_or_dict(path: str) -> Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    sidecar = Path(path)
    if not sidecar.is_file():
        return None
    try:
        suffix = sidecar.suffix.lower()
        if suffix == ".safetensors":
            from safetensors.torch import load_file as _load_st

            return _load_st(str(sidecar), device="cpu")
        if suffix in {".pt", ".pth"}:
            loaded = torch.load(str(sidecar), map_location="cpu", weights_only=True)
            if torch.is_tensor(loaded):
                return loaded
            if isinstance(loaded, dict):
                return {str(k): v for k, v in loaded.items() if torch.is_tensor(v)}
    except Exception as e:
        logger.debug("EasyControl v2 sidecar load failed for %s: %s", sidecar, e)
    return None


def _stack_optional_tensors(items: List[Any]) -> Optional[torch.Tensor]:
    if not items or any(not torch.is_tensor(item) for item in items):
        return None
    shapes = {tuple(item.shape) for item in items}
    if len(shapes) != 1:
        return None
    return torch.stack(items)


def _stack_optional_tensor_dicts(items: List[Any]) -> Optional[Dict[str, torch.Tensor]]:
    if not items or any(not isinstance(item, dict) for item in items):
        return None
    keys = set(items[0].keys())
    if any(set(item.keys()) != keys for item in items):
        return None
    stacked: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        value = _stack_optional_tensors([item[key] for item in items])
        if value is None:
            return None
        stacked[key] = value
    return stacked


def collate_fn(batch):
    """自定义 collate 函数，处理不同尺寸的图片"""
    images = torch.stack([item["image"] for item in batch])
    guidance_images = torch.stack([item["guidance_image"] for item in batch])
    control_images = torch.stack([item["control_images"] for item in batch])
    captions = [item["caption"] for item in batch]
    caption_weights = torch.tensor([item.get("caption_weight", 1.0) for item in batch])
    masks = [item.get("loss_mask") for item in batch]
    loss_masks = None
    if any(mask is not None for mask in masks):
        h, w = images.shape[-2:]
        loss_masks = torch.stack([
            mask if mask is not None else torch.ones((1, h, w), dtype=images.dtype)
            for mask in masks
        ])
    original_sizes = [item["original_size"] for item in batch]
    target_sizes = [item["target_size"] for item in batch]
    crop_coords = [item["crop_coords"] for item in batch]
    filenames = [item.get("filenames", "") for item in batch]

    result = {
        "images": images,
        "guidance_images": guidance_images,
        "control_images": control_images,
        "captions": captions,
        "caption_weights": caption_weights,
        "loss_masks": loss_masks,
        "original_sizes": original_sizes,
        "target_sizes": target_sizes,
        "crop_coords": crop_coords,
        "filenames": filenames,
    }
    if any("easycontrol_v2_plan" in item for item in batch):
        result["easycontrol_v2_plans"] = [item.get("easycontrol_v2_plan") for item in batch]
        result["easycontrol_v2_missing"] = [
            missing
            for item in batch
            for missing in list(item.get("easycontrol_v2_missing") or [])
        ]
        cond_latents = _stack_optional_tensors([item.get("cond_latents") for item in batch])
        if cond_latents is not None:
            result["cond_latents"] = cond_latents
            result["control_latents"] = cond_latents

        color_items = [item.get("color_text_embeds") for item in batch]
        color_text_embeds = _stack_optional_tensors(color_items)
        if color_text_embeds is None:
            color_text_embeds = _stack_optional_tensor_dicts(color_items)
        if color_text_embeds is not None:
            result["color_text_embeds"] = color_text_embeds
    return result


class SDXLCacheFirstDataset(Dataset):
    """On-demand SDXL latent/text cache wrapper for CaptionDataset.

    The wrapper intentionally runs in the main process because it owns live
    model modules.  Use ``create_sdxl_cache_first_dataloader`` below instead
    of the generic dataloader helper.
    """

    def __init__(
        self,
        dataset: CaptionDataset,
        *,
        vae: torch.nn.Module,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: torch.nn.Module,
        tokenizer_1: Any,
        tokenizer_2: Any,
        device: torch.device | str,
        dtype: torch.dtype,
        cache_dir: str,
        model_arch: str = "sdxl",
        model_id: str = "",
        cache_latents: bool = True,
        cache_text_encoder_outputs: bool = True,
        latent_disk_format: str = "safetensors",
        latent_disk_dtype: str = "float16",
        text_disk_format: str = "safetensors",
        text_disk_dtype: str = "float16",
        keep_vae_on_cpu: bool = True,
        keep_text_encoders_on_cpu: bool = True,
        use_model_to_condition: bool = True,
    ) -> None:
        self.dataset = dataset
        self.samples = getattr(dataset, "samples", [])
        self.bucket_manager = getattr(dataset, "bucket_manager", None)
        self.caption_length_bucket_size = getattr(dataset, "caption_length_bucket_size", 0)
        self.vae = vae
        self.text_encoder_1 = text_encoder_1
        self.text_encoder_2 = text_encoder_2
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.device = torch.device(device)
        self.dtype = dtype
        self.model_arch = str(model_arch or "sdxl").strip().lower()
        self.model_id = model_id
        self.keep_vae_on_cpu = bool(keep_vae_on_cpu)
        self.keep_text_encoders_on_cpu = bool(keep_text_encoders_on_cpu)
        self.use_model_to_condition = bool(use_model_to_condition)
        root = Path(cache_dir)
        self.latent_cache = LatentDiskCache(
            str(root / "latents"),
            enabled=bool(cache_latents),
            disk_format=latent_disk_format,
            disk_dtype=latent_disk_dtype,
        )
        self.text_cache = TextEncoderDiskCache(
            str(root / "text_encoder"),
            enabled=bool(cache_text_encoder_outputs),
            disk_format=text_disk_format,
            disk_dtype=text_disk_dtype,
        )
        self.condition_builder = SDXLModelToCondition(
            model_id=self.model_id,
            dtype=str(self.dtype).replace("torch.", ""),
            family=self.model_arch,
            encode_latents=self._encode_latents,
            encode_text=self._encode_text,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def set_current_epoch(self, epoch: int):
        if hasattr(self.dataset, "set_current_epoch"):
            self.dataset.set_current_epoch(epoch)

    def set_global_step(self, step: int):
        if hasattr(self.dataset, "set_global_step"):
            self.dataset.set_global_step(step)

    def increment_variant_epoch(self):
        if hasattr(self.dataset, "increment_variant_epoch"):
            self.dataset.increment_variant_epoch()

    @staticmethod
    def _as_sample_tensor(value: torch.Tensor) -> torch.Tensor:
        value = value.detach().cpu()
        return value[0] if value.dim() >= 1 and value.shape[0] == 1 else value

    def _runtime_context(self, module: Optional[torch.nn.Module], *, dtype: torch.dtype):
        from .device_state import module_runtime_state

        keep_cpu = self.keep_text_encoders_on_cpu
        if module is self.vae:
            keep_cpu = self.keep_vae_on_cpu
        if keep_cpu:
            return module_runtime_state(module, device=self.device, dtype=dtype)

        class _Noop:
            def __enter__(self_inner):
                if module is not None:
                    module.to(device=self.device, dtype=dtype)
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Noop()

    def _encode_latents(self, item: Dict[str, Any]) -> torch.Tensor:
        filename = str(item.get("filenames") or "")
        target_size = tuple(int(v) for v in item.get("target_size", (0, 0)))
        cached = self.latent_cache.get(filename, target_size)
        if cached and "latents" in cached:
            return cached["latents"].float()

        image = item["image"].unsqueeze(0)
        with self._runtime_context(self.vae, dtype=torch.float32):
            with torch.no_grad():
                latents = self.vae.encode(image.to(device=self.device, dtype=torch.float32)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
                latents = self._as_sample_tensor(latents)
        self.latent_cache.put(filename, target_size, {"latents": latents})
        return latents.float()

    def _encode_text(self, caption: str) -> Dict[str, torch.Tensor]:
        cached = self.text_cache.get(caption, self.model_id)
        if cached and "encoder_hidden_states" in cached:
            return {key: value.float() if value.is_floating_point() else value for key, value in cached.items()}

        with self._runtime_context(self.text_encoder_1, dtype=self.dtype), self._runtime_context(self.text_encoder_2, dtype=self.dtype):
            with torch.no_grad():
                tokens_1 = self.tokenizer_1(
                    [caption],
                    padding="max_length",
                    max_length=self.tokenizer_1.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                out_1 = self.text_encoder_1(
                    tokens_1.input_ids.to(self.device),
                    output_hidden_states=True,
                )
                if self.text_encoder_2 is not None and self.tokenizer_2 is not None:
                    tokens_2 = self.tokenizer_2(
                        [caption],
                        padding="max_length",
                        max_length=self.tokenizer_2.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    out_2 = self.text_encoder_2(
                        tokens_2.input_ids.to(self.device),
                        output_hidden_states=True,
                    )
                    hidden = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
                    pooled = out_2.text_embeds
                else:
                    hidden = out_1.last_hidden_state
                    pooled = None

        result = {
            "encoder_hidden_states": self._as_sample_tensor(hidden),
        }
        if pooled is not None:
            result["pooled_prompt_embeds"] = self._as_sample_tensor(pooled)
        self.text_cache.put(caption, result, self.model_id)
        return {key: value.float() if value.is_floating_point() else value for key, value in result.items()}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if self.use_model_to_condition:
            return self.condition_builder.build(item).as_training_item()

        caption = str(item.get("caption") or "")
        text = self._encode_text(caption)
        result = {
            "latents": self._encode_latents(item),
            "encoder_hidden_states": text["encoder_hidden_states"],
            "caption": caption,
            "caption_weight": item.get("caption_weight", 1.0),
            "loss_mask": item.get("loss_mask"),
            "original_size": item.get("original_size"),
            "target_size": item.get("target_size"),
            "crop_coords": item.get("crop_coords"),
            "filenames": item.get("filenames", ""),
        }
        if text.get("pooled_prompt_embeds") is not None:
            result["pooled_prompt_embeds"] = text["pooled_prompt_embeds"]
        return result


def sdxl_cache_first_collate(batch):
    latents = torch.stack([item["latents"] for item in batch])
    encoder_hidden_states = torch.stack([item["encoder_hidden_states"] for item in batch])
    pooled_prompt_embeds = None
    if all(item.get("pooled_prompt_embeds") is not None for item in batch):
        pooled_prompt_embeds = torch.stack([item["pooled_prompt_embeds"] for item in batch])
    caption_weights = torch.tensor([item.get("caption_weight", 1.0) for item in batch])
    masks = [item.get("loss_mask") for item in batch]
    loss_masks = None
    if any(mask is not None for mask in masks):
        h, w = latents.shape[-2] * 8, latents.shape[-1] * 8
        loss_masks = torch.stack([
            mask if mask is not None else torch.ones((1, h, w), dtype=torch.float32)
            for mask in masks
        ])
    result = {
        "latents": latents,
        "encoder_hidden_states": encoder_hidden_states,
        "captions": [item.get("caption", "") for item in batch],
        "caption_weights": caption_weights,
        "loss_masks": loss_masks,
        "original_sizes": [item.get("original_size", (1024, 1024)) for item in batch],
        "target_sizes": [item.get("target_size", (1024, 1024)) for item in batch],
        "crop_coords": [item.get("crop_coords", (0, 0, 1024, 1024)) for item in batch],
        "filenames": [item.get("filenames", "") for item in batch],
    }
    if pooled_prompt_embeds is not None:
        result["pooled_prompt_embeds"] = pooled_prompt_embeds
    return result


def create_sdxl_cache_first_dataloader(
    dataset: SDXLCacheFirstDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create a main-process dataloader for SDXL cache-first batches."""
    from .dataloader_rebuild_runtime import attach_dataloader_rebuild_descriptor
    from .multi_batch_contract import attach_dataloader_batching_contract
    from .training_data_pipeline_stage import attach_lulynx_dataloader_data_pipeline_report

    def attach_pipeline_report(dataloader: DataLoader) -> DataLoader:
        attached = attach_dataloader_batching_contract(dataloader, requested_physical_batch_size=batch_size)
        return attach_lulynx_dataloader_data_pipeline_report(
            attached,
            requested_physical_batch_size=batch_size,
            route="sdxl_cache_first",
            required_fields=("latents", "encoder_hidden_states", "captions"),
        )

    def rebuild_factory(descriptor):
        return create_sdxl_cache_first_dataloader(
            dataset,
            batch_size=max(int(descriptor.get("batch_size", batch_size) or 1), 1),
            shuffle=bool(descriptor.get("shuffle", shuffle)),
            pin_memory=bool(descriptor.get("pin_memory", pin_memory)),
            drop_last=bool(descriptor.get("drop_last", drop_last)),
        )

    if dataset.bucket_manager and batch_size > 1:
        attached = attach_dataloader_rebuild_descriptor(
            DataLoader(
                dataset,
                batch_sampler=BucketBatchSampler(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    caption_length_bucket_size=getattr(dataset, "caption_length_bucket_size", 0),
                    drop_last=drop_last,
                ),
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=sdxl_cache_first_collate,
            ),
            route="sdxl_cache_first",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
            persistent_workers=False,
            pin_memory=pin_memory,
            prefetch_factor=None,
            uses_batch_sampler=True,
            rebuild_factory=rebuild_factory,
            mutable_descriptor_fields=("pin_memory",),
        )
        return attach_pipeline_report(attached)

    attached = attach_dataloader_rebuild_descriptor(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
            pin_memory=pin_memory,
            collate_fn=sdxl_cache_first_collate,
        ),
        route="sdxl_cache_first",
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        persistent_workers=False,
        pin_memory=pin_memory,
        prefetch_factor=None,
        uses_batch_sampler=False,
        rebuild_factory=rebuild_factory,
        mutable_descriptor_fields=("pin_memory",),
    )
    return attach_pipeline_report(attached)


class BucketBatchSampler:
    """Group samples with the same target size (and optionally caption-length bucket) into the same batch."""

    def __init__(
        self,
        dataset: CaptionDataset,
        batch_size: int,
        shuffle: bool = True,
        caption_length_bucket_size: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.caption_length_bucket_size = caption_length_bucket_size
        self.drop_last = drop_last

    def _bucket_key(self, sample) -> tuple:
        """Return the grouping key for a sample."""
        if self.caption_length_bucket_size > 0:
            tok_len = getattr(sample, "caption_token_length", 0) or 0
            length_bucket = (tok_len // self.caption_length_bucket_size) * self.caption_length_bucket_size
            return (sample.target_size, length_bucket)
        return (sample.target_size,)

    def __iter__(self):
        buckets: Dict[tuple, List[int]] = {}
        for index, sample in enumerate(self.dataset.samples):
            buckets.setdefault(self._bucket_key(sample), []).append(index)

        batches: List[List[int]] = []
        for indices in buckets.values():
            if self.shuffle:
                random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batch = indices[offset:offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        yield from batches

    def __len__(self) -> int:
        buckets: Dict[tuple, int] = {}
        for sample in self.dataset.samples:
            key = self._bucket_key(sample)
            buckets[key] = buckets.get(key, 0) + 1
        if self.drop_last:
            return sum(count // self.batch_size for count in buckets.values())
        return sum((count + self.batch_size - 1) // self.batch_size for count in buckets.values())

class TextEncoderDiskCache:
    """
    Text Encoder 输出磁盘缓存

    将 TE 编码结果缓存到磁盘，避免每个 epoch 重复编码。
    缓存格式: .npz / .safetensors / .pt — controlled by ``disk_format``.
    Cached tensors are cast to ``disk_dtype`` on save and back to float32 on
    load (caller can downcast as needed).
    """

    _ALLOWED_FORMATS = {"npz", "safetensors", "pt"}
    _DTYPE_MAP = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    _AUTO_DTYPE_ALIASES = {"", "auto", "default"}

    def __init__(
        self,
        cache_dir: str,
        enabled: bool = True,
        *,
        disk_format: str = "npz",
        disk_dtype: str = "float16",
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        fmt = (disk_format or "npz").lower()
        if fmt not in self._ALLOWED_FORMATS:
            raise ValueError(
                f"TextEncoderDiskCache: unsupported disk_format '{disk_format}'. "
                f"Allowed: {sorted(self._ALLOWED_FORMATS)}"
            )
        self.disk_format = fmt
        dtype_key = str(disk_dtype or "").strip().lower()
        if dtype_key in self._AUTO_DTYPE_ALIASES:
            dtype_key = "float16"
        if dtype_key not in self._DTYPE_MAP:
            raise ValueError(
                f"TextEncoderDiskCache: unsupported disk_dtype '{disk_dtype}'. "
                f"Allowed: {sorted(self._DTYPE_MAP)}"
            )
        self.disk_dtype = self._DTYPE_MAP[dtype_key]
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, caption: str, model_id: str = "") -> str:
        """根据 caption 内容生成缓存键"""
        content = f"{model_id}:{caption}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        ext = {"npz": ".npz", "safetensors": ".safetensors", "pt": ".pt"}[self.disk_format]
        return self.cache_dir / f"te_{key}{ext}"

    def get(self, caption: str, model_id: str = "") -> Optional[Dict[str, torch.Tensor]]:
        """从磁盘加载缓存的 TE 输出"""
        if not self.enabled:
            return None

        key = self._hash_key(caption, model_id)
        path = self._cache_path(key)

        if not path.exists():
            return None

        try:
            if self.disk_format == "npz":
                data = np.load(str(path))
                return {k: torch.from_numpy(v) for k, v in data.items()}
            if self.disk_format == "safetensors":
                from safetensors.torch import load_file as _load_st
                return _load_st(str(path))
            # pt
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            logger.debug(f"TE cache load failed for {key}: {e}")
            return None

    def put(self, caption: str, outputs: Dict[str, torch.Tensor], model_id: str = "") -> None:
        """将 TE 输出保存到磁盘"""
        if not self.enabled:
            return

        key = self._hash_key(caption, model_id)
        path = self._cache_path(key)

        try:
            cast = {
                k: (v.detach().to(self.disk_dtype).cpu()
                    if v.is_floating_point() else v.detach().cpu())
                for k, v in outputs.items()
            }
            if self.disk_format == "npz":
                # numpy doesn't support bfloat16 — fall back to float16 on disk
                np.savez(
                    str(path),
                    **{
                        k: (v.to(torch.float16).numpy() if v.dtype == torch.bfloat16 else v.numpy())
                        for k, v in cast.items()
                    },
                )
            elif self.disk_format == "safetensors":
                from safetensors.torch import save_file as _save_st
                _save_st(cast, str(path))
            else:  # pt
                torch.save(cast, str(path))
        except Exception as e:
            logger.debug(f"TE cache save failed for {key}: {e}")

    def clear(self) -> int:
        """清理所有缓存文件"""
        count = 0
        patterns = ("te_*.npz", "te_*.safetensors", "te_*.pt")
        for pat in patterns:
            for f in self.cache_dir.glob(pat):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count


class LatentDiskCache:
    """Per-image VAE-latent disk cache with configurable format & dtype.

    Mirrors :class:`TextEncoderDiskCache` but for image latents.  The cache
    key is a hash of ``(image_path, resolution)`` so the same image at a
    different bucket size lands in a separate file.
    """

    _ALLOWED_FORMATS = {"npz", "safetensors", "pt"}
    _DTYPE_MAP = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    _AUTO_DTYPE_ALIASES = {"", "auto", "default"}

    def __init__(
        self,
        cache_dir: str,
        enabled: bool = True,
        *,
        disk_format: str = "npz",
        disk_dtype: str = "float16",
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        fmt = (disk_format or "npz").lower()
        if fmt not in self._ALLOWED_FORMATS:
            raise ValueError(
                f"LatentDiskCache: unsupported disk_format '{disk_format}'. "
                f"Allowed: {sorted(self._ALLOWED_FORMATS)}"
            )
        self.disk_format = fmt
        dtype_key = str(disk_dtype or "").strip().lower()
        if dtype_key in self._AUTO_DTYPE_ALIASES:
            dtype_key = "float16"
        if dtype_key not in self._DTYPE_MAP:
            raise ValueError(
                f"LatentDiskCache: unsupported disk_dtype '{disk_dtype}'. "
                f"Allowed: {sorted(self._DTYPE_MAP)}"
            )
        self.disk_dtype = self._DTYPE_MAP[dtype_key]
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, image_path: str, resolution: tuple[int, int]) -> str:
        h = hashlib.sha256(f"{image_path}:{resolution[0]}x{resolution[1]}".encode("utf-8"))
        return h.hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        ext = {"npz": ".npz", "safetensors": ".safetensors", "pt": ".pt"}[self.disk_format]
        return self.cache_dir / f"lat_{key}{ext}"

    def get(self, image_path: str, resolution: tuple[int, int]) -> Optional[Dict[str, torch.Tensor]]:
        if not self.enabled:
            return None
        path = self._cache_path(self._hash_key(image_path, resolution))
        if not path.exists():
            return None
        try:
            if self.disk_format == "npz":
                data = np.load(str(path))
                return {k: torch.from_numpy(v) for k, v in data.items()}
            if self.disk_format == "safetensors":
                from safetensors.torch import load_file as _load_st
                return _load_st(str(path))
            return torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as e:
            logger.debug(f"latent cache load failed for {path.name}: {e}")
            return None

    def put(
        self,
        image_path: str,
        resolution: tuple[int, int],
        latents: Dict[str, torch.Tensor],
    ) -> None:
        if not self.enabled:
            return
        path = self._cache_path(self._hash_key(image_path, resolution))
        try:
            cast = {
                k: (v.detach().to(self.disk_dtype).cpu()
                    if v.is_floating_point() else v.detach().cpu())
                for k, v in latents.items()
            }
            if self.disk_format == "npz":
                np.savez(
                    str(path),
                    **{
                        k: (v.to(torch.float16).numpy() if v.dtype == torch.bfloat16 else v.numpy())
                        for k, v in cast.items()
                    },
                )
            elif self.disk_format == "safetensors":
                from safetensors.torch import save_file as _save_st
                _save_st(cast, str(path))
            else:
                torch.save(cast, str(path))
        except Exception as e:
            logger.debug(f"latent cache save failed for {path.name}: {e}")

    def clear(self) -> int:
        count = 0
        for pat in ("lat_*.npz", "lat_*.safetensors", "lat_*.pt"):
            for f in self.cache_dir.glob(pat):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count


def create_dataloader(
    dataset: CaptionDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """创建 DataLoader，支持 pin_memory 和 prefetch_factor 优化"""
    from .dataloader_rebuild_runtime import attach_dataloader_rebuild_descriptor
    from .multi_batch_contract import attach_dataloader_batching_contract
    from .training_data_pipeline_stage import attach_lulynx_dataloader_data_pipeline_report

    def rebuild_factory(descriptor):
        workers = max(int(descriptor.get("num_workers", num_workers) or 0), 0)
        return create_dataloader(
            dataset,
            batch_size=max(int(descriptor.get("batch_size", batch_size) or 1), 1),
            shuffle=bool(descriptor.get("shuffle", shuffle)),
            num_workers=workers,
            pin_memory=bool(descriptor.get("pin_memory", pin_memory)),
            prefetch_factor=max(int(descriptor.get("prefetch_factor", prefetch_factor) or 1), 1),
            persistent_workers=bool(descriptor.get("persistent_workers", persistent_workers)) and workers > 0,
            drop_last=bool(descriptor.get("drop_last", drop_last)),
        )

    def attach_rebuild(dataloader: DataLoader, *, uses_batch_sampler: bool) -> DataLoader:
        attached = attach_dataloader_rebuild_descriptor(
            dataloader,
            route="caption",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            uses_batch_sampler=uses_batch_sampler,
            rebuild_factory=rebuild_factory,
        )
        attached = attach_dataloader_batching_contract(attached, requested_physical_batch_size=batch_size)
        return attach_lulynx_dataloader_data_pipeline_report(
            attached,
            requested_physical_batch_size=batch_size,
            route="caption",
            required_fields=("images", "captions"),
        )

    # Worker-only DataLoader options are invalid when num_workers == 0.
    dl_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor
        dl_kwargs["persistent_workers"] = persistent_workers

    if dataset.bucket_manager and batch_size > 1:
        caption_bucket = getattr(dataset, "caption_length_bucket_size", 0)
        dataloader = DataLoader(
            dataset,
            batch_sampler=BucketBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                caption_length_bucket_size=caption_bucket,
                drop_last=drop_last,
            ),
            **dl_kwargs,
        )
        attached = attach_rebuild(dataloader, uses_batch_sampler=True)
        wrapped = _attach_native_shadow_adapter_if_enabled(
            attached,
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return wrapped if wrapped is attached else attach_rebuild(wrapped, uses_batch_sampler=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **dl_kwargs,
    )
    attached = attach_rebuild(dataloader, uses_batch_sampler=False)
    wrapped = _attach_native_shadow_adapter_if_enabled(
        attached,
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return wrapped if wrapped is attached else attach_rebuild(wrapped, uses_batch_sampler=False)


def split_dataloader(
    dl: DataLoader,
    fraction: float,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split a DataLoader into train and validation subsets.

    Args:
        dl: The original DataLoader whose underlying dataset will be split.
        fraction: Fraction of data to allocate to validation (0 < fraction < 1).
        seed: Random seed for reproducible splits.

    Returns:
        A (train_dataloader, val_dataloader) tuple.  The validation loader
        does **not** shuffle; the train loader preserves the original
        shuffle setting.
    """
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(f"validation_split must be in (0, 1), got {fraction}")

    dataset = dl.dataset
    # Unwrap Subset if the original DataLoader already wraps one
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    total = len(dataset)
    val_count = max(1, int(total * fraction))

    # Deterministic shuffle of indices
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    # Preserve the original DataLoader settings
    original_kwargs: dict = {
        "num_workers": dl.num_workers,
        "pin_memory": dl.pin_memory,
        "collate_fn": getattr(dl, "collate_fn", None) or collate_fn,
    }
    if dl.num_workers > 0:
        original_kwargs["prefetch_factor"] = getattr(dl, "prefetch_factor", 2)
        original_kwargs["persistent_workers"] = getattr(dl, "persistent_workers", False)

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # Extract batch size from the existing DataLoader
    batch_size = 1
    if hasattr(dl, "batch_sampler") and dl.batch_sampler is not None:
        batch_size = getattr(dl.batch_sampler, "batch_size", 1)

    # For bucket-based datasets build a simple shuffled/sequential loader.
    # We skip BucketBatchSampler for Subset because it expects direct
    # access to .samples which Subset does not expose.
    train_dl = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        **original_kwargs,
    )
    val_dl = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        **original_kwargs,
    )

    logger.info(
        "Dataset split: %d train / %d validation (fraction=%.3f)",
        len(train_indices),
        len(val_indices),
        fraction,
    )
    return train_dl, val_dl

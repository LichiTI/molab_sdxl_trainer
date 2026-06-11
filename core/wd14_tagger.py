from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("WD14Tagger")
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning('[WD14] PIL not available')
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.info('[WD14] pandas not available, will use csv fallback')
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning('[WD14] huggingface_hub not available')

class WD14Tagger:
    MODELS = {'wd-vit-v3': 'SmilingWolf/wd-vit-tagger-v3', 'wd-convnext-v3': 'SmilingWolf/wd-convnext-tagger-v3', 'wd-swinv2-v3': 'SmilingWolf/wd-swinv2-tagger-v3', 'wd-eva02-large-v3': 'SmilingWolf/wd-eva02-large-tagger-v3'}
    MODEL_ALIASES = {
        'wd-vit-tagger-v3': 'wd-vit-v3',
        'wd-convnext-tagger-v3': 'wd-convnext-v3',
        'wd-swinv2-tagger-v3': 'wd-swinv2-v3',
        'wd-eva02-large-tagger-v3': 'wd-eva02-large-v3',
        'wd14-vit-v2': 'wd-vit-v3',
        'wd14-convnextv2-v2': 'wd-convnext-v3',
        'wd14-swinv2-v2': 'wd-swinv2-v3',
        'wd14-moat-v2': 'wd-convnext-v3',
        'wd-vit-large-tagger-v3': 'wd-vit-v3',
        'eva02_large_E621_FULL_V1': 'wd-eva02-large-v3',
        'cl_tagger_1_01': 'wd-convnext-v3',
    }
    MODEL_FILENAME = 'model.onnx'
    LABEL_FILENAME = 'selected_tags.csv'

    def __init__(self, model_name: str='wd-eva02-large-v3', cache_dir: Optional[str]=None):
        self.model_name = self.normalize_model_name(model_name)
        self.cache_dir = cache_dir
        self.model = None
        self.labels: List[str] = []
        self.rating_labels: List[str] = []
        self.tag_labels: List[str] = []
        if self.model_name not in self.MODELS:
            raise ValueError(f'Unknown model: {model_name}. Available: {list(self.MODELS.keys())}')
        self.repo_id = self.MODELS[self.model_name]

    @classmethod
    def normalize_model_name(cls, model_name: Optional[str]) -> str:
        if not model_name:
            return 'wd-eva02-large-v3'
        return cls.MODEL_ALIASES.get(model_name, model_name)

    def load(self) -> bool:
        if not HF_AVAILABLE:
            logger.error('[WD14] huggingface_hub not installed. Run: pip install huggingface_hub')
            return False
        try:
            logger.info(f'[WD14] Downloading model from {self.repo_id}...')
            model_path = hf_hub_download(repo_id=self.repo_id, filename=self.MODEL_FILENAME, cache_dir=self.cache_dir)
            label_path = hf_hub_download(repo_id=self.repo_id, filename=self.LABEL_FILENAME, cache_dir=self.cache_dir)
            self._load_onnx_model(model_path)
            self._load_labels(label_path)
            logger.info(f'[WD14] Model loaded: {self.model_name}')
            logger.info(f'[WD14] Total tags: {len(self.labels)} (ratings: {len(self.rating_labels)}, tags: {len(self.tag_labels)})')
            return True
        except Exception as e:
            logger.error(f'[WD14] Failed to load model: {e}')
            return False

    def _load_onnx_model(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                'WD14 requires onnxruntime. Open Launcher > Runtime and run "Install WD14 deps" for the selected runtime.'
            ) from exc
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]
        logger.info(f'[WD14] Using providers: {providers}')
        self.model = ort.InferenceSession(model_path, providers=providers)

    def _load_labels(self, label_path: str):
        if PANDAS_AVAILABLE:
            import pandas as pd
            df = pd.read_csv(label_path)
            self.labels = df['name'].tolist()
        else:
            import csv
            with open(label_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.labels = [row['name'] for row in reader]
        self.rating_labels = self.labels[:4]
        self.tag_labels = self.labels[4:]

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            logger.info(f'[WD14] Model unloaded: {self.model_name}')

    def tag_image(self, image, threshold: float=0.35, character_threshold: float=0.85, exclude_tags: List[str]=None, replace_underscore: bool=True) -> Tuple[Dict[str, float], Dict[str, float]]:
        if self.model is None:
            if not self.load():
                raise RuntimeError(f'WD14 model failed to load: {self.model_name}')
        if exclude_tags is None:
            exclude_tags = []
        exclude_set = set()
        for tag in exclude_tags:
            tag_text = str(tag).strip().lower()
            if not tag_text:
                continue
            exclude_set.add(tag_text)
            exclude_set.add(tag_text.replace(' ', '_'))
            exclude_set.add(tag_text.replace('_', ' '))
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image_np = self._preprocess_image(image)
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        probs = self.model.run([output_name], {input_name: image_np})[0][0]
        ratings = {}
        for i, label in enumerate(self.rating_labels):
            ratings[label] = float(probs[i])
        tags = {}
        for i, label in enumerate(self.tag_labels):
            prob = float(probs[i + 4])
            if prob < threshold:
                continue
            if label.lower() in exclude_set:
                continue
            tag_name = label
            if replace_underscore:
                tag_name = tag_name.replace('_', ' ')
            tags[tag_name] = prob
        tags = dict(sorted(tags.items(), key=lambda x: x[1], reverse=True))
        return (ratings, tags)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        _, height, width, _ = self.model.get_inputs()[0].shape
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                new_image = Image.new('RGB', image.size, (255, 255, 255))
                new_image.paste(image, mask=image.split()[3])
                image = new_image
            else:
                image = image.convert('RGB')
        image = self._smart_resize(image, (width, height))
        image_np = np.array(image, dtype=np.float32)
        image_np = image_np[:, :, ::-1]
        image_np = np.expand_dims(image_np, 0)
        return image_np

    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        w, h = image.size
        target_w, target_h = target_size
        if w != h:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return image

    def tag_to_string(self, tags: Dict[str, float], separator: str=', ', include_confidence: bool=False) -> str:
        if include_confidence:
            return separator.join((f'({tag}:{conf:.2f})' for tag, conf in tags.items()))
        else:
            return separator.join(tags.keys())

    def interrogate(
        self,
        image,
        threshold: float = 0.35,
        character_threshold: float = 0.85,
        exclude_tags: Optional[List[str]] = None,
        replace_underscore: bool = True,
        separator: str = ', ',
    ) -> str:
        _, tags = self.tag_image(
            image,
            threshold=threshold,
            character_threshold=character_threshold,
            exclude_tags=exclude_tags,
            replace_underscore=replace_underscore,
        )
        return self.tag_to_string(tags, separator=separator, include_confidence=False)

def get_available_models() -> List[str]:
    return list(WD14Tagger.MODELS.keys())

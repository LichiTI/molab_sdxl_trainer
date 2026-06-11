import random
from pathlib import Path
import logging

logger = logging.getLogger("TagProcessor")
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    logger.info('[INFO] BLIP not available. Will use fallback tagging.')
try:
    from core.wd14_tagger import WD14Tagger, get_available_models
    WD14_AVAILABLE = True
except ImportError:
    WD14_AVAILABLE = False
    logger.info('[INFO] WD14 Tagger not available.')

class BLIPTagger:

    def __init__(self, model_path=None):
        self.model = None
        self.processor = None
        self.device = 'cpu'
        if TORCH_AVAILABLE and BLIP_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._load_model(model_path)

    def _load_model(self, model_path=None):
        try:
            model_name = model_path or 'Salesforce/blip-image-captioning-base'
            logger.info(f'[BLIPTagger] Loading model: {model_name}')
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'[BLIPTagger] Model loaded on {self.device}')
        except Exception as e:
            logger.error(f'[BLIPTagger] Failed to load model: {e}')
            self.model = None

    def generate_caption(self, image) -> str:
        if self.model is None:
            return self._fallback_tags()
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            elif TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                image = Image.fromarray(image.cpu().numpy().astype('uint8'))
            inputs = self.processor(image, return_tensors='pt').to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f'[BLIPTagger] Error generating caption: {e}')
            return self._fallback_tags()

    def _fallback_tags(self) -> str:
        tags = ['detailed', 'high quality', 'best quality']
        return ', '.join(tags)

class LLMTagger:
    """Uses the shared LLMEngine for advanced image captioning."""
    def __init__(self, provider: str = "gemini", model: str = "", api_key: str = "", base_url: str = ""):
        from .llm_client import LLMClient
        self.client = LLMClient(provider=provider, api_key=api_key, base_url=base_url, model=model)
        
    def generate_caption(self, image: Image.Image) -> str:
        system_prompt = "You are an expert image captioner for Stable Diffusion. Describe this image in a single, detailed sentence focusing on style, subjects, and atmosphere."
        user_prompt = "Describe this image for a prompt."
        
        try:
            caption = self.client.chat_with_image(image, system_prompt, user_prompt)
            return caption or ""
        except Exception as e:
            logger.error(f'[LLMTagger] Error: {e}')
            return ""

class TagProcessor:

    def __init__(
        self,
        method: str='blip',
        model_name: str='wd-vit-v3',
        llm_provider: str='gemini',
        llm_model: str='',
        llm_api_key: str='',
        llm_base_url: str='',
    ):
        self.method = method
        self.blip_tagger = None
        self.wd14_tagger = None
        self.llm_tagger = None
        if method == 'blip' and TORCH_AVAILABLE and BLIP_AVAILABLE:
            self.blip_tagger = BLIPTagger()
        elif method == 'wd14' and WD14_AVAILABLE:
            self.wd14_tagger = WD14Tagger(model_name)
        elif method == 'llm':
            if not llm_api_key and llm_provider not in ('ollama', 'local_llm'):
                logger.warning('[TagProcessor] LLM API key missing; LLM tagging may fail.')
            self.llm_tagger = LLMTagger(provider=llm_provider, model=llm_model, api_key=llm_api_key, base_url=llm_base_url)

    def process(self, images, filenames, config, progress_callback=None):
        trigger_word = config.get('trigger_word', '')
        blacklist = config.get('blacklist', '')
        ordering = config.get('ordering', 'Original')
        threshold = config.get('threshold', 0.35)
        blacklist_set = set([b.strip().lower() for b in blacklist.replace('\n', ',').split(',') if b.strip()])
        result = {}
        total = len(filenames)
        for i, filename in enumerate(filenames):
            if i < len(images):
                image = images[i]
            else:
                image = None
            if self.method == 'wd14' and self.wd14_tagger is not None and (image is not None):
                _, tags_dict = self.wd14_tagger.tag_image(image, threshold=threshold, exclude_tags=list(blacklist_set))
                tags = list(tags_dict.keys())
            elif self.method == 'blip' and self.blip_tagger is not None and (image is not None):
                caption = self.blip_tagger.generate_caption(image)
                tags = [t.strip() for t in caption.replace(',', ' ').split() if t.strip()]
                tags = [t for t in tags if t.lower() not in blacklist_set]
            elif self.method == 'llm' and self.llm_tagger is not None and (image is not None):
                caption = self.llm_tagger.generate_caption(image)
                tags = [caption] if caption else []
                tags = [t for t in tags if t.lower() not in blacklist_set]
            else:
                tags = ['furry', 'solo', 'detailed', 'high quality']
            if ordering == 'Shuffle':
                random.shuffle(tags)
            elif ordering == 'Alphabetical':
                tags.sort()
            if trigger_word and trigger_word.strip():
                tw = trigger_word.strip()
                tags = [x for x in tags if x.lower() != tw.lower()]
                tags.insert(0, tw)
            result[filename] = ', '.join(tags)
            if progress_callback:
                progress_callback(i + 1, total)
        return result

    def process_from_paths(self, file_paths, filenames, config, progress_callback=None):
        """Process images from file paths with lazy loading to prevent RAM explosion.
        
        Unlike process(), this loads images one by one instead of requiring all images
        to be pre-loaded into RAM. This is critical for handling large datasets (1000+ images).
        """
        trigger_word = config.get('trigger_word', '')
        blacklist = config.get('blacklist', '')
        ordering = config.get('ordering', 'Original')
        threshold = config.get('threshold', 0.35)
        blacklist_set = set([b.strip().lower() for b in blacklist.replace('\n', ',').split(',') if b.strip()])
        result = {}
        total = len(filenames)
        for i, (fpath, filename) in enumerate(zip(file_paths, filenames)):
            # Lazy load image on-demand
            image = None
            try:
                with Image.open(fpath) as img:
                    image = img.convert('RGB')
                    # Process immediately while image is loaded
                    if self.method == 'wd14' and self.wd14_tagger is not None:
                        _, tags_dict = self.wd14_tagger.tag_image(image, threshold=threshold, exclude_tags=list(blacklist_set))
                        tags = list(tags_dict.keys())
                    elif self.method == 'blip' and self.blip_tagger is not None:
                        caption = self.blip_tagger.generate_caption(image)
                        tags = [t.strip() for t in caption.replace(',', ' ').split() if t.strip()]
                        tags = [t for t in tags if t.lower() not in blacklist_set]
                    elif self.method == 'llm' and self.llm_tagger is not None:
                        caption = self.llm_tagger.generate_caption(image)
                        tags = [caption] if caption else []
                        tags = [t for t in tags if t.lower() not in blacklist_set]
                    else:
                        tags = ['furry', 'solo', 'detailed', 'high quality']
            except Exception as ex:
                logger.error(f'[TagProcessor] Failed to load {fpath}: {ex}')
                tags = ['error_loading_image']
            # Image is now released from memory (with statement closed)
            if ordering == 'Shuffle':
                random.shuffle(tags)
            elif ordering == 'Alphabetical':
                tags.sort()
            if trigger_word and trigger_word.strip():
                tw = trigger_word.strip()
                tags = [x for x in tags if x.lower() != tw.lower()]
                tags.insert(0, tw)
            result[filename] = ', '.join(tags)
            if progress_callback:
                progress_callback(i + 1, total)
        return result

    def process_single(self, image, config) -> str:
        result = self.process([image], ['single'], config)
        return result.get('single', '')

    def unload(self):
        if self.wd14_tagger:
            self.wd14_tagger.unload()

def get_tagger_methods() -> list:
    methods = []
    if BLIP_AVAILABLE:
        methods.append(('blip', 'BLIP 描述生成'))
    methods.append(('wd14', 'WD14 标签识别'))
    methods.append(('llm', 'LLM 智能描述 (Qwen3/Gemini)'))
    methods.append(('manual', '手动标签'))
    return methods

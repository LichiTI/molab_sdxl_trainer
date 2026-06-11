import random
from pathlib import Path
import logging

logger = logging.getLogger("ImageScreener")
try:
    import numpy as np
    from PIL import Image
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.info('OpenCV not available. Using simplified scoring.')
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning('PyTorch not installed. ImageScreener will use dummy scores.')
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.info('CLIP not available. Will use fallback scoring.')

class DimensionalScorer:

    def score_composition(self, image) -> float:
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return random.uniform(0.4, 0.8)
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            h, w = gray.shape
            edges = cv2.Canny(gray, 50, 150)
            grid_scores = []
            for i in range(3):
                for j in range(3):
                    y1, y2 = (i * h // 3, (i + 1) * h // 3)
                    x1, x2 = (j * w // 3, (j + 1) * w // 3)
                    region = edges[y1:y2, x1:x2]
                    density = np.sum(region > 0) / region.size
                    grid_scores.append(density)
            center_density = grid_scores[4]
            corner_avg = (grid_scores[0] + grid_scores[2] + grid_scores[6] + grid_scores[8]) / 4
            thirds_score = min(1.0, corner_avg * 3 + 0.3)
            left_sum = sum(grid_scores[0:3])
            right_sum = sum(grid_scores[6:9])
            balance = 1 - abs(left_sum - right_sum) / max(left_sum + right_sum, 0.01)
            score = thirds_score * 0.6 + balance * 0.4
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.error(f'[DimensionalScorer] Composition error: {e}')
            return 0.5

    def score_color(self, image) -> float:
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return random.uniform(0.4, 0.8)
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            if len(img_array.shape) == 2:
                return 0.5
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            sat_mean = np.mean(saturation) / 255
            sat_score = 1 - abs(sat_mean - 0.5) * 2
            sat_score = max(0.0, sat_score)
            value = hsv[:, :, 2]
            val_mean = np.mean(value) / 255
            val_score = 1 - abs(val_mean - 0.5) * 2
            val_score = max(0.0, val_score)
            hue = hsv[:, :, 0]
            hue_hist, _ = np.histogram(hue.flatten(), bins=18, range=(0, 180))
            hue_hist = hue_hist / hue_hist.sum()
            hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-10)) / np.log(18)
            score = sat_score * 0.35 + val_score * 0.35 + hue_entropy * 0.3
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.error(f'[DimensionalScorer] Color error: {e}')
            return 0.5

    def score_detail(self, image) -> float:
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return random.uniform(0.4, 0.8)
        try:
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            sharp_score = min(1.0, sharpness / 2000)
            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            local_contrast = np.mean(np.abs(gray.astype(float) - blur))
            contrast_score = min(1.0, local_contrast / 50)
            score = sharp_score * 0.7 + contrast_score * 0.3
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.error(f'[DimensionalScorer] Detail error: {e}')
            return 0.5

    def score_all(self, image) -> dict:
        return {'composition': self.score_composition(image), 'color': self.score_color(image), 'detail': self.score_detail(image)}

class AestheticMLP(torch.nn.Module):

    def __init__(self, input_size: int=512, hidden_size: int=256, output_size: int=1):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden_size, hidden_size // 2), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden_size // 2, output_size), torch.nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

class WaifuAestheticScorer:
    MODEL_URL = 'https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth'
    MODEL_FILENAME = 'aes-B32-v0.pth'

    def __init__(self, model_dir: Path=None):
        self.clip_model = None
        self.clip_processor = None
        self.aes_model = None
        self.device = 'cpu'
        if model_dir is None:
            model_dir = Path(__file__).parent.parent / 'models' / 'aesthetic'
        self.model_dir = Path(model_dir)
        if TORCH_AVAILABLE and CLIP_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._load_models()

    def _download_model(self) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / self.MODEL_FILENAME
        if model_path.exists():
            return model_path
        logger.info(f'[WaifuAesthetic] Downloading model from {self.MODEL_URL}...')
        try:
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            logger.info(f'[WaifuAesthetic] Model downloaded to {model_path}')
            return model_path
        except Exception as e:
            logger.error(f'[WaifuAesthetic] Download failed: {e}')
            return None

    def _load_models(self):
        try:
            clip_name = 'openai/clip-vit-base-patch32'
            logger.info(f'[WaifuAesthetic] Loading CLIP: {clip_name}')
            self.clip_model = CLIPModel.from_pretrained(clip_name)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_name)
            self.clip_model.to(self.device)
            self.clip_model.eval()
            model_path = self._download_model()
            if model_path and model_path.exists():
                logger.info(f'[WaifuAesthetic] Loading MLP: {model_path}')
                self.aes_model = AestheticMLP(512, 256, 1)
                from core.safe_pickle import safe_torch_load
                state_dict = safe_torch_load(model_path, map_location=self.device)
                self.aes_model.load_state_dict(state_dict)
                self.aes_model.to(self.device)
                self.aes_model.eval()
                logger.info(f'[WaifuAesthetic] Models loaded on {self.device}')
            else:
                logger.warning('[WaifuAesthetic] MLP model not available, using CLIP-only fallback')
        except Exception as e:
            logger.error(f'[WaifuAesthetic] Failed to load models: {e}')
            self.clip_model = None
            self.aes_model = None

    def score_image(self, image) -> float:
        if self.clip_model is None:
            return random.uniform(0.5, 1.0)
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            elif TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                image = Image.fromarray(image.cpu().numpy().astype('uint8'))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if self.aes_model is not None:
                with torch.no_grad():
                    score = self.aes_model(image_features)
                    return score.item()
            else:
                return 0.5
        except Exception as e:
            logger.error(f'[WaifuAesthetic] Error scoring image: {e}')
            return random.uniform(0.4, 0.7)

    def score_batch(self, images, progress_callback=None) -> list:
        scores = []
        total = len(images)
        for i, img in enumerate(images):
            score = self.score_image(img)
            scores.append(score)
            if progress_callback:
                progress_callback(i + 1, total)
        return scores

class CLIPScorer:

    def __init__(self, model_path=None):
        self.model = None
        self.processor = None
        self.device = 'cpu'
        if TORCH_AVAILABLE and CLIP_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self._load_model(model_path)

    def _load_model(self, model_path=None):
        try:
            model_name = model_path or 'openai/clip-vit-base-patch32'
            logger.info(f'[CLIPScorer] Loading model: {model_name}')
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f'[CLIPScorer] Model loaded on {self.device}')
        except Exception as e:
            logger.error(f'[CLIPScorer] Failed to load model: {e}')
            self.model = None

    def score_image(self, image) -> float:
        if self.model is None:
            return random.uniform(0.5, 1.0)
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype('uint8'))
            elif TORCH_AVAILABLE and isinstance(image, torch.Tensor):
                image = Image.fromarray(image.cpu().numpy().astype('uint8'))
            prompts = ['a high quality, beautiful, aesthetic image', 'a low quality, ugly, poorly composed image']
            inputs = self.processor(text=prompts, images=image, return_tensors='pt', padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()
            return score
        except Exception as e:
            logger.error(f'[CLIPScorer] Error scoring image: {e}')
            return random.uniform(0.5, 1.0)

    def score_batch(self, images, progress_callback=None) -> list:
        scores = []
        total = len(images)
        for i, img in enumerate(images):
            score = self.score_image(img)
            scores.append(score)
            if progress_callback:
                progress_callback(i + 1, total)
        return scores

class ImageScreener:

    def __init__(self, use_clip=True):
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.waifu_scorer = None
        self.clip_scorer = None
        self.hybrid_scorer = None  # New SwinV2 + Percentile scorer
        if use_clip and TORCH_AVAILABLE and CLIP_AVAILABLE:
            self.waifu_scorer = WaifuAestheticScorer()
            self.clip_scorer = CLIPScorer()
        self.dimensional_scorer = DimensionalScorer()

    def _init_hybrid_scorer(self):
        """Lazy-load the HybridAestheticScorer for SwinV2_Percentile method."""
        if self.hybrid_scorer is None:
            try:
                from core.scorers.aesthetic_scorer import HybridAestheticScorer
                self.hybrid_scorer = HybridAestheticScorer()
                if not self.hybrid_scorer.load():
                    logger.warning("[ImageScreener] Failed to load HybridAestheticScorer, falling back to CLIP")
                    self.hybrid_scorer = None
            except ImportError as e:
                logger.error(f"[ImageScreener] HybridAestheticScorer not available: {e}")
                self.hybrid_scorer = None
        return self.hybrid_scorer is not None

    def _calculate_weighted_score(self, image, config) -> dict:
        weights = config.get('stars_config', {})
        comp_weight = weights.get('composition', 0.5)
        color_weight = weights.get('color', 0.5)
        detail_weight = weights.get('detail', 0.5)
        mlp_weight = weights.get('mlp', 0.0)
        dim_scores = self.dimensional_scorer.score_all(image)
        method = config.get('method', 'Simple')
        mlp_score = 0.0
        
        # New SwinV2_Percentile method - uses HybridAestheticScorer
        if method == 'SwinV2_Percentile':
            if self._init_hybrid_scorer():
                try:
                    result = self.hybrid_scorer.score(image)
                    return {
                        'composition': dim_scores['composition'],
                        'color': dim_scores['color'],
                        'detail': dim_scores['detail'],
                        'mlp': result.percentile,
                        'percentile': result.percentile,
                        'tier': result.tier_label,
                        'rank_display': result.rank_display,
                        'diagnostics': result.diagnostics.to_dict() if result.diagnostics else None,
                        'final': result.percentile  # Use percentile as final score
                    }
                except Exception as e:
                    logger.error(f"[ImageScreener] SwinV2 scoring failed: {e}, falling back")
            # Fallback to MLP_Model if hybrid scorer fails
            method = 'MLP_Model'
        
        if method == 'MLP_Model' and self.waifu_scorer:
            mlp_score = self.waifu_scorer.score_image(image)
        elif method == 'CLIP_Heuristic' and self.clip_scorer:
            mlp_score = self.clip_scorer.score_image(image)
        total_weight = comp_weight + color_weight + detail_weight
        if method != 'Simple':
            total_weight += mlp_weight
        if total_weight == 0:
            total_weight = 1.0
        weighted_sum = dim_scores['composition'] * comp_weight + dim_scores['color'] * color_weight + dim_scores['detail'] * detail_weight
        if method != 'Simple':
            weighted_sum += mlp_score * mlp_weight
        final_score = weighted_sum / total_weight
        return {'composition': dim_scores['composition'], 'color': dim_scores['color'], 'detail': dim_scores['detail'], 'mlp': mlp_score if method != 'Simple' else None, 'final': final_score}

    def score_and_filter(self, images, filenames, config, progress_callback=None):
        method = config.get('method', 'Simple')
        logger.info(f'[ImageScreener] Scoring {len(images)} images using {method}...')
        results = []
        total = len(images)
        
        # Batch scoring for SwinV2_Percentile (more efficient)
        if method == 'SwinV2_Percentile' and self._init_hybrid_scorer():
            try:
                hybrid_results = self.hybrid_scorer.score_batch(
                    images, 
                    batch_size=8,
                    progress_callback=progress_callback
                )
                for i, (filename, hybrid_res) in enumerate(zip(filenames, hybrid_results)):
                    dim_scores = self.dimensional_scorer.score_all(images[i])
                    results.append({
                        'filename': filename,
                        'score': hybrid_res.percentile,
                        'scores': {
                            'composition': dim_scores['composition'],
                            'color': dim_scores['color'],
                            'detail': dim_scores['detail'],
                            'percentile': hybrid_res.percentile,
                            'tier': hybrid_res.tier_label,
                            'rank_display': hybrid_res.rank_display,
                            'diagnostics': hybrid_res.diagnostics.to_dict() if hybrid_res.diagnostics else None,
                            'final': hybrid_res.percentile
                        },
                        'image': images[i],
                        'index': i
                    })
                results.sort(key=lambda x: x['score'], reverse=True)
                min_score = config.get('min_score', 0)
                if min_score > 0:
                    results = [r for r in results if r['score'] >= min_score]
                top_k = config.get('top_k', 0)
                if top_k > 0:
                    results = results[:top_k]
                logger.info(f'[ImageScreener] Completed. {len(results)} images passed filter.')
                return results
            except Exception as e:
                logger.error(f"[ImageScreener] Batch SwinV2 scoring failed: {e}, falling back to individual scoring")
        
        # Individual scoring (original logic)
        for i, (img, filename) in enumerate(zip(images, filenames)):
            scores = self._calculate_weighted_score(img, config)
            results.append({'filename': filename, 'score': scores['final'], 'scores': scores, 'image': img, 'index': i})
            if progress_callback:
                progress_callback(i + 1, total)
        results.sort(key=lambda x: x['score'], reverse=True)
        min_score = config.get('min_score', 0)
        if min_score > 0:
            results = [r for r in results if r['score'] >= min_score]
        top_k = config.get('top_k', 0)
        if top_k > 0:
            results = results[:top_k]
        logger.info(f'[ImageScreener] Completed. {len(results)} images passed filter.')
        return results

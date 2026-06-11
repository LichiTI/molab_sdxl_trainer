"""
Hybrid Aesthetic Scoring System
===============================

This module implements a dual-engine aesthetic scoring system that combines:
1. SwinV2 Transformer (ONNX) - For precise aesthetic percentile ranking
2. OpenCV DimensionalScorer - For diagnostic analysis of low-score images

The SwinV2 model outputs a percentile score (0.0-1.0) representing where
the image ranks compared to millions of anime images globally.

Architecture inspired by HWtagger's wd14_based_taggers.py, but enhanced
with diagnostic capabilities for explainability.
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Conditional imports with graceful fallbacks
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

try:
    import torch
    import torchvision.transforms.v2 as transforms
    import torchvision.transforms.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    transforms = None
    F = None


# ============================================================================
# Constants & Enums
# ============================================================================

class QualityTier(Enum):
    """Quality tier labels based on percentile thresholds."""
    MASTERPIECE = "masterpiece"  # >= 85%
    BEST = "best"                # >= 75%
    GREAT = "great"              # >= 60%
    GOOD = "good"                # >= 40%
    AVERAGE = "average"          # >= 15%
    WORSE = "worse"              # >= 8%
    WORST = "worst"              # < 8%


QUALITY_THRESHOLDS = {
    QualityTier.MASTERPIECE: 0.85,
    QualityTier.BEST: 0.75,
    QualityTier.GREAT: 0.60,
    QualityTier.GOOD: 0.40,
    QualityTier.AVERAGE: 0.15,
    QualityTier.WORSE: 0.08,
    QualityTier.WORST: 0.0,
}

# Reverse mapping for label lookup
QUALITY_LABELS = ["worst", "worse", "average", "good", "great", "best", "masterpiece"]

IMAGE_SIZE = 448  # SwinV2 input size


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DiagnosticResult:
    """Detailed diagnostic breakdown of image quality dimensions."""
    sharpness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    brightness: float = 0.0
    composition: float = 0.0
    
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "sharpness": round(self.sharpness, 3),
            "contrast": round(self.contrast, 3),
            "saturation": round(self.saturation, 3),
            "brightness": round(self.brightness, 3),
            "composition": round(self.composition, 3),
            "warnings": self.warnings,
        }


@dataclass
class AestheticScore:
    """Complete aesthetic score result with percentile ranking and diagnostics."""
    
    # Primary score (0.0 - 1.0 percentile)
    percentile: float
    
    # Convenience display
    display_score: float  # 0-10 scale
    tier: QualityTier
    tier_label: str
    
    # Raw model output (before percentile mapping)
    raw_score: float
    
    # Confidence of the prediction
    confidence: float
    
    # Diagnostic breakdown (only populated for low scores)
    diagnostics: Optional[DiagnosticResult] = None
    
    def to_dict(self) -> dict:
        return {
            "percentile": round(self.percentile, 4),
            "display_score": round(self.display_score, 2),
            "tier": self.tier.value,
            "tier_label": self.tier_label,
            "raw_score": round(self.raw_score, 4),
            "confidence": round(self.confidence, 3),
            "diagnostics": self.diagnostics.to_dict() if self.diagnostics else None,
        }
    
    @property
    def is_low_quality(self) -> bool:
        return self.percentile < 0.30
    
    @property
    def rank_display(self) -> str:
        """Returns a human-readable rank like 'Top 5%' or 'Bottom 20%'."""
        if self.percentile >= 0.50:
            return f"Top {int((1 - self.percentile) * 100)}%"
        else:
            return f"Bottom {int(self.percentile * 100)}%"


# ============================================================================
# Image Preprocessing
# ============================================================================

class SquarePad:
    """Pads an image to a square with white background."""
    
    def __call__(self, image: "Image.Image") -> "Image.Image":
        if not PIL_AVAILABLE:
            return image
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        if TORCH_AVAILABLE and F is not None:
            return F.pad(image, padding, (255, 255, 255), 'constant')
        else:
            # PIL fallback
            new_img = Image.new('RGB', (max_wh, max_wh), (255, 255, 255))
            new_img.paste(image, (p_left, p_top))
            return new_img


def preprocess_image(image: "Image.Image", target_size: int = IMAGE_SIZE) -> "np.ndarray":
    """
    Preprocess a PIL image for SwinV2 inference.
    
    Args:
        image: PIL Image in any mode
        target_size: Target size for the model (default 448)
    
    Returns:
        numpy array in shape (1, H, W, 3) with float32 dtype
    """
    if not PIL_AVAILABLE or not NUMPY_AVAILABLE:
        raise RuntimeError("PIL and NumPy are required for image preprocessing")
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Square pad and resize
    padder = SquarePad()
    image = padder(image)
    image = image.resize((target_size, target_size), Image.LANCZOS)
    
    # Convert to numpy array
    arr = np.array(image, dtype=np.float32)
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    arr = np.expand_dims(arr, axis=0)
    
    return arr


# ============================================================================
# Dimensional Scorer (OpenCV-based diagnostics)
# ============================================================================

class DimensionalScorer:
    """
    OpenCV-based image quality dimension analyzer.
    
    This provides explainable metrics for WHY an image might score low,
    including sharpness, contrast, saturation, and composition analysis.
    """
    
    def analyze(self, image: "Image.Image") -> DiagnosticResult:
        """Perform full diagnostic analysis on an image."""
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            return DiagnosticResult(warnings=["OpenCV not available for diagnostics"])
        
        result = DiagnosticResult()
        warnings = []
        
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert('RGB'))
            else:
                img_array = image
            
            # Sharpness analysis
            result.sharpness = self._analyze_sharpness(img_array)
            if result.sharpness < 0.3:
                warnings.append("图像模糊 (Blurry image)")
            
            # Contrast analysis
            result.contrast = self._analyze_contrast(img_array)
            if result.contrast < 0.25:
                warnings.append("对比度不足 (Low contrast)")
            
            # Saturation analysis
            result.saturation = self._analyze_saturation(img_array)
            if result.saturation < 0.2:
                warnings.append("色彩饱和度低 (Low saturation)")
            elif result.saturation > 0.9:
                warnings.append("色彩过饱和 (Oversaturated)")
            
            # Brightness analysis
            result.brightness = self._analyze_brightness(img_array)
            if result.brightness < 0.2:
                warnings.append("画面过暗 (Too dark)")
            elif result.brightness > 0.85:
                warnings.append("画面过曝 (Overexposed)")
            
            # Composition analysis (Rule of thirds)
            result.composition = self._analyze_composition(img_array)
            
            result.warnings = warnings
            
        except Exception as e:
            result.warnings = [f"Diagnostic error: {str(e)}"]
        
        return result
    
    def _analyze_sharpness(self, img_array: "np.ndarray") -> float:
        """Measure image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Normalize to 0-1 range (2000 is empirically determined threshold)
        return min(1.0, variance / 2000)
    
    def _analyze_contrast(self, img_array: "np.ndarray") -> float:
        """Measure local contrast using blur difference."""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        local_contrast = np.mean(np.abs(gray.astype(float) - blur))
        # Normalize to 0-1 range
        return min(1.0, local_contrast / 50)
    
    def _analyze_saturation(self, img_array: "np.ndarray") -> float:
        """Analyze color saturation in HSV space."""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        sat_mean = np.mean(saturation) / 255
        return sat_mean
    
    def _analyze_brightness(self, img_array: "np.ndarray") -> float:
        """Analyze overall brightness in HSV space."""
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        value = hsv[:, :, 2]
        return np.mean(value) / 255
    
    def _analyze_composition(self, img_array: "np.ndarray") -> float:
        """
        Analyze composition using edge detection and rule of thirds.
        Returns a score indicating how well the image follows good composition.
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Analyze edge density in rule-of-thirds grid
        grid_scores = []
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region = edges[y1:y2, x1:x2]
                density = np.sum(region > 0) / region.size
                grid_scores.append(density)
        
        # Good composition has interesting content at intersection points
        thirds_positions = [grid_scores[0], grid_scores[2], grid_scores[6], grid_scores[8]]
        thirds_score = min(1.0, np.mean(thirds_positions) * 3 + 0.3)
        
        # Balance check
        left_sum = sum(grid_scores[0:3])
        right_sum = sum(grid_scores[6:9])
        balance = 1 - abs(left_sum - right_sum) / max(left_sum + right_sum, 0.01)
        
        return thirds_score * 0.6 + balance * 0.4


# ============================================================================
# Reference Data Manager
# ============================================================================

class ReferenceDataManager:
    """
    Manages the percentile mapping reference data.
    
    This uses pre-computed score distributions from millions of anime images
    to map raw model outputs to meaningful percentile rankings.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.x_samples: Optional["np.ndarray"] = None
        self.y_samples: Optional["np.ndarray"] = None
        self._loaded = False
    
    def load(self) -> bool:
        """Load reference distribution data from disk or download if needed."""
        if not NUMPY_AVAILABLE:
            return False
        
        x_file = self.data_dir / "samples_x.npy"
        y_file = self.data_dir / "samples_y.npy"
        
        # Try loading from disk first
        if x_file.exists() and y_file.exists():
            try:
                self.x_samples = np.load(x_file)
                self.y_samples = np.load(y_file)
                self._loaded = True
                return True
            except Exception as e:
                print(f"[ReferenceData] Failed to load from disk: {e}")
        
        # Try downloading from HuggingFace
        return self._download_reference_data()
    
    def _download_reference_data(self) -> bool:
        """Download reference distribution data from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
            repo_id = 'deepghs/anime_aesthetic'
            model_name = 'swinv2pv3_v0_448_ls0.2_x'
            
            print("[ReferenceData] Downloading distribution data from HuggingFace...")
            
            stacked = np.load(hf_hub_download(
                repo_id=repo_id,
                repo_type='model',
                filename=f'{model_name}/samples.npz',
            ))['arr_0']
            
            x, y = stacked[0], stacked[1]
            
            # Subsample for efficiency (every 100th sample after first 3000)
            self.x_samples = x[3000::100]
            self.y_samples = y[3000::100]
            
            # Cache to disk
            self.data_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.data_dir / "samples_x.npy", self.x_samples)
            np.save(self.data_dir / "samples_y.npy", self.y_samples)
            
            self._loaded = True
            print(f"[ReferenceData] Downloaded and cached {len(self.x_samples)} samples")
            return True
            
        except ImportError:
            print("[ReferenceData] huggingface_hub not available")
            return False
        except Exception as e:
            print(f"[ReferenceData] Download failed: {e}")
            return False
    
    def score_to_percentile(self, raw_score: float) -> float:
        """
        Map a raw model score to a percentile using linear interpolation.
        
        Args:
            raw_score: Raw score from the SwinV2 model
            
        Returns:
            Percentile value (0.0 - 1.0) representing global ranking
        """
        if not self._loaded or self.x_samples is None:
            # Fallback: use sigmoid-like mapping
            return 1 / (1 + np.exp(-raw_score + 3))
        
        x = self.x_samples
        y = self.y_samples
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Clamp input to valid range
        clamped_score = np.clip(raw_score, x_min, x_max)
        
        # Find insertion point
        idx = np.searchsorted(x, clamped_score)
        
        if idx >= len(x) - 1:
            return y[-1]
        
        # Linear interpolation
        x0, y0 = x[idx], y[idx]
        x1, y1 = x[idx + 1], y[idx + 1]
        
        if np.isclose(x1, x0):
            return y0
        
        interpolated = (clamped_score - x0) / (x1 - x0) * (y1 - y0) + y0
        return float(np.clip(interpolated, y_min, y_max))


# ============================================================================
# Main Hybrid Aesthetic Scorer
# ============================================================================

class HybridAestheticScorer:
    """
    Dual-engine aesthetic scoring system.
    
    Engine 1: SwinV2 Transformer (ONNX) for precise percentile ranking
    Engine 2: OpenCV DimensionalScorer for diagnostic analysis
    
    Example usage:
        scorer = HybridAestheticScorer()
        scorer.load_model("/path/to/model.onnx")
        
        result = scorer.score(image)
        print(f"Score: {result.display_score}/10 ({result.rank_display})")
        if result.diagnostics:
            print(f"Issues: {result.diagnostics.warnings}")
    """
    
    # Default model download location
    DEFAULT_MODEL_URL = "https://huggingface.co/deepghs/anime_aesthetic/resolve/main/swinv2pv3_v0_448_ls0.2_x/model.onnx"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        enable_diagnostics: bool = True,
        diagnostic_threshold: float = 0.30,
    ):
        """
        Initialize the hybrid scorer.
        
        Args:
            model_path: Path to the ONNX model file. If None, will auto-download.
            enable_diagnostics: Whether to run diagnostic analysis on low-score images.
            diagnostic_threshold: Percentile threshold below which to run diagnostics.
        """
        self.model_path = model_path
        self.enable_diagnostics = enable_diagnostics
        self.diagnostic_threshold = diagnostic_threshold
        
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        
        self._reference_data = ReferenceDataManager()
        self._dimensional_scorer = DimensionalScorer() if enable_diagnostics else None
        
        self._loaded = False
    
    @property
    def is_available(self) -> bool:
        """Check if all required dependencies are available."""
        return ONNX_AVAILABLE and NUMPY_AVAILABLE and PIL_AVAILABLE
    
    def load(self, model_path: Optional[Path] = None) -> bool:
        """
        Load the ONNX model and reference data.
        
        Args:
            model_path: Override the model path set in constructor.
            
        Returns:
            True if loading succeeded, False otherwise.
        """
        if not self.is_available:
            print("[HybridScorer] Required dependencies not available")
            return False
        
        # Resolve model path
        path = model_path or self.model_path
        if path is None:
            path = self._download_model()
        
        if path is None or not Path(path).exists():
            print(f"[HybridScorer] Model not found at {path}")
            return False
        
        try:
            print(f"[HybridScorer] Loading ONNX model from {path}...")
            
            # Create ONNX session with GPU fallback
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self._session = ort.InferenceSession(str(path), providers=providers)
            
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            
            print(f"[HybridScorer] Model loaded. Input: {self._input_name}")
            
            # Load reference data for percentile mapping
            self._reference_data.load()
            
            self._loaded = True
            return True
            
        except Exception as e:
            print(f"[HybridScorer] Failed to load model: {e}")
            return False
    
    def _download_model(self) -> Optional[Path]:
        """Download the SwinV2 aesthetic model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
            print("[HybridScorer] Downloading SwinV2 aesthetic model...")
            
            model_path = hf_hub_download(
                repo_id='deepghs/anime_aesthetic',
                repo_type='model',
                filename='swinv2pv3_v0_448_ls0.2_x/model.onnx',
            )
            
            print(f"[HybridScorer] Model downloaded to {model_path}")
            return Path(model_path)
            
        except ImportError:
            print("[HybridScorer] huggingface_hub not installed")
            return None
        except Exception as e:
            print(f"[HybridScorer] Download failed: {e}")
            return None
    
    def score(self, image: "Image.Image") -> AestheticScore:
        """
        Score a single image.
        
        Args:
            image: PIL Image to score
            
        Returns:
            AestheticScore with percentile ranking and optional diagnostics
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Preprocess image
        input_data = preprocess_image(image)
        
        # Run inference
        outputs = self._session.run([self._output_name], {self._input_name: input_data})[0]
        
        # Calculate raw score (weighted sum of class probabilities)
        # The model outputs probabilities for 7 quality classes
        probs = outputs[0]
        weights = np.arange(len(QUALITY_LABELS))[::-1]  # [6, 5, 4, 3, 2, 1, 0]
        raw_score = np.sum(probs * weights)
        
        # Map to percentile
        percentile = self._reference_data.score_to_percentile(raw_score)
        
        # Get quality tier
        tier = self._percentile_to_tier(percentile)
        
        # Calculate confidence (max probability)
        confidence = float(np.max(probs))
        
        # Build result
        result = AestheticScore(
            percentile=percentile,
            display_score=percentile * 10,  # 0-10 scale
            tier=tier,
            tier_label=tier.value,
            raw_score=raw_score,
            confidence=confidence,
        )
        
        # Run diagnostics for low-score images
        if self.enable_diagnostics and percentile < self.diagnostic_threshold:
            result.diagnostics = self._dimensional_scorer.analyze(image)
        
        return result
    
    def score_batch(
        self,
        images: List["Image.Image"],
        batch_size: int = 8,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[AestheticScore]:
        """
        Score multiple images in batches.
        
        Args:
            images: List of PIL Images to score
            batch_size: Number of images per batch
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            List of AestheticScore results
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        results = []
        total = len(images)
        
        for i in range(0, total, batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_data = np.vstack([preprocess_image(img) for img in batch])
            
            # Run inference
            outputs = self._session.run([self._output_name], {self._input_name: batch_data})[0]
            
            # Process each result in the batch
            for j, probs in enumerate(outputs):
                weights = np.arange(len(QUALITY_LABELS))[::-1]
                raw_score = np.sum(probs * weights)
                percentile = self._reference_data.score_to_percentile(raw_score)
                tier = self._percentile_to_tier(percentile)
                confidence = float(np.max(probs))
                
                result = AestheticScore(
                    percentile=percentile,
                    display_score=percentile * 10,
                    tier=tier,
                    tier_label=tier.value,
                    raw_score=raw_score,
                    confidence=confidence,
                )
                
                # Run diagnostics for low scores
                if self.enable_diagnostics and percentile < self.diagnostic_threshold:
                    result.diagnostics = self._dimensional_scorer.analyze(batch[j])
                
                results.append(result)
            
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)
        
        return results
    
    def _percentile_to_tier(self, percentile: float) -> QualityTier:
        """Map percentile to quality tier."""
        for tier, threshold in QUALITY_THRESHOLDS.items():
            if percentile >= threshold:
                return tier
        return QualityTier.WORST


# ============================================================================
# Convenience function
# ============================================================================

_global_scorer: Optional[HybridAestheticScorer] = None


def get_scorer() -> HybridAestheticScorer:
    """Get or create a global scorer instance."""
    global _global_scorer
    if _global_scorer is None:
        _global_scorer = HybridAestheticScorer()
        _global_scorer.load()
    return _global_scorer


def score_image(image: "Image.Image") -> AestheticScore:
    """Convenience function to score a single image using the global scorer."""
    return get_scorer().score(image)


def score_images(
    images: List["Image.Image"],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[AestheticScore]:
    """Convenience function to score multiple images using the global scorer."""
    return get_scorer().score_batch(images, progress_callback=progress_callback)


# ============================================================================
# CLI Entry Point for Testing
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python aesthetic_scorer.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not PIL_AVAILABLE:
        print("PIL is required. Install with: pip install Pillow")
        sys.exit(1)
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    
    print("Initializing scorer...")
    scorer = HybridAestheticScorer()
    
    if not scorer.load():
        print("Failed to load model!")
        sys.exit(1)
    
    print("Scoring image...")
    result = scorer.score(image)
    
    print("\n" + "=" * 50)
    print(f"📊 Aesthetic Score: {result.display_score:.1f}/10")
    print(f"🏆 Global Ranking: {result.rank_display}")
    print(f"🎖️ Quality Tier: {result.tier_label.upper()}")
    print(f"📈 Confidence: {result.confidence:.1%}")
    
    if result.diagnostics:
        print("\n⚠️ Diagnostics:")
        print(f"  Sharpness:   {result.diagnostics.sharpness:.2f}")
        print(f"  Contrast:    {result.diagnostics.contrast:.2f}")
        print(f"  Saturation:  {result.diagnostics.saturation:.2f}")
        print(f"  Brightness:  {result.diagnostics.brightness:.2f}")
        
        if result.diagnostics.warnings:
            print("\n  Warnings:")
            for warn in result.diagnostics.warnings:
                print(f"    • {warn}")
    
    print("=" * 50)

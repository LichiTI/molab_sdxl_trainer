"""
SVD Dataset Purifier Core
核心纯净原则：只负责计算逻辑，不甚至 UI 状态。
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger("DatasetPurifier")

# Lazy dependency loading
torch = None
Image = None
tqdm = list

def _ensure_deps():
    global torch, Image, tqdm
    if torch is None:
        try:
            import torch as _torch
            torch = _torch
            from PIL import Image as _Image
            Image = _Image
            from tqdm import tqdm as _tqdm
            tqdm = _tqdm
        except ImportError:
            pass

class DatasetPurifier:
    """
    SVD/rSVD-based Dataset Consistency Analyzer.

    Uses CLIP/ResNet embeddings plus spectral analysis to identify outliers and
    coherent subsets. Full SVD is exact but grows expensive on large datasets;
    randomized SVD keeps only the leading subspace and is the default for large
    embedding matrices.
    """

    def __init__(self, device: str = "cuda"):
        self._device_requested = device
        self.model = None
        self.preprocess = None

    @property
    def device(self):
        _ensure_deps()
        return self._device_requested if (torch and torch.cuda.is_available()) else "cpu"

    def _load_clip(self):
        """Lazy load CLIP model"""
        if self.model is not None:
            return

        logger.info("[DatasetPurifier] Loading CLIP model for embedding extraction...")
        try:
            import clip # Assuming openai-clip is installed
            model_name = "ViT-L/14"
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()
        except ImportError:
            logger.error("openai-clip not installed. Cannot use DatasetPurifier.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self.model = None

    def extract_embeddings(self, image_paths: List[str], batch_size: int = 4) -> Tuple[np.ndarray, List[str]]:
        """Extract embeddings for a list of images"""
        self._load_clip()
        if self.model is None or self.preprocess is None or Image is None or torch is None:
            return np.array([]), []

        embeddings = []
        valid_paths = []

        # Batch processing
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.preprocess(img)
                    batch_tensors.append(tensor)
                    valid_paths.append(path)
                except Exception as e:
                    logger.error(f"Error loading {path}: {e}")
                    continue

            if not batch_tensors:
                continue

            with torch.no_grad():
                batch_input = torch.stack(batch_tensors).to(self.device)
                batch_emb = self.model.encode_image(batch_input)
                batch_emb = batch_emb / batch_emb.norm(dim=-1, keepdim=True)
                embeddings.append(batch_emb.cpu().numpy())

        if not embeddings:
            return np.array([]), []

        return np.concatenate(embeddings, axis=0), valid_paths

    def analyze_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        variance_threshold: float = 0.9,
        outlier_percentile: int = 95,
        method: str = "auto",
        max_components: int = 64,
        oversamples: int = 8,
        n_iter: int = 2,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """Analyze precomputed embeddings with full SVD or randomized SVD.

        This entrypoint is intentionally independent from CLIP loading so tests
        and advisor probes can benchmark the spectral core cheaply.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
            return {"error": "embeddings must be a 2D array with at least 2 rows"}
        if labels is None:
            labels = [str(i) for i in range(matrix.shape[0])]
        if len(labels) != matrix.shape[0]:
            return {"error": "labels length must match embeddings rows"}

        method = self._choose_svd_method(matrix.shape, method)
        max_components = max(1, min(int(max_components), min(matrix.shape)))
        start = time.perf_counter()

        mean_emb = np.mean(matrix, axis=0, keepdims=True)
        centered = matrix - mean_emb
        total_variance = float(np.sum(centered * centered))
        if total_variance <= 0.0:
            return {"error": "embeddings have zero variance"}

        try:
            if method == "svd":
                U, S, Vh = self._full_svd(centered)
            else:
                U, S, Vh = self._randomized_svd(
                    centered,
                    max_components=max_components,
                    oversamples=oversamples,
                    n_iter=n_iter,
                    random_state=random_state,
                )
        except np.linalg.LinAlgError:
            return {"error": "SVD computation failed"}

        variance = S ** 2
        explained_ratios = variance / total_variance
        cumulative_variance = np.cumsum(explained_ratios)
        reached = np.where(cumulative_variance >= variance_threshold)[0]
        k = int(reached[0] + 1) if len(reached) else len(S)
        k = max(1, min(k, len(S)))

        # Combine distance inside the retained coherent subspace with residual
        # reconstruction error. The residual term catches off-manifold images
        # that do not align with the dominant style/content axes.
        scores = centered @ Vh[:k].T
        denom = np.maximum(S[:k], 1e-8)
        standardized_scores = scores / denom
        principal_distance = np.sum(standardized_scores ** 2, axis=1)
        reconstructed = scores @ Vh[:k]
        residual = centered - reconstructed
        residual_distance = np.sum(residual * residual, axis=1)
        residual_scale = float(np.median(residual_distance)) or 1.0
        distances = principal_distance + (residual_distance / max(residual_scale, 1e-8))
        threshold = float(np.percentile(distances, outlier_percentile))

        results = []
        for i, label in enumerate(labels):
            score = float(distances[i])
            results.append({
                "path": label,
                "score": round(score, 4),
                "is_outlier": score > threshold,
                "filename": Path(label).name,
            })
        results.sort(key=lambda x: x["score"])

        elapsed = time.perf_counter() - start
        return {
            "total_images": len(labels),
            "kept_components": int(k),
            "threshold": threshold,
            "images": results,
            "explained_variance": [float(x) for x in explained_ratios[:10]],
            "analysis": {
                "method": method,
                "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
                "max_components": int(max_components),
                "components_computed": int(len(S)),
                "variance_threshold": float(variance_threshold),
                "variance_reached": float(cumulative_variance[k - 1]),
                "elapsed_sec": round(elapsed, 6),
                "oversamples": int(oversamples) if method == "rsvd" else 0,
                "n_iter": int(n_iter) if method == "rsvd" else 0,
            },
        }

    def _choose_svd_method(self, shape: Tuple[int, int], method: str) -> str:
        method = (method or "auto").strip().lower()
        if method in {"full", "exact"}:
            return "svd"
        if method in {"randomized", "randomized_svd"}:
            return "rsvd"
        if method in {"svd", "rsvd"}:
            return method
        n_samples, n_features = shape
        # CLIP-like matrices with many samples benefit from rSVD. Small sets stay
        # exact because full SVD gives better variance accounting and is cheap.
        return "rsvd" if n_samples >= 512 or min(n_samples, n_features) >= 384 else "svd"

    def _full_svd(self, centered: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        U, S, Vh = np.linalg.svd(centered, full_matrices=False)
        return U, S, Vh

    def _randomized_svd(
        self,
        centered: np.ndarray,
        max_components: int,
        oversamples: int = 8,
        n_iter: int = 2,
        random_state: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_features = centered.shape
        rank = max(1, min(max_components, n_samples, n_features))
        sketch_size = max(1, min(rank + max(0, oversamples), n_samples, n_features))
        rng = np.random.default_rng(random_state)
        omega = rng.standard_normal((n_features, sketch_size), dtype=np.float32)
        Y = centered @ omega
        for _ in range(max(0, int(n_iter))):
            Y = centered @ (centered.T @ Y)
        Q, _ = np.linalg.qr(Y, mode="reduced")
        B = Q.T @ centered
        Ub, S, Vh = np.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
        return U[:, :rank], S[:rank], Vh[:rank]

    def analyze_consistency(
        self,
        image_dir: str,
        variance_threshold: float = 0.9,
        outlier_percentile: int = 95,
        method: str = "auto",
        max_components: int = 64,
        oversamples: int = 8,
        n_iter: int = 2,
        random_state: int = 0,
    ) -> Dict[str, Any]:
        """
        Analyze dataset consistency using SVD/rSVD.

        Algorithm:
        1. Get CLIP embeddings matrix (N, D)
        2. Centering: subtract mean
        3. Full SVD for small sets, randomized SVD for larger sets by default
        4. Score samples in the retained principal subspace
        5. Highest-distance samples are reported as outliers
        """

        types = ('*.jpg', '*.jpeg', '*.png', '*.webp')
        files = []
        path_obj = Path(image_dir)
        for t in types:
            files.extend([str(p) for p in path_obj.glob(t)])

        if len(files) < 10:
            return {"error": "Too few images (<10) for spectral analysis"}

        logger.info(f"[DatasetPurifier] Analyzing {len(files)} images...")
        embeddings, valid_files = self.extract_embeddings(files)

        if len(embeddings) == 0:
            return {"error": "Failed to extract embeddings"}

        result = self.analyze_embeddings(
            embeddings,
            labels=valid_files,
            variance_threshold=variance_threshold,
            outlier_percentile=outlier_percentile,
            method=method,
            max_components=max_components,
            oversamples=oversamples,
            n_iter=n_iter,
            random_state=random_state,
        )
        if "analysis" in result:
            logger.info(
                "[DatasetPurifier] %s kept %s components in %.3fs",
                result["analysis"]["method"],
                result["kept_components"],
                result["analysis"]["elapsed_sec"],
            )
        return result

# Global Instance
dataset_purifier = DatasetPurifier()
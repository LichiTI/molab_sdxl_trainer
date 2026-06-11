"""Smoke and micro-benchmark for DatasetPurifier randomized SVD."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np


def _load_purifier_module():
    module_path = Path(__file__).resolve().parents[1] / "dataset_purifier.py"
    spec = importlib.util.spec_from_file_location("dataset_purifier_smoke_target", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_embeddings(n: int = 768, d: int = 512, outliers: int = 24) -> tuple[np.ndarray, list[str], set[str]]:
    rng = np.random.default_rng(1234)
    latent = rng.normal(size=(n, 24)).astype(np.float32)
    basis = rng.normal(size=(24, d)).astype(np.float32)
    embeddings = latent @ basis + 0.08 * rng.normal(size=(n, d)).astype(np.float32)
    outlier_idx = np.arange(n - outliers, n)
    embeddings[outlier_idx] += rng.normal(loc=0.0, scale=14.0, size=(outliers, d)).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    labels = [f"sample_{i:04d}.png" for i in range(n)]
    return embeddings, labels, {labels[i] for i in outlier_idx}


def _outliers(report: dict) -> set[str]:
    return {item["path"] for item in report["images"] if item["is_outlier"]}


def main() -> int:
    module = _load_purifier_module()
    purifier = module.DatasetPurifier(device="cpu")
    embeddings, labels, injected = _make_embeddings()

    t0 = time.perf_counter()
    exact = purifier.analyze_embeddings(
        embeddings,
        labels=labels,
        method="svd",
        variance_threshold=0.9,
        outlier_percentile=97,
        max_components=96,
    )
    exact_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    approx = purifier.analyze_embeddings(
        embeddings,
        labels=labels,
        method="rsvd",
        variance_threshold=0.9,
        outlier_percentile=97,
        max_components=96,
        oversamples=12,
        n_iter=2,
        random_state=42,
    )
    approx_wall = time.perf_counter() - t1

    auto = purifier.analyze_embeddings(embeddings, labels=labels, method="auto", max_components=96)

    assert "error" not in exact, exact
    assert "error" not in approx, approx
    assert exact["analysis"]["method"] == "svd"
    assert approx["analysis"]["method"] == "rsvd"
    assert auto["analysis"]["method"] == "rsvd"
    assert approx["analysis"]["components_computed"] <= 96

    exact_outliers = _outliers(exact)
    approx_outliers = _outliers(approx)
    exact_hits = len(exact_outliers & injected)
    approx_hits = len(approx_outliers & injected)
    overlap = len(exact_outliers & approx_outliers) / max(len(exact_outliers | approx_outliers), 1)

    assert exact_hits >= 12, (exact_hits, len(exact_outliers))
    assert approx_hits >= 12, (approx_hits, len(approx_outliers))
    assert overlap >= 0.45, overlap

    print(
        "dataset_purifier_rsvd_smoke: ok "
        f"exact={exact_wall:.4f}s rsvd={approx_wall:.4f}s "
        f"overlap={overlap:.2f} exact_hits={exact_hits} rsvd_hits={approx_hits}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
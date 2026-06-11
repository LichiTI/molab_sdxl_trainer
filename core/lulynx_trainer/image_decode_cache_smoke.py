"""Smoke checks for CaptionDataset image decode cache policy."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from PIL import Image

try:
    from .dataset_loader import CaptionDataset
except ImportError:  # pragma: no cover - direct script usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.core.lulynx_trainer.dataset_loader import CaptionDataset


def _write_image(path: Path, color, size=(32, 24)) -> None:
    image = Image.new("RGBA", size, color)
    image.save(path)
    path.with_suffix(".txt").write_text("sample caption", encoding="utf-8")


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        img_a = root / "a.png"
        img_b = root / "b.png"
        _write_image(img_a, (255, 0, 0, 128))
        _write_image(img_b, (0, 255, 0, 255))

        dataset = CaptionDataset(
            data_dir=str(root),
            resolution=32,
            enable_bucket=False,
            alpha_mask=True,
            image_decode_backend="pil_lru",
            image_decode_cache_size=1,
        )

        first, alpha = dataset._load_image_rgb_alpha(str(img_a), need_alpha=True)
        assert first.getpixel((0, 0)) == (255, 0, 0)
        assert alpha is not None and alpha.getpixel((0, 0)) == 128
        assert dataset._image_decode_cache_misses == 1

        first.paste((0, 0, 0), (0, 0, first.width, first.height))
        second, _ = dataset._load_image_rgb_alpha(str(img_a), need_alpha=True)
        assert second.getpixel((0, 0)) == (255, 0, 0), "cached image must be returned as a copy"
        assert dataset._image_decode_cache_hits == 1

        dataset._load_image_rgb_alpha(str(img_b), need_alpha=True)
        assert len(dataset._image_decode_cache) == 1
        dataset._load_image_rgb_alpha(str(img_a), need_alpha=True)
        assert dataset._image_decode_cache_misses == 3, "LRU eviction should force a new decode"

        _write_image(img_a, (0, 0, 255, 64), size=(33, 24))
        updated, updated_alpha = dataset._load_image_rgb_alpha(str(img_a), need_alpha=True)
        assert updated.getpixel((0, 0)) == (0, 0, 255)
        assert updated_alpha is not None and updated_alpha.getpixel((0, 0)) == 64

        auto_no_cache = CaptionDataset(
            data_dir=str(root),
            resolution=32,
            enable_bucket=False,
            image_decode_backend="auto",
            image_decode_cache_size=0,
        )
        assert auto_no_cache.image_decode_backend == "pil"

        auto_cached = CaptionDataset(
            data_dir=str(root),
            resolution=32,
            enable_bucket=False,
            image_decode_backend="auto",
            image_decode_cache_size=2,
        )
        assert auto_cached.image_decode_backend == "pil_lru"

        torchvision_dataset = CaptionDataset(
            data_dir=str(root),
            resolution=32,
            enable_bucket=False,
            alpha_mask=True,
            image_decode_backend="torchvision_cpu",
            image_decode_cache_size=0,
        )
        tv_image, tv_alpha = torchvision_dataset._load_image_rgb_alpha(str(img_a), need_alpha=True)
        assert torchvision_dataset.image_decode_backend == "torchvision_cpu"
        assert tv_image.getpixel((0, 0)) == (0, 0, 255)
        assert tv_alpha is not None and tv_alpha.getpixel((0, 0)) == 64

    print("image_decode_cache_smoke: ok")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke for Colorization dataset preprocess (CPU only).

Verifies:
1. lineart mode writes one control image per source image at the contract path.
2. grayscale mode works too.
3. the produced control-image paths exactly match the contract audit (so the
   only missing_required entries are the trainer-side cond/text caches, never
   the control images this tool produced).
4. the launcher toolbox_runner dispatch branch works end-to-end.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from PIL import Image  # noqa: E402

from core.tools.colorize_preprocess import prepare_colorization_dataset  # noqa: E402


def _make_color_images(folder: Path, count: int) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    stems = []
    for i in range(count):
        img = Image.new("RGB", (32, 32), (10 * i % 255, 40, 200 - 10 * i))
        for x in range(0, 32, 4):
            for y in range(32):
                img.putpixel((x, y), (250, 250, 250))  # edges for lineart
        path = folder / f"img_{i}.png"
        img.save(path)
        stems.append(path.stem)
    return stems


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    tmp_path = Path(tempfile.mkdtemp())
    try:
        img_dir = tmp_path / "color"
        ctrl_dir = tmp_path / "control"
        cond_dir = tmp_path / "cond"
        text_dir = tmp_path / "text"
        stems = _make_color_images(img_dir, 3)

        # 1. lineart mode ---------------------------------------------------
        res = prepare_colorization_dataset(
            str(img_dir), str(ctrl_dir),
            mode="lineart", control_suffix="_cond",
            cond_cache_dir=str(cond_dir), text_cache_dir=str(text_dir),
            target_family="anima",
        )
        produced = sorted(p.name for p in ctrl_dir.iterdir() if p.is_file())
        checks.append(("lineart_count", res["processed"] == 3 and len(produced) == 3, f"produced={produced}"))
        checks.append(("lineart_naming", all(f"{s}_cond.png" in produced for s in stems), str(produced)))
        checks.append(("spec_colorize", res["task_spec"].get("task_id") == "colorize", str(res["task_spec"].get("task_id"))))

        # 3. audit only misses trainer-side cond/text, never our control imgs
        missing = res["audit"]["missing_required"]
        control_missing = [m for m in missing if "control image" in m]
        cond_text_missing = [m for m in missing if "cond latent" in m or "text cache" in m]
        checks.append(("audit_no_control_missing", control_missing == [], str(control_missing)))
        checks.append(("audit_reports_cond_text", len(cond_text_missing) >= 3, f"n={len(cond_text_missing)}"))

        # 2. grayscale mode -------------------------------------------------
        gray_dir = tmp_path / "control_gray"
        res_g = prepare_colorization_dataset(
            str(img_dir), str(gray_dir), mode="grayscale", control_suffix="_cond",
        )
        gray_imgs = [p for p in gray_dir.iterdir() if p.is_file()]
        # a flat-edge grayscale image is not all-black like a sparse edge map
        sample = Image.open(gray_imgs[0]).convert("L")
        extrema = sample.getextrema()
        checks.append(("grayscale_count", res_g["processed"] == 3, f"processed={res_g['processed']}"))
        checks.append(("grayscale_nontrivial", extrema[1] > extrema[0], f"extrema={extrema}"))

        # 4. toolbox_runner dispatch ---------------------------------------
        from lulynx_launcher.services.toolbox_runner import run_action

        runner_dir = tmp_path / "control_runner"
        runner_res = run_action(
            "colorize_preprocess",
            {"image_dir": str(img_dir), "control_image_dir": str(runner_dir), "mode": "lineart"},
        )
        checks.append(("runner_dispatch", runner_dir.is_dir() and runner_res.get("processed") == 3, str(runner_res.get("processed"))))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    ok = all(passed for _, passed, _ in checks)
    print("=== colorize_preprocess smoke ===")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"scorecard ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

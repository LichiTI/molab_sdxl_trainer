# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""CPU smoke: EasyControl v2 colorize cond-latent cache producer.

Proves the keystone ``build_colorize_cond_cache`` closes the colorize gap with a
toy VAE-encode callable + synthetic control images (no real model / no GPU):

* **Exact contract paths.** For each target image, a cond sidecar is written at
  exactly ``sidecar_plan_for_target(target, spec).cond_latent_path`` (i.e.
  ``cond_cache_dir/{stem}.latent.safetensors``).
* **Loader-compatible.** Each sidecar loads via the dataset loader's
  ``_load_sidecar_tensor`` (key ``cond_latents``) and round-trips the encoded
  latent shape/values.
* **Honest missing-control skip.** A target whose control image is absent is
  counted in ``missing_control`` and produces NO sidecar (never fabricated).
* **Idempotent + force.** A second run without ``force`` skips existing sidecars;
  with ``force=True`` it overwrites (content updates when control image changes).
* **Non-colorize / unconfigured no-op.** task_id != colorize, or empty
  cond_cache_dir, produces nothing.

Run directly:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/colorize_cond_cache_smoke.py
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import numpy as np
import torch
from PIL import Image

from core.lulynx_trainer.colorize_cond_cache import build_colorize_cond_cache
from core.lulynx_trainer.dataset_loader import _load_sidecar_tensor
from core.lulynx_trainer.easycontrol_v2_contract import (
    EasyControlV2TaskSpec,
    sidecar_plan_for_target,
)


def _toy_vae_encode(tensor: torch.Tensor) -> torch.Tensor:
    """Mimic the anima VAE contract: ``[1,3,H,W]`` -> ``[1,16,H//8,W//8]``.

    Deterministic and content-dependent (mean over a 3->16 channel lift then an
    8x avg-pool) so different control images yield different latents.
    """
    assert tensor.dim() == 4 and tensor.shape[1] == 3, tensor.shape
    # 3 -> 16 channel lift by tiling + content-dependent scaling.
    lifted = tensor.repeat(1, 6, 1, 1)[:, :16, :, :]
    pooled = torch.nn.functional.avg_pool2d(lifted, kernel_size=8, stride=8)
    return pooled  # [1, 16, H//8, W//8]


def _write_image(path: Path, color: tuple[int, int, int], size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = color
    Image.fromarray(arr).save(path)


def _check(label: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}", flush=True)
    if not cond:
        raise AssertionError(label)


def main() -> int:
    tmp = tempfile.mkdtemp()
    try:
        return _run(Path(tmp))
    finally:
        # safetensors load_file mmaps the file on Windows; release before cleanup.
        gc.collect()
        shutil.rmtree(tmp, ignore_errors=True)


def _run(root: Path) -> int:
    if True:
        data_dir = root / "dataset" / "6_lulu"
        control_dir = root / "post" / "colorize_control"
        cond_dir = root / "post" / "colorize_cond"
        text_dir = root / "post" / "colorize_text"

        # Three targets; only two have control images (third tests missing-control).
        stems = ["a", "b", "c"]
        for i, stem in enumerate(stems):
            _write_image(data_dir / f"{stem}.png", (40 * (i + 1), 80, 120))
        _write_image(control_dir / "a_lineart.png", (10, 10, 10))
        _write_image(control_dir / "b_lineart.png", (250, 250, 250))
        # c_lineart.png intentionally absent.

        spec = EasyControlV2TaskSpec(
            task_id="colorize",
            cond_cache_dir=str(cond_dir),
            text_cache_dir=str(text_dir),
            control_image_dir=str(control_dir),
            control_suffix="_lineart.png",
        )

        targets = [str(data_dir / f"{stem}.png") for stem in stems]

        report = build_colorize_cond_cache(
            target_image_paths=targets,
            spec=spec,
            vae_encode_fn=_toy_vae_encode,
            disk_dtype=torch.float16,
        )

        _check("written == 2 (a, b)", report.written == 2)
        _check("missing_control == 1 (c)", report.missing_control == 1)
        _check("no errors", not report.errors)

        # Exact contract paths + loader round-trip.
        plan_a = sidecar_plan_for_target(targets[0], spec)
        plan_c = sidecar_plan_for_target(targets[2], spec)
        cond_a = Path(plan_a.cond_latent_path)
        _check(
            "sidecar at exact contract path cond_cache_dir/a.latent.safetensors",
            cond_a == cond_dir / "a.latent.safetensors" and cond_a.is_file(),
        )
        _check("no sidecar for missing-control target c", not Path(plan_c.cond_latent_path).is_file())

        loaded = _load_sidecar_tensor(str(cond_a))
        _check("loader returns a tensor for a", torch.is_tensor(loaded))
        # Materialize into a plain tensor so the underlying mmap can be released
        # (Windows blocks overwriting an mmapped file during the force run below).
        loaded_vals = loaded.float().clone()
        # Expected latent: toy encode of the 64x64 control image -> [16, 8, 8].
        expected = _toy_vae_encode(
            torch.from_numpy(
                np.asarray(Image.open(plan_a.control_image_path).convert("RGB")).astype("float32") / 127.5 - 1.0
            ).permute(2, 0, 1).unsqueeze(0)
        )[0]
        _check("loaded latent shape == [16,8,8]", tuple(loaded.shape) == (16, 8, 8))
        _check(
            "loaded latent matches toy-encoded control (fp16 tol)",
            torch.allclose(loaded_vals, expected.float(), atol=1e-2),
        )

        # a and b differ because control images differ.
        loaded_b = _load_sidecar_tensor(str(Path(sidecar_plan_for_target(targets[1], spec).cond_latent_path)))
        loaded_b_vals = loaded_b.float().clone()
        _check("a != b (content-dependent encode)", not torch.allclose(loaded_vals, loaded_b_vals, atol=1e-3))

        # Idempotent: second run without force skips existing.
        report2 = build_colorize_cond_cache(
            target_image_paths=targets, spec=spec, vae_encode_fn=_toy_vae_encode, disk_dtype=torch.float16,
        )
        _check("second run skips existing (skipped == 2)", report2.skipped == 2 and report2.written == 0)

        # Release mmaps before forcing an overwrite (Windows file-lock).
        del loaded, loaded_b
        gc.collect()

        # Force overwrite: change control image, force=True -> content updates.
        _write_image(control_dir / "a_lineart.png", (200, 30, 30))
        report3 = build_colorize_cond_cache(
            target_image_paths=targets, spec=spec, vae_encode_fn=_toy_vae_encode, disk_dtype=torch.float16, force=True,
        )
        _check("force run rewrites (written == 2)", report3.written == 2)
        reloaded = _load_sidecar_tensor(str(cond_a))
        reloaded_vals = reloaded.float().clone()
        del reloaded
        gc.collect()
        _check("forced sidecar content changed", not torch.allclose(reloaded_vals, loaded_vals, atol=1e-3))

        # Non-colorize no-op.
        gen_spec = EasyControlV2TaskSpec(task_id="generic", cond_cache_dir=str(cond_dir), control_image_dir=str(control_dir))
        gen_report = build_colorize_cond_cache(
            target_image_paths=targets, spec=gen_spec, vae_encode_fn=_toy_vae_encode,
        )
        _check("non-colorize no-op (nothing written)", gen_report.written == 0 and gen_report.skipped == 0)

        # Empty cond_cache_dir -> honest error, no crash.
        bad_spec = EasyControlV2TaskSpec(task_id="colorize", cond_cache_dir="", control_image_dir=str(control_dir))
        bad_report = build_colorize_cond_cache(
            target_image_paths=targets, spec=bad_spec, vae_encode_fn=_toy_vae_encode,
        )
        _check("empty cond_cache_dir reports error, no write", bad_report.errors and bad_report.written == 0)

    print("\nAll colorize_cond_cache smoke checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

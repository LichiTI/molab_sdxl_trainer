# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke for Thin-SVD adapter export (CPU only).

Verifies:
1. exact mode round-trip parity (reconstructed delta == original delta).
2. approx mode truncates rank and retains < 1.0 spectral energy.
3. multi-adapter merge preserves the weighted-sum delta in exact mode.
4. metadata header is written.
5. the launcher toolbox_runner dispatch branch works end-to-end.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from core.tools.thin_svd_export import thin_svd_export  # noqa: E402


def _make_lora(path: Path, *, rank: int, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    layers = {}
    tensors = {}
    for name, (out_f, in_f) in {"layer_a": (16, 24), "layer_b": (32, 16)}.items():
        down = torch.randn(rank, in_f, generator=g) * 0.05
        up = torch.randn(out_f, rank, generator=g) * 0.05
        tensors[f"{name}.lora_down.weight"] = down
        tensors[f"{name}.lora_up.weight"] = up
        tensors[f"{name}.alpha"] = torch.tensor(float(rank))
        layers[name] = up.float() @ down.float()
    save_file(tensors, str(path))
    return layers


def _reconstruct(tensors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return tensors[f"{name}.lora_up.weight"].float() @ tensors[f"{name}.lora_down.weight"].float()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    tmp_path = Path(tempfile.mkdtemp())
    try:
        src_a = tmp_path / "a.safetensors"
        src_b = tmp_path / "b.safetensors"
        deltas_a = _make_lora(src_a, rank=8, seed=1)
        deltas_b = _make_lora(src_b, rank=8, seed=2)

        # 1. exact mode parity ---------------------------------------------
        out_exact = tmp_path / "exact.safetensors"
        res = thin_svd_export(str(src_a), str(out_exact), target_rank=4, mode="exact")
        exact_t = load_file(str(out_exact))
        max_err = max(float((_reconstruct(exact_t, n) - deltas_a[n]).abs().max()) for n in deltas_a)
        checks.append(("exact_parity", max_err < 1e-3, f"max_err={max_err:.2e}"))
        checks.append(("exact_energy", res["energy_retained"] > 0.999, f"energy={res['energy_retained']}"))
        checks.append(("exact_rank", res["target_rank"] == 8, f"rank={res['target_rank']}"))

        # 2. approx mode truncation ----------------------------------------
        out_approx = tmp_path / "approx.safetensors"
        res2 = thin_svd_export(str(src_a), str(out_approx), target_rank=3, mode="approx")
        approx_t = load_file(str(out_approx))
        got_rank = approx_t["layer_a.lora_down.weight"].shape[0]
        checks.append(("approx_rank", got_rank == 3, f"rank={got_rank}"))
        checks.append(("approx_energy_lt1", res2["energy_retained"] < 1.0, f"energy={res2['energy_retained']}"))
        checks.append(("approx_energy_pos", res2["energy_retained"] > 0.0, f"energy={res2['energy_retained']}"))

        # 3. multi-adapter merge (exact preserves sum) ---------------------
        out_merge = tmp_path / "merge.safetensors"
        thin_svd_export([str(src_a), str(src_b)], str(out_merge), mode="exact", merge_weights=[1.0, 1.0])
        merge_t = load_file(str(out_merge))
        merge_err = max(
            float((_reconstruct(merge_t, n) - (deltas_a[n] + deltas_b[n])).abs().max())
            for n in deltas_a
        )
        checks.append(("merge_parity", merge_err < 2e-3, f"merge_err={merge_err:.2e}"))

        # 4. metadata header -----------------------------------------------
        with open(out_approx, "rb") as fh:
            import json
            import struct

            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n).decode("utf-8"))
        meta = header.get("__metadata__", {})
        checks.append(("metadata", meta.get("thin_svd_mode") == "approx" and "energy_retained" in meta, str(meta)))

        # 5. toolbox_runner dispatch ---------------------------------------
        from lulynx_launcher.services.toolbox_runner import run_action

        out_runner = tmp_path / "runner.safetensors"
        runner_res = run_action(
            "thin_svd_export",
            {"input_path": str(src_a), "output_path": str(out_runner), "target_rank": 4, "mode": "approx", "device": "cpu"},
        )
        checks.append(("runner_dispatch", out_runner.is_file() and runner_res.get("layers_exported") == 2, str(runner_res.get("layers_exported"))))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    ok = all(passed for _, passed, _ in checks)
    print("=== thin_svd_export smoke ===")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    print(f"scorecard ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

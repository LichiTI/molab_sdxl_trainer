"""Smoke checks for the LXCS replacement loader policy."""

from __future__ import annotations

from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from lulynx_trainer.lossless_cache_replacement_policy import (  # type: ignore[no-redef]
        LosslessCacheReplacementPolicyConfig,
        evaluate_lossless_cache_replacement_policy,
    )
else:
    from .lossless_cache_replacement_policy import (
        LosslessCacheReplacementPolicyConfig,
        evaluate_lossless_cache_replacement_policy,
    )


def _prepare(raw_bytes: int, sidecar_bytes: int) -> dict[str, object]:
    return {
        "ok": True,
        "reports": [
            {
                "ok": True,
                "raw_bytes": int(raw_bytes),
                "sidecar_bytes": int(sidecar_bytes),
                "compression_ratio": round(sidecar_bytes / max(float(raw_bytes), 1.0), 6),
            }
        ],
    }


def test_policy_bypasses_tiny_or_light_compute_cases() -> None:
    tiny = evaluate_lossless_cache_replacement_policy(
        _prepare(12_960, 14_957),
        workload={"compute_repeat": 2},
    )
    assert tiny["enabled"] is False, tiny
    assert tiny["selected_path"] == "raw_dataloader", tiny
    assert tiny["reason"] in {"no_real_byte_savings", "raw_bytes_below_floor"}, tiny

    medium_light = evaluate_lossless_cache_replacement_policy(
        _prepare(216_096, 16_990),
        workload={"compute_repeat": 2},
    )
    assert medium_light["enabled"] is False, medium_light
    assert medium_light["reason"] == "light_compute_raw_bytes_below_floor", medium_light


def test_policy_allows_larger_or_heavier_cases() -> None:
    medium_heavy = evaluate_lossless_cache_replacement_policy(
        _prepare(216_096, 16_990),
        workload={"compute_repeat": 16},
    )
    assert medium_heavy["enabled"] is True, medium_heavy
    assert medium_heavy["selected_path"] == "lxcs_replacement", medium_heavy

    large_light = evaluate_lossless_cache_replacement_policy(
        _prepare(858_144, 22_019),
        workload={"compute_repeat": 2},
    )
    assert large_light["enabled"] is True, large_light


def test_policy_is_stricter_for_cuda_cases() -> None:
    medium_heavy_cuda = evaluate_lossless_cache_replacement_policy(
        _prepare(216_096, 16_990),
        workload={"compute_repeat": 16, "device": "cuda"},
    )
    assert medium_heavy_cuda["enabled"] is False, medium_heavy_cuda
    assert medium_heavy_cuda["reason"] == "cuda_raw_bytes_below_floor", medium_heavy_cuda

    large_light_cuda = evaluate_lossless_cache_replacement_policy(
        _prepare(858_144, 22_019),
        workload={"compute_repeat": 2, "device": "cuda"},
    )
    assert large_light_cuda["enabled"] is False, large_light_cuda
    assert large_light_cuda["reason"] == "cuda_raw_bytes_below_floor", large_light_cuda


def test_policy_modes_are_explicit() -> None:
    off = evaluate_lossless_cache_replacement_policy(
        _prepare(858_144, 22_019),
        workload={"compute_repeat": 2},
        config=LosslessCacheReplacementPolicyConfig(mode="off"),
    )
    assert off["enabled"] is False, off
    assert off["reason"] == "policy_mode_off", off

    always = evaluate_lossless_cache_replacement_policy(
        {"ok": False, "reports": []},
        workload={"compute_repeat": 1},
        config=LosslessCacheReplacementPolicyConfig(mode="always"),
    )
    assert always["enabled"] is True, always
    assert always["reason"] == "policy_mode_always", always


def main() -> int:
    test_policy_bypasses_tiny_or_light_compute_cases()
    test_policy_allows_larger_or_heavier_cases()
    test_policy_is_stricter_for_cuda_cases()
    test_policy_modes_are_explicit()
    print("lossless_cache_replacement_policy_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

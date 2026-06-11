"""Smoke checks for checkpoint policy resolution."""

from pathlib import Path
import sys
from types import SimpleNamespace

try:
    from .checkpoint_policy import normalize_checkpoint_policy, resolve_checkpoint_policy
except ImportError:  # pragma: no cover - direct script usage
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from backend.core.lulynx_trainer.checkpoint_policy import normalize_checkpoint_policy, resolve_checkpoint_policy


def main() -> None:
    assert normalize_checkpoint_policy("cpu-offload") == "offloaded"
    assert normalize_checkpoint_policy("sac") == "selective"
    assert normalize_checkpoint_policy("???") == "auto"

    auto_full = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="auto", gradient_checkpointing=True, cpu_offload_checkpointing=False),
        route="sdxl",
        cuda_available=True,
    )
    assert auto_full.effective_policy == "full"
    assert auto_full.gradient_checkpointing is True
    assert auto_full.cpu_offload_checkpointing is False

    explicit_off = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="off", gradient_checkpointing=True, cpu_offload_checkpointing=True),
        route="sdxl",
        cuda_available=True,
    )
    assert explicit_off.effective_policy == "off"
    assert explicit_off.gradient_checkpointing is False
    assert explicit_off.cpu_offload_checkpointing is False

    offloaded_cuda = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="offloaded", gradient_checkpointing=False, cpu_offload_checkpointing=False),
        route="anima",
        cuda_available=True,
    )
    assert offloaded_cuda.effective_policy == "offloaded"
    assert offloaded_cuda.cpu_offload_checkpointing is True

    offloaded_cpu = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="offloaded", gradient_checkpointing=False, cpu_offload_checkpointing=False),
        route="anima",
        cuda_available=False,
    )
    assert offloaded_cpu.effective_policy == "full"
    assert offloaded_cpu.fallback_reason

    selective_anima = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="selective", gradient_checkpointing=False, cpu_offload_checkpointing=False),
        route="anima",
        cuda_available=True,
    )
    assert selective_anima.effective_policy == "selective"
    assert selective_anima.gradient_checkpointing is False
    assert selective_anima.selective_profile.get("route") == "anima"
    assert selective_anima.selective_profile.get("forward_wired") is True
    assert selective_anima.selective_profile.get("wiring_state") == "experimental_live"
    assert not selective_anima.selective_profile.get("fallback_reason")

    selective = resolve_checkpoint_policy(
        SimpleNamespace(checkpoint_policy="selective", gradient_checkpointing=False, cpu_offload_checkpointing=False),
        route="newbie",
        cuda_available=True,
    )
    assert selective.effective_policy == "selective"
    assert selective.gradient_checkpointing is False
    assert not selective.fallback_reason
    assert selective.selective_profile.get("route") == "newbie"
    assert selective.selective_profile.get("api", {}).get("available") == selective.selective_available
    assert selective.selective_profile.get("forward_wired") is True
    assert selective.selective_profile.get("wiring_state") == "experimental_live"

    print("checkpoint_policy_smoke: ok")


if __name__ == "__main__":
    main()


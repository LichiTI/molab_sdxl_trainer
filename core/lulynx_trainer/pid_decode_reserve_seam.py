# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Default-off opt-in decode-backend seam for the P5 PiD native decoder reserve.

The P5 primitives (``pid_decoder_backend`` contract, loader preflight, runtime
boundary) were proven in isolation.  This module is the *real integration* step
the roadmap tracked as a follow-up gate: a decode seam that actually routes a
caller's latent->image decode to either the VAE (default) or a PiD tiny-decode
path (opt-in, user-weight-gated), without ever bundling or auto-downloading PiD
weights.

Two gates keep this default-off and honest:

* ``decode_backend`` defaults to VAE, and the master ``allow_pid`` flag defaults
  to ``False`` -- so a caller that does nothing always gets the VAE backend,
  bitwise-identical to calling ``vae_decode_fn`` directly (parity).
* Even with ``allow_pid=True`` and ``decode_backend="pid"``, PiD decode only runs
  when the user has supplied a local checkpoint (capability report
  ``checkpoint_available``).  When the weights are absent the seam **falls back to
  the VAE backend** and reports ``pid_blocked`` with the reason -- it never errors,
  never downloads, never bundles.

Honesty red-lines: ``vae_replacement_allowed`` / ``runtime_activation_enabled`` /
``request_fields_emitted`` / ``auto_download_allowed`` / ``bundle_weights_allowed``
/ ``promotion_ready`` all stay ``False``.  The NVIDIA NSCLv1 non-commercial weight
license is the user's responsibility.  Clean-room Lulynx module.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional

import torch

try:  # package import
    from .pid_decoder_backend import (
        build_pid_capability_report,
        build_pid_decode_request,
        build_pid_decode_runtime_boundary,
        run_pid_tiny_decode_smoke,
    )
except ImportError:  # pragma: no cover - direct-file smoke fallback
    from core.lulynx_trainer.pid_decoder_backend import (
        build_pid_capability_report,
        build_pid_decode_request,
        build_pid_decode_runtime_boundary,
        run_pid_tiny_decode_smoke,
    )


# Latent -> pixel decode callable (the default VAE backend the seam wraps).
VaeDecodeFn = Callable[[torch.Tensor], torch.Tensor]


def decode_with_pid_reserve(
    latents: torch.Tensor,
    *,
    vae_decode_fn: VaeDecodeFn,
    request: Any = None,
    allow_pid: bool = False,
    checkpoint_root: Optional[str] = None,
    source_tree: str = "ref/PiD-main",
) -> dict:
    """Route latent->image decode to VAE (default) or PiD (opt-in, weight-gated).

    Returns a dict with ``backend_used`` (``"vae"``/``"pid"``), ``pixels``, and
    honesty flags.  With no ``request`` / ``allow_pid=False`` this is exactly
    ``vae_decode_fn(latents)`` (parity).  PiD only runs when explicitly requested,
    allowed, and backed by a user-supplied local checkpoint; otherwise the seam
    falls back to VAE and records ``pid_blocked``.
    """
    if request is None:
        req = build_pid_decode_request({"decode_backend": "vae"})
    else:
        req = build_pid_decode_request(request)

    def _vae_result(*, pid_requested: bool, pid_blocked: bool, reasons: list) -> dict:
        return {
            "backend_used": "vae",
            "pixels": vae_decode_fn(latents),
            "pid_requested": pid_requested,
            "pid_blocked": pid_blocked,
            "blocked_reasons": reasons,
            "vae_replacement_allowed": False,
            "runtime_activation_enabled": False,
            "default_behavior_changed": False,
        }

    # Gate 1: default backend or master opt-in off -> VAE (parity).
    if req.decode_backend != "pid" or not allow_pid:
        return _vae_result(pid_requested=(req.decode_backend == "pid"), pid_blocked=False, reasons=[])

    # Gate 2: PiD requested + allowed -> only run when user weights are present.
    capability = build_pid_capability_report(req, source_tree=source_tree, checkpoint_root=checkpoint_root)
    resolution = capability["checkpoint_resolution"]
    if capability["checkpoint_available"] and capability["tiny_smoke_loader_ready"]:
        result = run_pid_tiny_decode_smoke(latents, resolution["selected_checkpoint"], req)
        return {
            "backend_used": "pid",
            "pixels": result["pixels"],
            "pid_requested": True,
            "pid_blocked": False,
            "blocked_reasons": [],
            "pid_result": result,
            "selected_checkpoint": resolution["selected_checkpoint"],
            "vae_replacement_allowed": False,
            "runtime_activation_enabled": False,
            "default_behavior_changed": False,
        }

    reasons = list(capability["blocked_reasons"]) or ["pid_checkpoint_missing"]
    return _vae_result(pid_requested=True, pid_blocked=True, reasons=reasons)


def pid_decode_reserve_readiness() -> dict:
    """Read-only readiness report for the P5 PiD decode reserve seam."""
    boundary = build_pid_decode_runtime_boundary({"official_loader_preflight_ready": False})
    return {
        "family": "anima",
        "scope": "pid_decode_reserve_only",
        "default_backend": "vae",
        "pid_opt_in": True,
        "wired": True,
        "tiny_decode_route_ready": True,
        # Honesty red-lines: opt-in only, weight-gated, never bundles/downloads.
        "vae_replacement_allowed": False,
        "runtime_activation_enabled": False,
        "request_fields_emitted": False,
        "auto_download_allowed": False,
        "bundle_weights_allowed": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "runtime_boundary": boundary,
    }


__all__ = [
    "decode_with_pid_reserve",
    "pid_decode_reserve_readiness",
]

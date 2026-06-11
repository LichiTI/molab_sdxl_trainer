"""
Warehouse runtime hooks for manual B-tier experiments.

These paths are deliberately opt-in and conservative.  They reuse the
existing math modules, but keep orchestration local to the trainer instead of
reviving the old Lulynx wrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _split_tokens(value: Any) -> List[str]:
    return [
        part.strip()
        for part in str(value or "").replace(";", ",").replace("\n", ",").split(",")
        if part.strip()
    ]


def _feature_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _feature_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _feature_tensor(item)
            if tensor is not None:
                return tensor
    return None


def normalize_feature_for_geometry(tensor: torch.Tensor) -> torch.Tensor:
    """Return features as [batch, tokens, channels] for geometry losses."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 4:
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, tensor.shape[1])
    if tensor.dim() > 4:
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
    return tensor.reshape(1, 1, -1)


class TrainingFeatureCapture:
    """Small forward-hook collector shared by Manifold and Ghost Replay."""

    def __init__(self, model: torch.nn.Module, anchor_layers: Optional[Iterable[str]] = None):
        self.model = model
        self.anchor_layers = list(anchor_layers or [])
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []

    def _should_track(self, name: str) -> bool:
        if self.anchor_layers:
            return any(anchor in name for anchor in self.anchor_layers)
        lower = name.lower()
        if "mid_block" in lower:
            return True
        if "double" in lower and any(f".{idx}." in name for idx in (4, 10)):
            return True
        if "single" in lower and ".0." in name:
            return True
        return False

    def install(self) -> "TrainingFeatureCapture":
        if self._hooks:
            return self

        def make_hook(name: str):
            def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
                tensor = _feature_tensor(output)
                if tensor is not None and tensor.dim() >= 2:
                    self.features[name] = normalize_feature_for_geometry(tensor)
            return hook

        for name, module in self.model.named_modules():
            if name and self._should_track(name):
                self._hooks.append(module.register_forward_hook(make_hook(name)))
        return self

    def clear(self) -> None:
        self.features.clear()

    def close(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
        self.clear()

    @property
    def installed(self) -> bool:
        return bool(self._hooks)


class BTierRuntime:
    """Manual Manifold/Ghost Replay training loss runtime."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: Any,
        model_arch: str = "",
        manifold_enabled: bool = False,
        manifold_weight: float = 0.01,
        proj_dim: int = 128,
        sparse_freq: int = 1,
        anchor_layers: str = "",
        ghost_enabled: bool = False,
        ghost_path: str = "",
        ghost_interval: int = 100,
        ghost_weight: float = 0.05,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model_arch = str(model_arch or "").strip().lower()
        self.manifold_enabled = bool(manifold_enabled) and float(manifold_weight or 0.0) > 0.0
        self.manifold_weight = max(float(manifold_weight or 0.0), 0.0)
        self.ghost_requested = bool(ghost_enabled) and float(ghost_weight or 0.0) > 0.0
        self.ghost_enabled = bool(self.ghost_requested)
        self.ghost_path = str(ghost_path or "").strip()
        self.ghost_interval = max(int(ghost_interval or 100), 1)
        self.ghost_weight = max(float(ghost_weight or 0.0), 0.0)
        self.anchor_layers = _split_tokens(anchor_layers)
        self.capture: Optional[TrainingFeatureCapture] = None
        self.manifold = None
        self.ghost = None
        self._warned: set[str] = set()
        self.stats: Dict[str, Any] = {
            "enabled": self.manifold_enabled or self.ghost_enabled,
            "manifold_enabled": self.manifold_enabled,
            "ghost_enabled": self.ghost_enabled,
            "installed_hooks": 0,
            "manifold_baseline_ready": False,
            "ghost_replay": {
                "requested": self.ghost_requested,
                "enabled": self.ghost_enabled,
                "loaded": False,
                "path": self.ghost_path,
                "interval": self.ghost_interval,
                "weight": self.ghost_weight,
                "attempts": 0,
                "matches": 0,
                "misses": 0,
                "errors": 0,
                "loss_events": 0,
                "total_loss": 0.0,
                "avg_loss": 0.0,
                "last_loss": None,
                "last_status": "idle",
                "last_timestep": None,
                "last_matched_layers": 0,
                "compatibility": {},
            },
            "last": {},
        }

        if self.manifold_enabled:
            try:
                from ..training_components.manifold_constraint import ManifoldConstraint, LogMethod

                self.manifold = ManifoldConstraint(
                    proj_dim=max(int(proj_dim or 128), 1),
                    anchor_layers=self.anchor_layers or None,
                    device=str(self.device),
                    log_method=LogMethod.PADE_SERIES,
                    sparse_freq=max(int(sparse_freq or 1), 1),
                )
            except Exception as exc:
                self.manifold_enabled = False
                logger.warning("[B-tier] Manifold runtime disabled: %s", exc)

        if self.ghost_enabled:
            self._load_ghost()

        if self.manifold_enabled or self.ghost_enabled:
            self.capture = TrainingFeatureCapture(model, self.anchor_layers).install()
            self.stats["installed_hooks"] = len(self.capture._hooks)
            if not self.capture.installed:
                self._warn_once("no_hooks", "[B-tier] No feature hooks matched anchor layers; Manifold/Ghost losses will be no-op.")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    def _ghost_stats(self) -> Dict[str, Any]:
        return self.stats.setdefault("ghost_replay", {})

    def _ghost_state_snapshot(self, *, status: str, captured_layers: int) -> Dict[str, Any]:
        ghost_stats = self._ghost_stats()
        compatibility = ghost_stats.get("compatibility", {}) or {}
        warnings = list(compatibility.get("warnings", []))[:2]
        return {
            "requested": bool(ghost_stats.get("requested", self.ghost_requested)),
            "enabled": bool(ghost_stats.get("enabled", False)),
            "loaded": bool(ghost_stats.get("loaded", False)),
            "path": self.ghost_path,
            "interval": int(ghost_stats.get("interval", self.ghost_interval) or self.ghost_interval),
            "weight": float(ghost_stats.get("weight", self.ghost_weight) or self.ghost_weight),
            "attempts": int(ghost_stats.get("attempts", 0) or 0),
            "matches": int(ghost_stats.get("matches", 0) or 0),
            "misses": int(ghost_stats.get("misses", 0) or 0),
            "errors": int(ghost_stats.get("errors", 0) or 0),
            "loss_events": int(ghost_stats.get("loss_events", 0) or 0),
            "avg_loss": float(ghost_stats.get("avg_loss", 0.0) or 0.0),
            "last_loss": ghost_stats.get("last_loss"),
            "last_status": str(status or ghost_stats.get("last_status", "idle")),
            "last_timestep": ghost_stats.get("last_timestep"),
            "last_matched_layers": int(ghost_stats.get("last_matched_layers", 0) or 0),
            "compatibility_status": str(compatibility.get("status", "unknown") or "unknown"),
            "recorded_layer_count": int(compatibility.get("recorded_layer_count", 0) or 0),
            "model_matched_layer_count": int(compatibility.get("matched_layer_count", 0) or 0),
            "warning_count": len(compatibility.get("warnings", []) or []),
            "warnings": warnings,
            "captured_layers": int(captured_layers or 0),
        }

    def _load_ghost(self) -> None:
        if not self.ghost_path:
            self.ghost_enabled = False
            self._ghost_stats()["enabled"] = False
            self.stats["ghost_enabled"] = False
            self._warn_once("ghost_missing_path", "[B-tier] Ghost Replay enabled without lulynx_ghost_path; disabling replay.")
            return
        path = Path(self.ghost_path)
        if not path.exists():
            self.ghost_enabled = False
            self._ghost_stats()["enabled"] = False
            self.stats["ghost_enabled"] = False
            self._warn_once("ghost_missing_file", f"[B-tier] Ghost Replay fingerprint not found: {path}")
            return
        try:
            from ..training_components.ghost_replay import GhostReplayer

            self.ghost = GhostReplayer.load(str(path))
            self.ghost.device = str(self.device)
            compatibility = self.ghost.validate_against_model(
                self.model,
                model_arch=self.model_arch,
                anchor_layers=self.anchor_layers or None,
            )
            ghost_stats = self._ghost_stats()
            ghost_stats["loaded"] = True
            ghost_stats["compatibility"] = compatibility
            self.stats["ghost_enabled"] = self.ghost_enabled
            self.stats["ghost_info"] = self.ghost.info
            if compatibility.get("warnings"):
                self._warn_once(
                    "ghost_compat_warning",
                    "[B-tier] Ghost Replay compatibility warning: "
                    + "; ".join(str(item) for item in compatibility["warnings"][:2]),
                )
            if not compatibility.get("usable", True):
                self.ghost_enabled = False
                ghost_stats["enabled"] = False
                self.stats["ghost_enabled"] = False
                self.ghost = None
                self._warn_once(
                    "ghost_incompatible",
                    "[B-tier] Ghost Replay fingerprint is incompatible with the current model; disabling replay.",
                )
        except Exception as exc:
            self.ghost_enabled = False
            self._ghost_stats()["enabled"] = False
            self.stats["ghost_enabled"] = False
            self._warn_once("ghost_load_failed", f"[B-tier] Ghost Replay load failed: {exc}")

    def compute_loss(self, *, step: int, timesteps: Any, loss_device: torch.device) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if self.capture is None:
            state = {"captured_layers": 0}
            if self.ghost_requested or self.ghost is not None:
                state["ghost_replay"] = self._ghost_state_snapshot(status="capture_unavailable", captured_layers=0)
            self.stats["last"] = state
            return None, state

        if not self.capture.features:
            state = {"captured_layers": 0}
            if self.ghost_requested or self.ghost is not None:
                state["ghost_replay"] = self._ghost_state_snapshot(status="no_features", captured_layers=0)
            self.stats["last"] = state
            return None, state

        features = dict(self.capture.features)
        extras: List[torch.Tensor] = []
        state: Dict[str, Any] = {"captured_layers": len(features)}

        if self.manifold_enabled and self.manifold is not None:
            if not getattr(self.manifold, "_baseline_grams", {}):
                with torch.no_grad():
                    for name, feat in features.items():
                        self.manifold._baseline_grams[name] = self.manifold.compute_gram(feat.detach()).detach()
                self.stats["manifold_baseline_ready"] = True
                state["manifold_baseline_captured"] = len(getattr(self.manifold, "_baseline_grams", {}))
            else:
                manifold_loss = self.manifold.compute_loss(features, weight=self.manifold_weight).to(loss_device)
                if torch.isfinite(manifold_loss).all() and float(manifold_loss.detach().abs().item()) > 0.0:
                    extras.append(manifold_loss)
                    state["manifold_loss"] = float(manifold_loss.detach().float().item())

        if self.ghost_enabled and self.ghost is not None:
            ghost_stats = self._ghost_stats()
            ghost_status = "interval_skip"
            try:
                if int(step) % self.ghost_interval == 0:
                    if hasattr(timesteps, "detach"):
                        timestep = int(timesteps.detach().flatten()[0].item())
                    else:
                        timestep = int(timesteps)
                    ghost_stats["attempts"] = int(ghost_stats.get("attempts", 0) or 0) + 1
                    ghost_loss = self.ghost.compute_loss(
                        features,
                        timestep=timestep,
                        sample_idx=0,
                        weight=self.ghost_weight,
                    ).to(loss_device)
                    last_result = dict(getattr(self.ghost, "last_result", {}) or {})
                    ghost_stats["last_timestep"] = int(timestep)
                    ghost_stats["last_matched_layers"] = int(last_result.get("matched_layers", 0) or 0)
                    if last_result.get("matched"):
                        ghost_stats["matches"] = int(ghost_stats.get("matches", 0) or 0) + 1
                    else:
                        ghost_stats["misses"] = int(ghost_stats.get("misses", 0) or 0) + 1
                    if torch.isfinite(ghost_loss).all():
                        ghost_value = float(ghost_loss.detach().float().item())
                        ghost_stats["last_loss"] = ghost_value
                        if last_result.get("matched"):
                            ghost_stats["loss_events"] = int(ghost_stats.get("loss_events", 0) or 0) + 1
                            ghost_stats["total_loss"] = float(ghost_stats.get("total_loss", 0.0) or 0.0) + ghost_value
                            ghost_stats["avg_loss"] = ghost_stats["total_loss"] / max(
                                int(ghost_stats.get("loss_events", 0) or 0),
                                1,
                            )
                        if float(ghost_loss.detach().abs().item()) > 0.0:
                            extras.append(ghost_loss)
                            state["ghost_loss"] = ghost_value
                            ghost_status = "matched"
                        elif last_result.get("matched"):
                            ghost_status = "matched_zero_loss"
                        else:
                            state["ghost_matched"] = False
                            ghost_status = "no_match"
                    else:
                        state["ghost_matched"] = False
                        ghost_status = "non_finite"
                ghost_stats["last_status"] = ghost_status
            except Exception as exc:
                ghost_stats["errors"] = int(ghost_stats.get("errors", 0) or 0) + 1
                ghost_stats["last_status"] = "compute_error"
                self._warn_once("ghost_compute_failed", f"[B-tier] Ghost Replay compute failed once; continuing without replay loss: {exc}")
            state["ghost_replay"] = self._ghost_state_snapshot(
                status=str(self._ghost_stats().get("last_status", ghost_status)),
                captured_layers=len(features),
            )

        self.stats["last"] = state
        if not extras:
            return None, state
        total = extras[0]
        for item in extras[1:]:
            total = total + item
        return total, state

    def clear(self) -> None:
        if self.capture is not None:
            self.capture.clear()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.close()


def run_hutchinson_auto_freeze(
    model: torch.nn.Module,
    *,
    output_dir: Any,
    num_probes: int = 30,
    freeze_ratio: float = 0.5,
    device: Any = "cpu",
) -> Dict[str, Any]:
    """Run an opt-in Hutchinson scan and freeze low-entropy trainable params."""
    from ..training_components.hutchinson_scan import HutchinsonScanner

    freeze_ratio = min(max(float(freeze_ratio or 0.0), 0.0), 1.0)
    scanner = HutchinsonScanner(num_probes=max(int(num_probes or 30), 1), device=str(torch.device(device)))
    results = scanner.scan(model)
    ordered = sorted(results, key=lambda item: (item.entropy, item.trace, item.name))
    freeze_count = int(len(ordered) * freeze_ratio)
    selected = {item.name for item in ordered[:freeze_count]}
    frozen_params = 0
    frozen_tensors = 0
    for name, param in model.named_parameters():
        if name in selected and param.requires_grad:
            param.requires_grad_(False)
            frozen_params += int(param.numel())
            frozen_tensors += 1

    report = {
        "enabled": True,
        "num_probes": scanner.num_probes,
        "freeze_ratio": freeze_ratio,
        "scanned_layers": len(results),
        "frozen_tensors": frozen_tensors,
        "frozen_params": frozen_params,
        "stats": scanner.get_stats(),
        "heatmap": scanner.generate_heatmap(),
        "frozen_layers": sorted(selected),
    }
    try:
        path = Path(output_dir) / "hutchinson_scan_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(path)
    except Exception as exc:
        report["report_error"] = str(exc)
    return report


# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Train/readback matrix for supported LoRA-family adapter formats.

The matrix intentionally uses tiny in-process model fixtures by default.  It
still performs real forward/backward/optimizer steps and writes real adapter
artifacts, but avoids spending hours and VRAM just to answer whether an adapter
format can train, save, and be read back.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.configs import LyCORISAlgo, ModelArch, NetworkType, UnifiedTrainingConfig
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.lora_injector import LoRAInjector
from core.lulynx_trainer.lycoris_layers import LyCORISConfig, LyCORISInjector, LyCORISType
from core.lulynx_trainer.trainer import LulynxTrainer


DEFAULT_ADAPTERS = (
    "lora",
    "lora_plus",
    "dora",
    "lora_fa",
    "vera",
    "tlora",
    "hydralora",
    "fera",
    "loha",
    "locon",
    "lokr",
    "ia3",
    "full",
    "diag_oft",
)
DEFAULT_FAMILIES = ("sdxl", "anima", "newbie")
LYCORIS_ADAPTERS = {"loha", "locon", "lokr", "ia3", "full", "diag-oft", "diag_oft"}


class TinySDXL(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(torch.tanh(self.to_q(x) + self.to_k(x)))


class TinyAnimaBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.layer2 = nn.Linear(dim * 2, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.self_attn.output_proj(
            torch.tanh(self.self_attn.q_proj(x) + self.self_attn.k_proj(x) + self.self_attn.v_proj(x))
        )
        return self.mlp.layer2(F.silu(self.mlp.layer1(x + attn)))


class TinyAnima(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = nn.Module()
        self.net.blocks = nn.ModuleList([TinyAnimaBlock(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net.blocks[0](x)


class TinyNewbie(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.qkv = nn.Linear(dim, dim, bias=False)
        self.attention.out = nn.Linear(dim, dim, bias=False)
        self.feed_forward = nn.Module()
        self.feed_forward.w1 = nn.Linear(dim, dim, bias=False)
        self.feed_forward.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention.out(torch.tanh(self.attention.qkv(x)))
        return self.feed_forward.w2(F.silu(self.feed_forward.w1(x)))


@dataclass
class MatrixResult:
    family: str
    adapter: str
    ok: bool
    artifact: str
    steps: int
    key_count: int = 0
    metadata: dict[str, str] | None = None
    error: str = ""


def _normalize_adapter(adapter: str) -> str:
    adapter = adapter.strip().lower().replace("-", "_")
    return "diag_oft" if adapter in {"diag-oft", "oft"} else adapter


def _family_model_and_targets(family: str) -> tuple[nn.Module, list[str]]:
    if family == "sdxl":
        return TinySDXL(), ["to_q", "to_k", "ff"]
    if family == "anima":
        return TinyAnima(), ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.output_proj", "mlp.layer1", "mlp.layer2"]
    if family == "newbie":
        return TinyNewbie(), ["attention.qkv", "attention.out", "feed_forward.w1", "feed_forward.w2"]
    raise ValueError(f"Unsupported family: {family}")


def _make_config(family: str, adapter: str, steps: int, output_dir: Path) -> Any:
    adapter = _normalize_adapter(adapter)
    data: dict[str, Any] = {
        "schema_id": f"{family}-lora",
        "model_type": family,
        "training_type": "lora",
        "pretrained_model_name_or_path": f"H:/tmp/{family}-matrix-placeholder.safetensors",
        "train_data_dir": f"H:/tmp/{family}-matrix-data",
        "output_dir": str(output_dir),
        "output_name": f"{family}_{adapter}",
        "network_dim": 2,
        "network_alpha": 2,
        "learning_rate": 1e-3,
        "optimizer_type": "AdamW",
        "lr_scheduler": "constant",
        "max_train_steps": steps,
        "lokr_factor": -1,
        "lokr_no_materialize_forward": True,
        "hydralora_num_experts": 2,
        "hydralora_top_k": 1,
        "hydralora_sparse_top_k": True,
        "tlora_rank_schedule": "linear",
    }
    if family == "newbie":
        data["adapter_type"] = adapter
    else:
        data["lora_type"] = adapter
    return ConfigAdapter.from_frontend_dict(data)


def _is_lycoris_config(config: Any, family: str, adapter: str) -> bool:
    network_module = str(getattr(getattr(config, "network_module", ""), "value", getattr(config, "network_module", "")))
    if network_module == "lycoris.locon":
        return True
    if family == "newbie" and adapter in LYCORIS_ADAPTERS:
        return True
    return False


def _build_injector(config: Any, family: str, adapter: str) -> Any:
    if _is_lycoris_config(config, family, adapter):
        lycoris_algo = str(getattr(getattr(config, "lycoris_algo", ""), "value", getattr(config, "lycoris_algo", "")))
        return LyCORISInjector(
            LyCORISConfig(
                lycoris_type=LyCORISType(lycoris_algo),
                rank=int(config.network_dim),
                alpha=float(config.network_alpha),
                dropout=float(config.network_dropout),
                lokr_factor=int(getattr(config, "lycoris_lokr_factor", -1)),
                lokr_rank_dropout=float(getattr(config, "lokr_rank_dropout", 0.0)),
                lokr_module_dropout=float(getattr(config, "lokr_module_dropout", 0.0)),
                lokr_no_materialize_forward=bool(getattr(config, "lokr_no_materialize_forward", False)),
                train_norm=bool(getattr(config, "lycoris_train_norm", False) or getattr(config, "lokr_train_norm", False)),
            )
        )

    network_module = str(getattr(getattr(config, "network_module", ""), "value", getattr(config, "network_module", "")))
    return LoRAInjector(
        rank=int(config.network_dim),
        alpha=float(config.network_alpha),
        dropout=float(config.network_dropout),
        model_arch=family,
        dora_enabled=bool(getattr(config, "use_dora", False) or getattr(config, "dora_enabled", False)),
        tlora_enabled=network_module.endswith("tlora"),
        tlora_min_rank=int(getattr(config, "tlora_min_rank", 1)),
        tlora_rank_schedule=str(getattr(config, "tlora_rank_schedule", "linear")),
        tlora_orthogonal_init=bool(getattr(config, "tlora_orthogonal_init", False)),
        tlora_total_steps=max(int(getattr(config, "max_train_steps", 1) or 1), 1),
        vera_enabled=network_module == "networks.vera" or bool(getattr(config, "vera_enabled", False)),
        vera_d_initial=float(getattr(config, "vera_d_initial", 0.1)),
        vera_prng_key=int(getattr(config, "vera_prng_key", 0)),
        lora_fa_enabled=network_module == "networks.lora_fa" or bool(getattr(config, "lora_fa_enabled", False)),
        hydralora_enabled=bool(getattr(config, "hydralora_enabled", False)),
        hydralora_num_experts=int(getattr(config, "hydralora_num_experts", 2) or 2),
        hydralora_routing=str(getattr(config, "hydralora_routing", "top_k")),
        hydralora_top_k=int(getattr(config, "hydralora_top_k", 1) or 1),
        hydralora_sparse_top_k=bool(getattr(config, "hydralora_sparse_top_k", False)),
        fera_enabled=bool(getattr(config, "fera_enabled", False)),
        fera_gate_init=float(getattr(config, "fera_gate_init", 0.0) or 0.0),
    )


def _metadata_for(config: Any, family: str, adapter: str, steps: int) -> dict[str, str]:
    network_module = str(getattr(getattr(config, "network_module", ""), "value", getattr(config, "network_module", "")))
    metadata = {
        "ss_base_model_version": family,
        "ss_output_name": f"{family}_{adapter}",
        "ss_network_module": network_module,
        "ss_network_dim": str(config.network_dim),
        "ss_network_alpha": str(config.network_alpha),
        "ss_training_step": str(steps),
        "lulynx_matrix_family": family,
        "lulynx_matrix_adapter": adapter,
    }
    lycoris_algo = str(getattr(getattr(config, "lycoris_algo", ""), "value", getattr(config, "lycoris_algo", "")))
    if lycoris_algo:
        metadata["ss_lycoris_algo"] = lycoris_algo
    return metadata


def _prepare_export_state(config: Any, family: str, state: dict[str, torch.Tensor], metadata: dict[str, str]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if family != "anima":
        return state, metadata
    # Reuse the production save compatibility hook, especially for Anima LoKr.
    trainer_cfg = UnifiedTrainingConfig(
        model_type=ModelArch.ANIMA,
        network_module=NetworkType.LYCORIS if str(getattr(config.network_module, "value", config.network_module)) == "lycoris.locon" else config.network_module,
        lycoris_algo=LyCORISAlgo.LOKR if str(getattr(config.lycoris_algo, "value", config.lycoris_algo)) == "lokr" else config.lycoris_algo,
    )
    trainer = LulynxTrainer(trainer_cfg)
    return trainer._prepare_anima_lokr_export_for_save(state, metadata)


def _readback(path: Path) -> tuple[int, dict[str, str]]:
    with safe_open(str(path), framework="pt") as handle:
        keys = list(handle.keys())
        metadata = dict(handle.metadata() or {})
        if not keys:
            raise RuntimeError("saved adapter has no tensors")
        for key in keys:
            tensor = handle.get_tensor(key)
            if tensor.numel() > 0 and not torch.isfinite(tensor.float()).all():
                raise RuntimeError(f"non-finite tensor in {key}")
        return len(keys), metadata


def train_case(family: str, adapter: str, steps: int, output_dir: Path, device: torch.device) -> MatrixResult:
    adapter = _normalize_adapter(adapter)
    artifact = output_dir / f"{family}_{adapter}.safetensors"
    try:
        config = _make_config(family, adapter, steps, output_dir)
        model, targets = _family_model_and_targets(family)
        model.to(device)

        injector = _build_injector(config, family, adapter)
        injected = injector.inject(model, targets, prefix="unet")
        if not injected:
            raise RuntimeError("no adapter layers were injected")

        optimizer = torch.optim.AdamW(injector.get_trainable_params(), lr=float(config.learning_rate))
        torch.manual_seed(1337)
        for step in range(steps):
            x = torch.randn(2, 3, 8, device=device)
            target = torch.zeros_like(model(x).detach())
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x).float(), target.float())
            loss.backward()
            optimizer.step()

        state = {key: value.detach().cpu().contiguous() for key, value in injector.get_lora_state_dict().items()}
        metadata = _metadata_for(config, family, adapter, steps)
        state, metadata = _prepare_export_state(config, family, state, metadata)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_file(state, str(artifact), metadata=metadata)
        key_count, read_meta = _readback(artifact)
        return MatrixResult(family=family, adapter=adapter, ok=True, artifact=str(artifact), steps=steps, key_count=key_count, metadata=read_meta)
    except Exception as exc:
        return MatrixResult(family=family, adapter=adapter, ok=False, artifact=str(artifact), steps=steps, error=str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/readback supported LoRA adapter matrix.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", default="H:/tmp/lulynx_lora_training_matrix")
    parser.add_argument("--families", nargs="*", default=list(DEFAULT_FAMILIES), choices=list(DEFAULT_FAMILIES))
    parser.add_argument("--adapters", nargs="*", default=list(DEFAULT_ADAPTERS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    steps = max(1, int(args.steps))
    output_dir = Path(args.output_dir)

    started = time.perf_counter()
    results: list[MatrixResult] = []
    for family in args.families:
        for adapter in args.adapters:
            result = train_case(family, adapter, steps, output_dir, device)
            results.append(result)
            status = "PASS" if result.ok else "FAIL"
            detail = f"keys={result.key_count}" if result.ok else result.error
            print(f"{status}: {family}/{_normalize_adapter(adapter)} -> {result.artifact} ({detail})", flush=True)

    report = {
        "ok": all(result.ok for result in results),
        "steps": steps,
        "device": str(device),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "results": [asdict(result) for result in results],
    }
    report_path = Path(args.json) if args.json else output_dir / "matrix_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

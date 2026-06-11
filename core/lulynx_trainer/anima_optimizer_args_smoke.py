# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test custom optimizer args parsing and filtering for Anima trainer.

Proves:
1. _parse_custom_args parses comma-separated key=value pairs.
2. _filtered_custom_args keeps allowed keys and drops disallowed ones.
3. Boolean-like strings (true/false/yes/no/on/off) are parsed correctly.
4. Anima grouped-LR param groups receive custom args through to the optimizer.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_xformers_stub()
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
configs_mod = _load_module("core.configs", CORE_ROOT / "configs.py")
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
_load_module("core.lulynx_trainer.anima_targets", TRAINER_ROOT / "anima_targets.py")
_load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")
trainer_mod = _load_module("core.lulynx_trainer.trainer", TRAINER_ROOT / "trainer.py")

OptimizerType = configs_mod.OptimizerType
SchedulerType = configs_mod.SchedulerType
LulynxTrainer = trainer_mod.LulynxTrainer


def _make_trainer(optimizer_type=OptimizerType.ADAMW, optimizer_args="", optimizer_backend="auto") -> LulynxTrainer:
    trainer = LulynxTrainer.__new__(LulynxTrainer)
    trainer.config = SimpleNamespace(
        optimizer=optimizer_type,
        optimizer_args=optimizer_args,
        optimizer_backend=optimizer_backend,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_args="",
        lr_scheduler_num_cycles=1,
        scheduler=SchedulerType.COSINE,
        semantic_tuner_enabled=False,
        model_arch="anima",
    )
    trainer._log = lambda _msg: None
    return trainer


def test_parse_custom_args() -> None:
    """_parse_custom_args handles key=value, booleans, and numeric values."""
    trainer = _make_trainer()
    result = trainer._parse_custom_args("eps=1e-8,amsgrad=true,foreach=no")
    assert result["eps"] == 1e-8, f"Expected eps=1e-8, got {result['eps']}"
    assert result["amsgrad"] is True, f"Expected amsgrad=True, got {result['amsgrad']}"
    assert result["foreach"] is False, f"Expected foreach=False, got {result['foreach']}"


def test_filtered_keeps_allowed() -> None:
    """_filtered_custom_args keeps keys in the allowed set."""
    trainer = _make_trainer(optimizer_args="eps=1e-8,amsgrad=true")
    allowed = {"eps", "amsgrad", "betas"}
    result = trainer._filtered_custom_args("eps=1e-8,amsgrad=true", allowed, "test")
    assert "eps" in result
    assert "amsgrad" in result


def test_filtered_drops_disallowed() -> None:
    """_filtered_custom_args drops keys not in the allowed set."""
    trainer = _make_trainer(optimizer_args="eps=1e-8,evil_key=42")
    allowed = {"eps"}
    result = trainer._filtered_custom_args("eps=1e-8,evil_key=42", allowed, "test")
    assert "eps" in result
    assert "evil_key" not in result


def test_adamw_allowed_args() -> None:
    """AdamW optimizer accepts the documented set of custom args."""
    trainer = _make_trainer(OptimizerType.ADAMW)
    allowed = trainer._optimizer_allowed_args()
    expected = {"betas", "eps", "amsgrad", "foreach", "maximize", "capturable", "fused"}
    assert allowed == expected, f"AdamW allowed args mismatch: {allowed}"


def test_prodigy_allowed_args() -> None:
    """Prodigy optimizer accepts its documented set of custom args."""
    trainer = _make_trainer(OptimizerType.PRODIGY)
    allowed = trainer._optimizer_allowed_args()
    assert "safeguard_warmup" in allowed, "Prodigy should allow safeguard_warmup"
    assert "d_coef" in allowed, "Prodigy should allow d_coef from frontend optimizer_args"


def test_prodigy_plus_schedule_free_allowed_args() -> None:
    """ProdigyPlusScheduleFree exposes its focused custom-arg allowlist."""
    trainer = _make_trainer(OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE)
    allowed = trainer._optimizer_allowed_args()
    for key in ("d0", "d_coef", "use_schedulefree", "schedulefree_c", "factored"):
        assert key in allowed, f"ProdigyPlusScheduleFree should allow {key}"


def test_automagic_plus_plus_allowed_args() -> None:
    """Automagic++ optimizer accepts Warehouse stability controls."""
    trainer = _make_trainer(OptimizerType.AUTOMAGIC_PLUS_PLUS)
    allowed = trainer._optimizer_allowed_args()
    for key in ("min_lr", "max_lr", "lr_up", "lr_down", "max_update_rms_ratio", "lr_granularity"):
        assert key in allowed, f"Automagic++ should allow {key}"


def test_auto_prodigy_allowed_args() -> None:
    """AutoProdigy optimizer accepts Warehouse stability controls."""
    trainer = _make_trainer(OptimizerType.AUTO_PRODIGY)
    allowed = trainer._optimizer_allowed_args()
    for key in ("d0", "d_coef", "growth_rate", "max_update_rms_ratio", "damping"):
        assert key in allowed, f"AutoProdigy should allow {key}"


def test_mn_lora_plus_plus_profiles() -> None:
    """MN-LoRA++ profiles map to expected controller parameters."""
    trainer = _make_trainer()
    trainer.config.mn_lora_plus_plus_profile = "balanced"
    balanced = trainer._mn_lora_plus_plus_config()
    assert balanced["lr_up"] == 1.03
    assert balanced["lora_up_max_mult"] == 1.5
    assert balanced["update_rms_cap"] == 0.02

    trainer.config.mn_lora_plus_plus_profile = "aggressive"
    aggressive = trainer._mn_lora_plus_plus_config()
    assert aggressive["lr_up"] > balanced["lr_up"]
    assert aggressive["max_mult"] > balanced["max_mult"]

    trainer.config.mn_lora_plus_plus_profile = "custom"
    trainer.config.mn_lora_plus_plus_lr_up = 1.07
    trainer.config.mn_lora_plus_plus_lora_up_max_mult = 1.8
    custom = trainer._mn_lora_plus_plus_config()
    assert custom["lr_up"] == 1.07
    assert custom["lora_up_max_mult"] == 1.8


def test_auto_prodigy_profiles() -> None:
    """AutoProdigy profiles map to expected Warehouse optimizer parameters."""
    trainer = _make_trainer(OptimizerType.AUTO_PRODIGY)
    trainer.config.auto_prodigy_profile = "balanced"
    balanced = trainer._auto_prodigy_config()
    assert balanced["growth_rate"] == 1.02
    assert balanced["max_update_rms_ratio"] == 0.01

    trainer.config.auto_prodigy_profile = "safe"
    safe = trainer._auto_prodigy_config()
    assert safe["growth_rate"] < balanced["growth_rate"]
    assert safe["damping"] > balanced["damping"]

    trainer.config.auto_prodigy_profile = "aggressive"
    aggressive = trainer._auto_prodigy_config()
    assert aggressive["growth_rate"] > balanced["growth_rate"]
    assert aggressive["max_update_rms_ratio"] > balanced["max_update_rms_ratio"]

    trainer.config.auto_prodigy_profile = "custom"
    trainer.config.auto_prodigy_d0 = 2e-6
    trainer.config.auto_prodigy_d_coef = 1.25
    trainer.config.auto_prodigy_growth_rate = 1.04
    trainer.config.auto_prodigy_max_update_rms_ratio = 0.02
    trainer.config.auto_prodigy_damping = 0.9
    trainer.config.auto_prodigy_beta3 = 0.985
    trainer.config.auto_prodigy_safeguard_warmup = False
    custom = trainer._auto_prodigy_config()
    assert custom["d0"] == 2e-6
    assert custom["d_coef"] == 1.25
    assert custom["growth_rate"] == 1.04
    assert custom["max_update_rms_ratio"] == 0.02
    assert custom["damping"] == 0.9
    assert custom["beta3"] == 0.985
    assert custom["safeguard_warmup"] is False


def test_auto_prodigy_scheduler_constant_guard() -> None:
    """AutoProdigy keeps external LR scheduling constant for every selected scheduler."""
    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], lr=1e-4)
    for scheduler in (
        SchedulerType.COSINE,
        SchedulerType.COSINE_RESTARTS,
        SchedulerType.LINEAR,
        SchedulerType.POLYNOMIAL,
        SchedulerType.CONSTANT,
        SchedulerType.CONSTANT_WARMUP,
    ):
        trainer = _make_trainer(OptimizerType.AUTO_PRODIGY)
        trainer.config.scheduler = scheduler
        lr_scheduler = trainer._create_scheduler(optimizer, total_steps=12)
        assert lr_scheduler.__class__.__name__ == "ConstantLR", (
            f"AutoProdigy should use ConstantLR guard for {scheduler}, "
            f"got {lr_scheduler.__class__.__name__}"
        )
        assert lr_scheduler.get_last_lr()[0] == 1e-4


def test_new_optimizer_allowed_args() -> None:
    """New Warehouse third-party routes expose focused custom-arg allowlists."""
    checks = {
        OptimizerType.PAGED_ADAMW: "betas",
        OptimizerType.PAGED_ADAMW_32BIT: "eps",
        OptimizerType.PAGED_ADAMW_8BIT: "amsgrad",
        OptimizerType.PAGED_LION_8BIT: "betas",
        OptimizerType.SGD_NESTEROV_8BIT: "momentum",
        OptimizerType.DADAPT_ADAGRAD: "eps",
        OptimizerType.DADAPT_LION: "growth_rate",
        OptimizerType.DADAPT_ADAN_IP: "betas",
        OptimizerType.DADAPTATION: "d0",
        OptimizerType.DADAPT_ADAM_PREPRINT: "d0",
        OptimizerType.ADAMW_SCHEDULE_FREE: "warmup_steps",
        OptimizerType.RADAM_SCHEDULE_FREE: "betas",
        OptimizerType.SGD_SCHEDULE_FREE: "momentum",
        OptimizerType.PYTORCH_OPTIMIZER: "name",
    }
    for opt, expected_key in checks.items():
        trainer = _make_trainer(opt)
        allowed = trainer._optimizer_allowed_args()
        assert expected_key in allowed, f"{opt} should allow {expected_key}; got {allowed}"


class _StubOptimizer(torch.optim.SGD):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        self.received_kwargs = dict(kwargs)
        super().__init__(params, lr=lr, weight_decay=weight_decay, momentum=float(kwargs.get("momentum", 0.0)))


class _StubScheduleFreeOptimizer(_StubOptimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        super().__init__(params, lr=lr, weight_decay=weight_decay, **kwargs)
        self.mode = "init"

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


def _install_fake_bitsandbytes() -> None:
    bnb = ModuleType("bitsandbytes")
    optim = SimpleNamespace(
        AdamW8bit=_StubOptimizer,
        PagedAdamW=_StubOptimizer,
        PagedAdamW32bit=_StubOptimizer,
        PagedAdamW8bit=_StubOptimizer,
        PagedLion8bit=_StubOptimizer,
        SGD8bit=_StubOptimizer,
    )
    bnb.optim = optim
    sys.modules["bitsandbytes"] = bnb


def _install_fake_dadaptation() -> None:
    dadaptation = ModuleType("dadaptation")
    experimental = ModuleType("dadaptation.experimental")
    dadaptation.DAdaptAdaGrad = _StubOptimizer
    dadaptation.DAdaptAdam = _StubOptimizer
    dadaptation.DAdaptAdan = _StubOptimizer
    dadaptation.DAdaptLion = _StubOptimizer
    dadaptation.DAdaptSGD = _StubOptimizer
    experimental.DAdaptAdamPreprint = _StubOptimizer
    experimental.DAdaptAdanIP = _StubOptimizer
    sys.modules["dadaptation"] = dadaptation
    sys.modules["dadaptation.experimental"] = experimental


def _install_fake_schedulefree() -> None:
    schedulefree = ModuleType("schedulefree")
    schedulefree.AdamWScheduleFree = _StubScheduleFreeOptimizer
    schedulefree.RAdamScheduleFree = _StubScheduleFreeOptimizer
    schedulefree.SGDScheduleFree = _StubScheduleFreeOptimizer
    sys.modules["schedulefree"] = schedulefree


def _install_fake_prodigyplus() -> None:
    prodigyplus = ModuleType("prodigyplus")
    prodigyplus.ProdigyPlusScheduleFree = _StubScheduleFreeOptimizer
    sys.modules["prodigyplus"] = prodigyplus


def test_optimizer_backend_resolver_routes() -> None:
    """AdamW backend resolver keeps optimizer type and implementation backend separate."""
    param = nn.Parameter(torch.ones(2))

    foreach_trainer = _make_trainer(OptimizerType.ADAMW, optimizer_backend="foreach_adamw")
    foreach_trainer.config.semantic_tuner_enabled = True
    foreach_trainer.trainable_params = [param]
    foreach_optimizer = foreach_trainer._create_optimizer()
    assert foreach_optimizer.__class__.__name__ == "AdamW"
    assert foreach_optimizer.param_groups[0].get("foreach") is True
    assert foreach_trainer._optimizer_backend_profile["resolved"] == "foreach_adamw"

    override_trainer = _make_trainer(
        OptimizerType.ADAMW,
        optimizer_args="foreach=False",
        optimizer_backend="foreach_adamw",
    )
    override_trainer.config.semantic_tuner_enabled = True
    override_trainer.trainable_params = [nn.Parameter(torch.ones(2))]
    override_optimizer = override_trainer._create_optimizer()
    assert override_optimizer.param_groups[0].get("foreach") is False
    assert override_trainer._optimizer_backend_profile["resolved"] == "torch_adamw"
    assert "explicitly disabled" in override_trainer._optimizer_backend_profile["fallback_reason"]

    _install_fake_bitsandbytes()
    bnb_trainer = _make_trainer(OptimizerType.ADAMW_8BIT, optimizer_backend="auto")
    bnb_trainer.config.semantic_tuner_enabled = True
    bnb_trainer.trainable_params = [nn.Parameter(torch.ones(2))]
    bnb_optimizer = bnb_trainer._create_optimizer()
    assert bnb_optimizer.__class__.__name__ == "_StubOptimizer"
    assert bnb_trainer._optimizer_backend_profile["resolved"] == "bnb_8bit"

    fused_trainer = _make_trainer(OptimizerType.ADAMW, optimizer_backend="lulynx_fused")
    fused_trainer.config.semantic_tuner_enabled = True
    fused_trainer.trainable_params = [nn.Parameter(torch.ones(2))]
    fused_optimizer = fused_trainer._create_optimizer()
    assert fused_optimizer.__class__.__name__ == "FusedAdamW"
    assert fused_trainer._optimizer_backend_profile["resolved"] == "lulynx_fused"


def test_create_new_third_party_optimizer_routes() -> None:
    """New routes instantiate through third-party package APIs without rescript glue."""
    _install_fake_bitsandbytes()
    _install_fake_dadaptation()
    _install_fake_schedulefree()
    _install_fake_prodigyplus()
    param = nn.Parameter(torch.ones(2))
    for opt in (
        OptimizerType.PAGED_ADAMW,
        OptimizerType.PAGED_ADAMW_32BIT,
        OptimizerType.PAGED_ADAMW_8BIT,
        OptimizerType.PAGED_LION_8BIT,
        OptimizerType.SGD_NESTEROV_8BIT,
        OptimizerType.DADAPT_ADAGRAD,
        OptimizerType.DADAPT_LION,
        OptimizerType.DADAPT_ADAN_IP,
        OptimizerType.DADAPTATION,
        OptimizerType.DADAPT_ADAM_PREPRINT,
        OptimizerType.ADAMW_SCHEDULE_FREE,
        OptimizerType.RADAM_SCHEDULE_FREE,
        OptimizerType.SGD_SCHEDULE_FREE,
        OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
    ):
        trainer = _make_trainer(opt, optimizer_args="momentum=0.9")
        trainer.config.semantic_tuner_enabled = True
        trainer.trainable_params = [param]
        optimizer = trainer._create_optimizer()
        assert isinstance(optimizer, torch.optim.Optimizer), f"{opt} did not create an optimizer"


def test_schedulefree_scheduler_constant_guard() -> None:
    """Schedule-free optimizers keep external scheduler constant."""
    param = nn.Parameter(torch.ones(2))
    optimizer = _StubScheduleFreeOptimizer([param], lr=1e-4)
    for opt in (
        OptimizerType.ADAMW_SCHEDULE_FREE,
        OptimizerType.RADAM_SCHEDULE_FREE,
        OptimizerType.SGD_SCHEDULE_FREE,
        OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
    ):
        trainer = _make_trainer(opt)
        lr_scheduler = trainer._create_scheduler(optimizer, total_steps=10)
        assert lr_scheduler.__class__.__name__ == "ConstantLR"


def test_create_prodigy_plus_schedule_free_optimizer_with_custom_args() -> None:
    """Trainer routes ProdigyPlusScheduleFree through the prodigyplus package API."""
    _install_fake_prodigyplus()
    param = nn.Parameter(torch.ones(2))
    trainer = _make_trainer(
        OptimizerType.PRODIGY_PLUS_SCHEDULE_FREE,
        optimizer_args="d0=1e-5,d_coef=2.0,use_schedulefree=false,use_speed=true,unsupported=1",
    )
    trainer.config.semantic_tuner_enabled = True
    trainer.trainable_params = [param]

    optimizer = trainer._create_optimizer()
    assert optimizer.__class__.__name__ == "_StubScheduleFreeOptimizer"
    assert optimizer.mode == "train"
    assert optimizer.received_kwargs["d0"] == 1e-5
    assert optimizer.received_kwargs["d_coef"] == 2.0
    assert optimizer.received_kwargs["use_schedulefree"] is False
    assert optimizer.received_kwargs["use_speed"] is True


def test_new_scheduler_routes() -> None:
    """New scheduler routes, including loss-aware cosine variants, are available."""
    param = nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([param], lr=1e-4)

    trainer = _make_trainer(OptimizerType.ADAMW)
    trainer.config.scheduler = SchedulerType.COSINE_WITH_MIN_LR
    trainer.config.lr_scheduler_args = "min_lr_ratio=0.2"
    cosine_min = trainer._create_scheduler(optimizer, total_steps=12)
    assert cosine_min.__class__.__name__ == "LambdaLR"

    trainer.config.scheduler = SchedulerType.PIECEWISE_CONSTANT
    trainer.config.lr_scheduler_args = "rules=0:1.0;5:0.5"
    piecewise = trainer._create_scheduler(optimizer, total_steps=12)
    assert piecewise.__class__.__name__ == "LambdaLR"

    trainer = _make_trainer(OptimizerType.ADAMW)
    trainer.config.scheduler = SchedulerType.LOSS_GATED_COSINE
    trainer.config.warmup_ratio = 0.0
    trainer.config.loss_scheduler_ema_alpha = 1.0
    trainer.config.loss_scheduler_patience = 2
    trainer.config.loss_scheduler_max_hold_steps = 10
    gated_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(2))], lr=1e-4)
    gated = trainer._create_scheduler(gated_optimizer, total_steps=20)
    assert gated.__class__.__name__ == "LossAwareCosineScheduler"
    assert gated.get_loss_aware_state()["mode"] == "gated"
    assert gated.get_loss_aware_state()["auto_max_hold_steps"] is False
    assert gated.get_loss_aware_state()["max_hold_steps"] == 10
    gated.step(1.0)
    gated.step(0.9)
    gated.step(0.8)
    assert gated.get_loss_aware_state()["phase_step"] == 0.0, (
        "Loss-gated cosine should hold while loss is improving"
    )
    gated.step(0.8)
    assert gated.get_loss_aware_state()["phase_step"] == 0.0, (
        "Loss-gated cosine should wait until patience is reached"
    )
    gated.step(0.8)
    assert gated.get_loss_aware_state()["phase_step"] == 1.0, (
        "Loss-gated cosine should advance after a plateau reaches patience"
    )

    trainer = _make_trainer(OptimizerType.ADAMW)
    trainer.config.scheduler = SchedulerType.LOSS_GATED_COSINE
    trainer.config.warmup_ratio = 0.0
    trainer.config.loss_scheduler_ema_alpha = 1.0
    trainer.config.loss_scheduler_max_hold_steps = 0
    capped_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(2))], lr=1e-4)
    capped = trainer._create_scheduler(capped_optimizer, total_steps=20)
    capped_state = capped.get_loss_aware_state()
    assert capped_state["auto_max_hold_steps"] is True
    assert capped_state["max_hold_steps"] == 4
    for loss in (1.0, 0.9, 0.8, 0.7):
        capped.step(loss)
    capped_state = capped.get_loss_aware_state()
    assert capped_state["phase_step"] == 1.0, (
        "Auto max-hold guard should force cosine advancement even while loss keeps improving"
    )
    assert capped_state["action"] == "max_hold_advance"

    trainer = _make_trainer(OptimizerType.ADAMW)
    trainer.config.scheduler = SchedulerType.LOSS_WEIGHTED_ANNEALED_COSINE
    trainer.config.warmup_ratio = 0.0
    trainer.config.loss_scheduler_ema_alpha = 1.0
    trainer.config.loss_scheduler_lock_weight_threshold = 0.0
    trainer.config.loss_scheduler_max_hold_steps = 2
    weighted_optimizer = torch.optim.AdamW([nn.Parameter(torch.ones(2))], lr=1e-4)
    weighted = trainer._create_scheduler(weighted_optimizer, total_steps=20)
    assert weighted.__class__.__name__ == "LossAwareCosineScheduler"
    assert weighted.get_loss_aware_state()["mode"] == "weighted"
    weighted.step(1.0)
    weighted.step(0.9)
    weighted_state = weighted.get_loss_aware_state()
    assert weighted_state["phase_step"] > 0.0
    assert weighted_state["action"] == "weighted_max_hold_advance"


def test_empty_args_string() -> None:
    """Empty optimizer_args string produces empty dict."""
    trainer = _make_trainer(optimizer_args="")
    result = trainer._parse_custom_args("")
    assert result == {}


def test_anima_grouped_lr_with_custom_args() -> None:
    """Anima grouped-LR param groups are compatible with custom optimizer args."""
    injector_mod = sys.modules["core.lulynx_trainer.lora_injector"]

    class _Block(nn.Module):
        def __init__(self, dim: int = 4) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
            self.mlp = nn.Module()
            self.mlp.layer1 = nn.Linear(dim, dim, bias=False)

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block()])

    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = _Net()

    model = _Root()
    injector = injector_mod.LoRAInjector(rank=1, alpha=1, model_arch="anima")
    targets = ["self_attn.q_proj", "mlp.layer1"]
    injector._inject_model(model, targets, prefix="net")

    trainer = _make_trainer(OptimizerType.ADAMW, optimizer_args="eps=1e-6,amsgrad=true")
    trainer.config.anima_self_attn_lr = 1e-4
    trainer.config.anima_mlp_lr = 2e-4
    trainer.config.anima_cross_attn_lr = 0
    trainer.config.anima_mod_lr = 0
    trainer.config.anima_llm_adapter_lr = 0
    trainer.model = type("M", (), {"unet": model})()
    trainer.lora_injector = injector

    param_groups = trainer._build_anima_grouped_param_groups()
    assert param_groups is not None, "Expected grouped param groups"
    assert len(param_groups) == 2, f"Expected 2 groups, got {len(param_groups)}"

    optimizer = trainer._create_optimizer()
    assert optimizer.defaults.get("eps") == 1e-6, (
        f"Expected eps=1e-6 from custom args, got {optimizer.defaults.get('eps')}"
    )
    assert optimizer.defaults.get("amsgrad") is True, (
        f"Expected amsgrad=True from custom args, got {optimizer.defaults.get('amsgrad')}"
    )


def test_create_automagic_plus_plus_optimizer_with_custom_args() -> None:
    """Trainer routes OptimizerType.AUTOMAGIC_PLUS_PLUS to the local Warehouse optimizer."""
    param = nn.Parameter(torch.ones(2))
    trainer = _make_trainer(
        OptimizerType.AUTOMAGIC_PLUS_PLUS,
        optimizer_args="min_lr=1e-7,max_lr=1e-3,lr_up=1.02,lr_granularity=low_overhead,unsupported=1",
    )
    trainer.config.semantic_tuner_enabled = True
    trainer.trainable_params = [param]

    optimizer = trainer._create_optimizer()
    assert optimizer.__class__.__name__ == "AutomagicPlusPlus"
    assert optimizer.min_lr == 1e-7
    assert optimizer.max_lr == 1e-3
    assert optimizer.lr_up == 1.02
    assert optimizer.lr_granularity == "low_overhead"


def test_create_auto_prodigy_optimizer_with_custom_args() -> None:
    """Trainer routes OptimizerType.AUTO_PRODIGY to the local Warehouse optimizer."""
    param = nn.Parameter(torch.ones(2))
    trainer = _make_trainer(
        OptimizerType.AUTO_PRODIGY,
        optimizer_args="d0=1e-5,d_coef=2.0,growth_rate=1.01,unsupported=1",
    )
    trainer.config.semantic_tuner_enabled = True
    trainer.trainable_params = [param]

    optimizer = trainer._create_optimizer()
    assert optimizer.__class__.__name__ == "AutoProdigy"
    assert optimizer.defaults["d0"] == 1e-5
    assert optimizer.defaults["d_coef"] == 2.0
    assert optimizer.defaults["growth_rate"] == 1.01


def main() -> int:
    test_parse_custom_args()
    print("  _parse_custom_args: key=value, booleans, numerics -- PASS")

    test_filtered_keeps_allowed()
    print("  _filtered_custom_args: keeps allowed keys -- PASS")

    test_filtered_drops_disallowed()
    print("  _filtered_custom_args: drops disallowed keys -- PASS")

    test_adamw_allowed_args()
    print("  AdamW allowed args set -- PASS")

    test_prodigy_allowed_args()
    print("  Prodigy allowed args set -- PASS")

    test_prodigy_plus_schedule_free_allowed_args()
    print("  ProdigyPlusScheduleFree allowed args set -- PASS")

    test_automagic_plus_plus_allowed_args()
    print("  Automagic++ allowed args set -- PASS")

    test_auto_prodigy_allowed_args()
    print("  AutoProdigy allowed args set -- PASS")

    test_mn_lora_plus_plus_profiles()
    print("  MN-LoRA++ profiles -- PASS")

    test_auto_prodigy_profiles()
    print("  AutoProdigy profiles -- PASS")

    test_auto_prodigy_scheduler_constant_guard()
    print("  AutoProdigy scheduler constant guard -- PASS")

    test_new_optimizer_allowed_args()
    print("  New optimizer allowed args sets -- PASS")

    test_optimizer_backend_resolver_routes()
    print("  Optimizer backend resolver routes -- PASS")

    test_create_new_third_party_optimizer_routes()
    print("  New third-party optimizer routes -- PASS")

    test_schedulefree_scheduler_constant_guard()
    print("  ScheduleFree scheduler constant guard -- PASS")

    test_new_scheduler_routes()
    print("  New scheduler routes -- PASS")

    test_empty_args_string()
    print("  Empty args string -- PASS")

    test_anima_grouped_lr_with_custom_args()
    print("  Anima grouped-LR + custom optimizer args -- PASS")

    test_create_automagic_plus_plus_optimizer_with_custom_args()
    print("  Automagic++ trainer route + custom optimizer args -- PASS")

    test_create_auto_prodigy_optimizer_with_custom_args()
    print("  AutoProdigy trainer route + custom optimizer args -- PASS")

    test_create_prodigy_plus_schedule_free_optimizer_with_custom_args()
    print("  ProdigyPlusScheduleFree trainer route + custom optimizer args -- PASS")

    print(
        "Anima optimizer-args smoke passed: custom args parsing, filtering, "
        "allowed sets per optimizer, and grouped-LR compatibility"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


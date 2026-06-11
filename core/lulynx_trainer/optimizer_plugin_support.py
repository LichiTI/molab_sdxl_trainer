# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Support matrix for pytorch_optimizer plugin routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PluginOptimizerCase:
    name: str
    extra_args: str = ""
    scheduler_expected: str = "CosineAnnealingLR"
    include_vector_param: bool = True
    expected_param_group_flags: tuple[bool, ...] = ()

    @property
    def canonical_name(self) -> str:
        return self.name.lower()

    @property
    def optimizer_args(self) -> str:
        return " ".join(part for part in (f"name={self.name}", self.extra_args) if part)


PLUGIN_SCHEDULE_FREE_OPTIMIZERS = frozenset({
    "schedulefreeadamw",
    "schedulefreeradam",
    "schedulefreesgd",
})

PLUGIN_MUON_FAMILY_OPTIMIZERS = frozenset({"adago", "adamuon", "muon"})

PLUGIN_DEFAULT_EXTRA_ARGS: dict[str, str] = {
    "adalomo": "loss_scale=1.0",
    "bsam": "num_data=1",
    "demo": "compression_top_k=1 compression_chunk=4",
    "lbfgs": "max_iter=1 history_size=4 line_search_fn=None",
    "ranger21": "num_iterations=8",
    "spam": "density=1.0",
}


PLUGIN_RESUME_SMOKE_PASSED = tuple(sorted({
    "a2grad",
    "accsgd",
    "adabelief",
    "adabound",
    "adadelta",
    "adafactor",
    "adagc",
    "adahessian",
    "adago",
    "adai",
    "adalite",
    "adalomo",
    "adam",
    "adamax",
    "adamc",
    "adamg",
    "adammini",
    "adamod",
    "adamp",
    "adams",
    "adamuon",
    "adamw",
    "adamwsn",
    "adan",
    "adanorm",
    "adapnm",
    "adashift",
    "adasmooth",
    "adatam",
    "ademamix",
    "adopt",
    "aggmo",
    "aida",
    "alig",
    "alice",
    "amos",
    "ano",
    "apollo",
    "apollodqn",
    "asgd",
    "avagrad",
    "bcos",
    "bsam",
    "came",
    "conda",
    "dadaptadagrad",
    "dadaptadam",
    "dadaptadan",
    "dadaptlion",
    "dadaptsgd",
    "demo",
    "diffgrad",
    "distributedmuon",
    "dualadam",
    "emofact",
    "emolynx",
    "emonavi",
    "exadam",
    "fadam",
    "fira",
    "flashadamw",
    "focus",
    "fromage",
    "ftrl",
    "galore",
    "grams",
    "gravity",
    "grokfastadamw",
    "kate",
    "kron",
    "lamb",
    "laprop",
    "lars",
    "lbfgs",
    "lion",
    "lomo",
    "lorarite",
    "madgrad",
    "mars",
    "msvag",
    "muon",
    "nadam",
    "nero",
    "novograd",
    "padam",
    "pid",
    "pnm",
    "prodigy",
    "qhadam",
    "qhm",
    "racs",
    "radam",
    "ranger",
    "ranger21",
    "ranger25",
    "rmsprop",
    "rose",
    "scalableshampoo",
    "schedulefreeadamw",
    "schedulefreeradam",
    "schedulefreesgd",
    "scion",
    "scionlight",
    "sgd",
    "sgdp",
    "sgdsai",
    "sgdw",
    "shampoo",
    "signsgd",
    "simplifiedademamix",
    "sm3",
    "soap",
    "sophiah",
    "spam",
    "spectralsphere",
    "splus",
    "srmm",
    "stableadamw",
    "stablespam",
    "swats",
    "tam",
    "tiger",
    "vsgd",
    "yogi",
}))


PLUGIN_SPECIAL_HANDLING = {
    "adago": "Muon-family route; bridge splits tensor-rank param groups and injects use_muon=True/False.",
    "adamuon": "Muon-family route; bridge splits tensor-rank param groups and injects use_muon=True/False.",
    "muon": "Muon-family route; bridge splits tensor-rank param groups and injects use_muon=True/False.",
    "schedulefreeadamw": "Schedule-free-like route; trainer calls train() and uses ConstantLR instead of normal LR scheduler.",
    "schedulefreeradam": "Schedule-free-like route; trainer calls train() and uses ConstantLR instead of normal LR scheduler.",
    "schedulefreesgd": "Schedule-free-like route; trainer calls train() and uses ConstantLR instead of normal LR scheduler.",
    "sgdsai": "Bridge restores has_warmup after load_state_dict because upstream keeps it outside optimizer state.",
    "spam": "Bridge restores bool sparse masks after load_state_dict; default density=1.0 keeps LoRA tiny-route deterministic.",
    "spectralsphere": "Bridge routes 2D tensors to SpectralSphere and scalar/vector tensors to AdamW fallback.",
    "alice": "Bridge splits param groups by tensor shape and patches low-rank basis switching for small LoRA tensors.",
    "adammini": "Bridge provides a lightweight named-parameter module so upstream AdamMini keeps its model-aware grouping contract.",
    "adahessian": "Trainer marks create_graph backward before step so upstream Hutchinson Hessian state can be computed and resumed.",
    "alig": "Bridge wraps loss-only closure step; TrainingLoop binds the current loss before optimizer.step().",
    "demo": "Bridge uses identity gather when no torch.distributed default process group exists and persists upstream demo_state.",
    "adalomo": "Fused-backward route; bridge builds a lightweight named-parameter module and persists AdaLOMO factor states.",
    "lomo": "Fused-backward route; bridge builds a lightweight named-parameter module and TrainingLoop skips the normal optimizer.step().",
    "distributedmuon": "Muon-family distributed route; bridge splits use_muon groups and uses identity all_gather when no default process group exists.",
    "lbfgs": "Closure-required route; TrainingLoop binds a deterministic accumulated-batch recompute closure and defaults to max_iter=1.",
    "bsam": "Closure-required SAM route; TrainingLoop supplies the initial gradients plus deterministic recompute closures and defaults num_data=1.",
    "kron": "PSGD/Kron route; bridge preserves non-tensor einsum expressions and Python/NumPy RNG resume parity.",
}

PLUGIN_PENDING_OR_SPECIAL = {}


def plugin_resume_case(name: str) -> PluginOptimizerCase:
    canonical = str(name).strip().lower()
    return PluginOptimizerCase(
        name=canonical,
        extra_args=PLUGIN_DEFAULT_EXTRA_ARGS.get(canonical, ""),
        scheduler_expected="ConstantLR" if canonical in PLUGIN_SCHEDULE_FREE_OPTIMIZERS else "CosineAnnealingLR",
        include_vector_param=True,
        expected_param_group_flags=(True, False) if canonical in PLUGIN_MUON_FAMILY_OPTIMIZERS else (),
    )


def plugin_resume_cases(names: tuple[str, ...] | None = None) -> list[PluginOptimizerCase]:
    return [plugin_resume_case(name) for name in (names or PLUGIN_RESUME_SMOKE_PASSED)]


def canonical_plugin_resume_names(available_names: tuple[str, ...]) -> tuple[str, ...]:
    available = {str(name).strip().lower() for name in available_names}
    return tuple(name for name in PLUGIN_RESUME_SMOKE_PASSED if name in available)


def plugin_special_handling_for_available(available_names: tuple[str, ...]) -> dict[str, str]:
    available = {str(name).strip().lower() for name in available_names}
    notes: dict[str, str] = {}
    for source in (PLUGIN_SPECIAL_HANDLING, PLUGIN_PENDING_OR_SPECIAL):
        for name, note in source.items():
            if name in available:
                notes[name] = note
    return notes


def plugin_support_summary(available_names: tuple[str, ...]) -> dict[str, Any]:
    resume = canonical_plugin_resume_names(available_names)
    special = plugin_special_handling_for_available(available_names)
    available = {str(name).strip().lower() for name in available_names}
    return {
        "available_count": len(available),
        "resume_passed_count": len(resume),
        "pending_or_special_count": sum(1 for name in PLUGIN_PENDING_OR_SPECIAL if name in available),
        "resume_passed": list(resume),
        "pending_or_special": {name: note for name, note in PLUGIN_PENDING_OR_SPECIAL.items() if name in available},
        "special_handling": special,
    }

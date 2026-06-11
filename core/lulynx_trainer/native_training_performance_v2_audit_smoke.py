"""Smoke checks for the Native Training Performance Roadmap V2 audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from devtools.audit_native_training_performance_roadmap_v2 import build_v2_roadmap_audit  # noqa: E402


def run_smoke() -> dict[str, Any]:
    audit = build_v2_roadmap_audit(
        skip_first_round_cache_reader=True,
        include_post_p5_milestones=True,
        quick_post_p5_milestones=True,
    )
    assert audit["audit"] == "native_training_performance_roadmap_v2_audit_v0", audit
    assert audit["ok"] is True, audit
    required = audit["required_promotions"]
    expected = {
        "lora_mixed_precision_dispatch",
        "optimizer_multitensor_update",
        "native_data_prefetch_batch_path",
        "runtime_native_router_canary",
        "e2e_training_performance_gate",
    }
    assert set(required) == expected, required
    assert "sections" in audit and set(expected).issubset(audit["sections"]), audit
    milestones = audit["sections"]["post_p5_research_and_review_milestones"]
    milestone_gates = milestones["progress_gates"]
    expected_milestones = {
        "p6_research_routes",
        "p7_optimizer_kernel_expansion",
        "p8_adamw_variant_state_layout",
        "p8a_adamw_schedule_free_state_machine",
        "p8ad_kahan_adamw8bit_dispatch_review",
        "p8ae_paged_adamw8bit_dispatch_review",
        "p9_adaptive_lr_state_machine",
        "p10_factored_custom_state_layout",
        "p11_plugin_optimizer_selector",
        "p12_plugin_schedulefree_selected_optimizer",
        "p13_plugin_schedulefree_native_abi_sketch",
        "p14_plugin_schedulefree_checkpoint_adapter",
        "p15_plugin_schedulefree_training_tensor_binding",
        "p16_plugin_schedulefree_runtime_dispatch_shadow",
        "p17_plugin_schedulefree_e2e_shadow_matrix",
        "p18_plugin_schedulefree_canary_rollout_policy",
        "p19_plugin_schedulefree_dispatch_review",
        "p20_automagicpp_native_scratch_kernel",
        "p21_automagicpp_training_tensor_binding",
        "p22_automagicpp_runtime_dispatch_shadow",
        "p23_automagicpp_training_loop_canary",
        "p24_anima_factored_adamw_native_scratch_kernel",
        "p25_anima_factored_adamw_training_tensor_binding",
        "p26_anima_factored_adamw_runtime_dispatch_shadow",
        "p27_anima_factored_adamw_training_loop_canary",
        "p28_anima_factored_adamw_e2e_shadow_matrix",
        "p29_anima_factored_adamw_explicit_canary_rollout_policy",
        "p30_anima_factored_adamw_real_dispatch_review",
        "p31_automagicpp_e2e_shadow_matrix",
        "p32_automagicpp_explicit_canary_rollout_policy",
        "p33_automagicpp_real_dispatch_review",
        "p34_adafactor_native_scratch_kernel",
        "p35_adafactor_training_tensor_binding",
        "p36_adafactor_runtime_dispatch_shadow",
        "p37_adafactor_training_loop_canary",
        "p38_adafactor_e2e_shadow_matrix",
        "p39_adafactor_explicit_canary_rollout_policy",
        "p40_adafactor_real_dispatch_review",
        "p41_plugin_adamlike_selected_optimizer",
        "p42_plugin_adamw_training_loop_canary",
        "p43_plugin_adam_native_scratch_kernel",
        "p44_plugin_adam_training_tensor_binding",
        "p45_plugin_adam_runtime_dispatch_shadow",
        "p46_plugin_adam_training_loop_canary",
        "p47_plugin_adamax_native_scratch_kernel",
        "p48_plugin_adamax_training_tensor_binding",
        "p49_plugin_adamax_runtime_dispatch_shadow",
        "p50_plugin_adamax_training_loop_canary",
        "p51_plugin_adamc_native_scratch_kernel",
        "p52_plugin_adamc_training_tensor_binding",
        "p53_plugin_adamc_runtime_dispatch_shadow",
        "p54_plugin_adamc_training_loop_canary",
        "p55_plugin_adamg_native_scratch_kernel",
        "p56_plugin_adamg_training_tensor_binding",
        "p57_plugin_adamg_runtime_dispatch_shadow",
        "p58_plugin_adamg_training_loop_canary",
        "p59_plugin_adamod_native_scratch_kernel",
        "p60_plugin_adamod_training_tensor_binding",
        "p61_plugin_adamod_runtime_dispatch_shadow",
        "p62_plugin_adamod_training_loop_canary",
        "p63_plugin_adamp_native_scratch_kernel",
        "p64_plugin_adamp_training_tensor_binding",
        "p65_plugin_adamp_runtime_dispatch_shadow",
    }
    assert expected_milestones == set(milestone_gates), milestone_gates
    assert milestones["training_path_enabled"] is False, milestones
    assert milestones["default_behavior_changed"] is False, milestones
    assert audit["optimizer_native_full_coverage_completed"] is False, audit
    optimizer_full = audit["summary"]["optimizer_native_full_coverage"]
    assert optimizer_full["plugin_optimizer_count"] >= 100, optimizer_full
    assert optimizer_full["remaining_optimizer_count"] > 0, optimizer_full
    assert optimizer_full["scratch_kernel_ready_count"] >= 6, optimizer_full
    assert "adamax" in optimizer_full["scratch_kernel_ready_optimizer_names"], optimizer_full
    assert "adamc" in optimizer_full["scratch_kernel_ready_optimizer_names"], optimizer_full
    assert "adamg" in optimizer_full["scratch_kernel_ready_optimizer_names"], optimizer_full
    assert "adamod" in optimizer_full["scratch_kernel_ready_optimizer_names"], optimizer_full
    assert "adamp" in optimizer_full["scratch_kernel_ready_optimizer_names"], optimizer_full
    assert optimizer_full["training_tensor_binding_ready_count"] >= 6, optimizer_full
    assert optimizer_full["adamax_training_tensor_binding_ready"] is True, optimizer_full
    assert optimizer_full["runtime_dispatch_shadow_ready_count"] >= 6, optimizer_full
    assert optimizer_full["adamax_runtime_dispatch_shadow_ready"] is True, optimizer_full
    assert optimizer_full["training_loop_canary_ready_count"] >= 5, optimizer_full
    assert optimizer_full["adamax_training_loop_canary_ready"] is True, optimizer_full
    assert optimizer_full["adamc_native_scratch_kernel_ready"] is True, optimizer_full
    assert optimizer_full["adamc_training_tensor_binding_ready"] is True, optimizer_full
    assert optimizer_full["adamc_runtime_dispatch_shadow_ready"] is True, optimizer_full
    assert optimizer_full["adamc_training_loop_canary_ready"] is True, optimizer_full
    assert optimizer_full["adamg_native_scratch_kernel_ready"] is True, optimizer_full
    assert optimizer_full["adamg_training_tensor_binding_ready"] is True, optimizer_full
    assert optimizer_full["adamg_runtime_dispatch_shadow_ready"] is True, optimizer_full
    assert optimizer_full["adamg_training_loop_canary_ready"] is True, optimizer_full
    assert optimizer_full["adamod_native_scratch_kernel_ready"] is True, optimizer_full
    assert optimizer_full["adamod_training_tensor_binding_ready"] is True, optimizer_full
    assert optimizer_full["adamod_runtime_dispatch_shadow_ready"] is True, optimizer_full
    assert optimizer_full["adamod_training_loop_canary_ready"] is True, optimizer_full
    assert optimizer_full["adamp_native_scratch_kernel_ready"] is True, optimizer_full
    assert optimizer_full["adamp_training_tensor_binding_ready"] is True, optimizer_full
    assert optimizer_full["adamp_runtime_dispatch_shadow_ready"] is True, optimizer_full
    assert isinstance(audit["remaining_blockers"], list), audit
    return {
        "schema_version": 1,
        "probe": "native_training_performance_v2_audit_smoke",
        "ok": True,
        "roadmap_completed": audit["roadmap_completed"],
        "post_p5_milestones_completed": audit["post_p5_milestones_completed"],
        "required_promotions": required,
        "post_p5_progress_gates": milestone_gates,
        "recommended_next_step": audit["summary"]["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

# LoKr Clean-Room Compatibility Spec

This document defines the `lulynx-trainer` LoKr compatibility surface in
project-local terms only. It is intentionally limited to externally observable
behavior so the implementation can evolve without copying third-party source.

## Supported Weight Layouts

LoKr checkpoints are recognized from the following tensor layouts:

- `direct`
  - `lokr_w1`
  - `lokr_w2`
- `decomposed`
  - `lokr_w1_a` and `lokr_w1_b`
  - `lokr_w2_a` and `lokr_w2_b`
- mixed materialization is allowed internally during load resolution as long as
  the checkpoint still describes one LoKr branch.

The loader accepts both project-internal base keys such as
`unet_blocks_0_self_attn_q_proj` and exported keys prefixed as
`lora_unet_blocks_0_self_attn_q_proj`.

## Rank / Alpha / Factor Resolution

- Explicit `lokr_rank` wins when present.
- If `lokr_rank` is absent, rank is inferred from decomposed matrix shapes.
- If rank still cannot be inferred, the loader may recover it from scalar
  `alpha` when `alpha` is an integer-like positive value.
- Exported native checkpoints continue to omit `lokr_rank`.
- The `full_matrix` sentinel remains supported by the local implementation, but
  this spec only relies on observable behavior:
  - direct `w1` + direct `w2`
  - scaling is already represented in the exported tensors

## Export Modes

Two export modes are supported:

- `native`
  - emits direct `lokr_w1` + `lokr_w2`
  - omits `lokr_rank`
  - writes metadata describing native export semantics
- `lora_compatible`
  - emits only standard LoRA tensors
  - strips LoKr-only tensor keys
  - writes compatibility metadata

## Metadata Contract

The exporter owns the following metadata keys:

- `ss_lokr_export_mode`
- `ss_lokr_native_export`
- `ss_lokr_compatible_export`
- `ss_lokr_rank_exported`
- `ss_lokr_scale_export_format`

When applicable, the exporter also preserves or sets:

- `ss_network_module`
- `ss_anima_adapter_type`
- `ss_adapter_variant`

The current project-side training config that influences LoKr export/load is:

- `lycoris_lokr_factor`
- `lokr_rank_dropout`
- `lokr_module_dropout`
- `lokr_full_matrix`
- `lokr_decompose_both`
- `lokr_unbalanced_factorization`
- `lokr_export_mode`


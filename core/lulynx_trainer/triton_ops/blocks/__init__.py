"""Fused DiT-block Triton kernels for Lulynx training (default-off).

These kernels fuse the *bandwidth-bound* elementwise glue of an Anima DiT
block — AdaLN modulation, the FFN activation, and the gated residual — that
otherwise costs an extra kernel launch and an HBM round-trip of a full
activation each. Unlike the LoRA-path kernels (whose backward is dominated by
an unavoidable base GEMM), these ops are memory-bound, so fusing them can help
the forward *and* the backward.

Each module mirrors the ``triton_ops.lora`` convention:

* ``reference(...)`` — pure-PyTorch oracle used for validation.
* ``fused(...)``     — autograd-aware accelerated entry point (Triton forward
  + PyTorch backward), safe to drop into a training graph.

Modules import ``triton`` at load, so import them lazily.
"""

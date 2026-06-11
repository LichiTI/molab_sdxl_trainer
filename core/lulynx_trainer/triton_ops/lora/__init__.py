"""Fused LoRA-path Triton kernels for Lulynx training.

Each kernel module exposes:

* ``reference(...)`` — a pure-PyTorch implementation used for validation.
* ``fused(...)``     — the autograd-aware accelerated entry point. It runs
  the Triton forward kernel and a PyTorch backward, so it is safe to drop
  into a training graph.

Modules import ``triton`` at module load, so import them lazily (only when
the fused path is actually selected).
"""

"""Distributed Data Parallel (DDP) training support.

Provides:
- ``setup_ddp()`` / ``cleanup_ddp()``: process group lifecycle
- ``DistributedSampler`` wrapper for dataloaders
- ``DDPModelWrapper``: wraps model + optimizer + dataloader via Accelerate or raw DDP
- Rank-aware logging/checkpointing helpers

Design constraints
------------------
- Single-machine multi-GPU is the primary target.
- Multi-node works but requires external launch (torchrun / accelerate launch).
- Falls back to single-process when ``multi_gpu=False`` or when torch.distributed
  is unavailable.
- The trainer must call ``cleanup_ddp()`` in a finally block.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_ddp_initialized: bool = False


def _copy_lulynx_dataloader_runtime_attrs(source: Any, target: Any) -> Any:
    try:
        from .dataloader_rebuild_runtime import DATALOADER_REBUILD_RUNTIME_ATTRS
    except Exception:
        return target
    for attr_name in DATALOADER_REBUILD_RUNTIME_ATTRS:
        try:
            value = getattr(source, attr_name, None)
            if value:
                setattr(target, attr_name, value)
        except Exception:
            pass
    return target


def _dataloader_runtime_kwargs(dataloader: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": int(getattr(dataloader, "num_workers", 0) or 0),
        "collate_fn": getattr(dataloader, "collate_fn", None),
        "pin_memory": bool(getattr(dataloader, "pin_memory", False)),
        "drop_last": bool(getattr(dataloader, "drop_last", False)),
        "timeout": getattr(dataloader, "timeout", 0),
        "worker_init_fn": getattr(dataloader, "worker_init_fn", None),
        "multiprocessing_context": getattr(dataloader, "multiprocessing_context", None),
        "generator": getattr(dataloader, "generator", None),
        "persistent_workers": bool(getattr(dataloader, "persistent_workers", False)),
    }
    if kwargs["num_workers"] > 0:
        prefetch = getattr(dataloader, "prefetch_factor", None)
        if prefetch is not None:
            kwargs["prefetch_factor"] = prefetch
    pin_memory_device = getattr(dataloader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return kwargs


# ---------------------------------------------------------------------------
# Process group lifecycle
# ---------------------------------------------------------------------------

def setup_ddp(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    num_processes: int = 1,
    num_machines: int = 1,
    main_process_ip: str = "localhost",
    main_process_port: int = 29500,
) -> bool:
    """Initialize the default process group for DDP.

    Returns ``True`` if DDP was successfully initialised, ``False`` otherwise
    (e.g. single-GPU mode or distributed unavailable).

    When ``init_method`` is *None* and the standard ``RANK`` / ``WORLD_SIZE``
    environment variables are already set (as with ``torchrun``), those are
    used directly.  Otherwise a ``tcp://`` init method is constructed from
    *main_process_ip* and *main_process_port*.
    """
    global _ddp_initialized

    if not torch.cuda.is_available():
        logger.info("DDP setup skipped: CUDA not available")
        return False

    if num_processes <= 1 and num_machines <= 1:
        logger.info("DDP setup skipped: single-process mode (num_processes=%d, num_machines=%d)",
                     num_processes, num_machines)
        return False

    # If torchrun/accelerate already set env vars, respect them
    world_size_env = os.environ.get("WORLD_SIZE")
    rank_env = os.environ.get("RANK") or os.environ.get("LOCAL_RANK")

    if world_size_env is None and init_method is None:
        # Construct tcp init method for manual launch
        os.environ["MASTER_ADDR"] = main_process_ip
        os.environ["MASTER_PORT"] = str(main_process_port)
        if num_machines > 1:
            os.environ.setdefault("WORLD_SIZE", str(num_processes * num_machines))
        else:
            os.environ.setdefault("WORLD_SIZE", str(num_processes))

    try:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
            )
        _ddp_initialized = True
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        logger.info("DDP initialised: rank=%d, world_size=%d, backend=%s", rank, world_size, backend)
        return True
    except Exception as exc:
        logger.warning("DDP init failed (%s); falling back to single-process", exc)
        return False


def cleanup_ddp() -> None:
    """Destroy the default process group if it was initialised by us."""
    global _ddp_initialized
    if _ddp_initialized and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        _ddp_initialized = False


def is_ddp_active() -> bool:
    """Return ``True`` if DDP is currently active."""
    return _ddp_initialized and torch.distributed.is_initialized()


def get_rank() -> int:
    """Return the current process rank (0 in single-process mode)."""
    if is_ddp_active():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of DDP processes (1 in single-process mode)."""
    if is_ddp_active():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return ``True`` on rank 0 (or in single-process mode)."""
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Distributed sampler helper
# ---------------------------------------------------------------------------

def make_distributed_sampler(
    dataset: Any,
    shuffle: bool = True,
    seed: int = 42,
) -> Optional[torch.utils.data.distributed.DistributedSampler]:
    """Create a ``DistributedSampler`` when DDP is active, else ``None``."""
    if not is_ddp_active():
        return None
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )


def wrap_dataloader_for_ddp(
    dataloader: torch.utils.data.DataLoader,
    dataset: Any,
    shuffle: bool = True,
    seed: int = 42,
) -> torch.utils.data.DataLoader:
    """Replace a dataloader's sampler with a DistributedSampler when DDP is active.

    Returns the original dataloader unchanged if DDP is not active.
    """
    sampler = make_distributed_sampler(dataset, shuffle=shuffle, seed=seed)
    if sampler is None:
        return dataloader

    wrapped = torch.utils.data.DataLoader(
        dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        **_dataloader_runtime_kwargs(dataloader),
    )
    return _copy_lulynx_dataloader_runtime_attrs(dataloader, wrapped)


# ---------------------------------------------------------------------------
# DDP model wrapper
# ---------------------------------------------------------------------------

class DDPModelWrapper:
    """Wraps model, optimizer, and dataloader for DDP training.

    On construction:
    - If DDP is active, wraps the model in ``torch.nn.parallel.DistributedDataParallel``.
    - Replaces the dataloader sampler with a DistributedSampler.
    - Provides ``backward()``, ``clip_grad_norm()``, and rank-gated helpers.

    If DDP is *not* active, all methods are no-ops / pass-throughs.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        dataloader: Optional[torch.utils.data.DataLoader] = None,
        dataset: Optional[Any] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
        device: Optional[torch.device] = None,
    ):
        self._raw_model = model
        self._optimizer = optimizer
        self._ddp_model: Optional[torch.nn.parallel.DistributedDataParallel] = None
        self._ddp_sampler: Optional[torch.utils.data.distributed.DistributedSampler] = None
        self._dataloader = dataloader

        if is_ddp_active():
            local_rank = get_rank() % torch.cuda.device_count()
            device = device or torch.device(f"cuda:{local_rank}")
            model = model.to(device)

            self._ddp_model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=gradient_as_bucket_view,
                static_graph=static_graph,
            )
            logger.info("Model wrapped in DDP (rank=%d, device=%s)", get_rank(), device)

            # Wrap dataloader sampler
            if dataloader is not None and dataset is not None:
                self._ddp_sampler = make_distributed_sampler(dataset, shuffle=True)
                self._dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=dataloader.batch_size,
                    sampler=self._ddp_sampler,
                    **_dataloader_runtime_kwargs(dataloader),
                )
                self._dataloader = _copy_lulynx_dataloader_runtime_attrs(dataloader, self._dataloader)
        else:
            self._dataloader = dataloader

    # -- model access --------------------------------------------------------

    @property
    def model(self) -> torch.nn.Module:
        """The (possibly DDP-wrapped) model."""
        return self._ddp_model if self._ddp_model is not None else self._raw_model

    @property
    def raw_model(self) -> torch.nn.Module:
        """The unwrapped model (for saving, inspection, etc.)."""
        if self._ddp_model is not None:
            return self._ddp_model.module
        return self._raw_model

    @property
    def dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """The (possibly DDP-wrapped) dataloader."""
        return self._dataloader

    @property
    def sampler(self) -> Optional[torch.utils.data.distributed.DistributedSampler]:
        """The DistributedSampler (None if not using DDP)."""
        return self._ddp_sampler

    # -- training helpers ----------------------------------------------------

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass — DDP synchronises gradients automatically."""
        loss.backward()

    def clip_grad_norm(self, max_norm: float) -> None:
        """Clip gradient norm on the raw model's parameters."""
        torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm)

    def set_epoch(self, epoch: int) -> None:
        """Inform the DistributedSampler of the current epoch for shuffling."""
        if self._ddp_sampler is not None:
            self._ddp_sampler.set_epoch(epoch)

    def all_reduce(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """All-reduce a tensor across DDP ranks. No-op in single-process mode."""
        if not is_ddp_active():
            return tensor
        if op == "mean":
            dist_op = torch.distributed.ReduceOp.SUM
            torch.distributed.all_reduce(tensor, op=dist_op)
            tensor /= get_world_size()
        elif op == "sum":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        elif op == "max":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        elif op == "min":
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        else:
            raise ValueError(f"Unsupported all_reduce op: {op!r}")
        return tensor

    # -- rank-gated helpers --------------------------------------------------

    def save_on_main(self, save_fn: callable, *args, **kwargs) -> None:
        """Call *save_fn* only on the main process."""
        if is_main_process():
            save_fn(*args, **kwargs)

    def log_on_main(self, log_fn: callable, *args, **kwargs) -> None:
        """Call *log_fn* only on the main process."""
        if is_main_process():
            log_fn(*args, **kwargs)

    # -- barrier / sync -------------------------------------------------------

    def wait_for_everyone(self) -> None:
        """Barrier — all processes wait here."""
        if is_ddp_active():
            torch.distributed.barrier()

"""
SOTA Trainer Swap Helpers.

This module uses the `optimizer_sync` component to provide true training
continuity across weight updates.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from .optimizer_sync import sync_optimizer_state


@contextlib.contextmanager
def swap_barrier() -> Iterable[None]:
    """Ensures all ranks in a distributed group are synchronized."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    try:
        yield
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def sota_in_place_swap(
    model: torch.nn.Module,
    optimizer: Optimizer,
    staged_tensors: dict[str, torch.Tensor],
    plan: dict[str, Any],
    name_map: dict[str, str],
) -> None:
    """
    Performs a SOTA in-place swap, updating both weights and optimizer state.

    Args:
        model: The model to update.
        optimizer: The optimizer to synchronize.
        staged_tensors: A dict mapping tensor keys to new tensor data,
                        presumably from a CudaIPCAgent.
        plan: The hotweights replication plan.
        name_map: The mapping from tensor keys to model parameter names.
    """
    print("Starting SOTA in-place swap for training...")

    # 1. Update model weights in-place
    with torch.no_grad(), swap_barrier():
        for key, target_name in name_map.items():
            if key in staged_tensors:
                new_tensor = staged_tensors[key]
                try:
                    param = model.get_parameter(target_name)
                    assert param.shape == new_tensor.shape
                    param.data.copy_(new_tensor, non_blocking=True)
                except Exception as e:
                    print(f"Warning: Could not swap param {target_name}: {e}")
        torch.cuda.synchronize()

    # 2. Synchronize the optimizer state using the SOTA module
    with swap_barrier():
        report = sync_optimizer_state(optimizer, model, plan, name_map)
        print(f"Optimizer sync report: {report}")
        torch.cuda.synchronize()

    print("SOTA in-place swap complete.")

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

    # 0. Pre-validate every target before touching any weight: a partial swap
    #    followed by optimizer sync is silent training corruption, so any
    #    unresolvable parameter or shape mismatch aborts the whole swap.
    planned: list[tuple[str, object, object]] = []
    failures: list[str] = []
    for key, target_name in name_map.items():
        if key not in staged_tensors:
            continue
        new_tensor = staged_tensors[key]
        try:
            param = model.get_parameter(target_name)
        except Exception as e:
            failures.append(f"{target_name}: resolve failed: {e}")
            continue
        if param.shape != new_tensor.shape:
            failures.append(
                f"{target_name}: shape {tuple(param.shape)} != "
                f"staged {tuple(new_tensor.shape)}"
            )
            continue
        planned.append((target_name, param, new_tensor))
    if failures:
        raise RuntimeError(
            "sota_in_place_swap aborted before mutating any weight; "
            + "; ".join(failures)
        )

    # 1. Update model weights in-place
    with torch.no_grad(), swap_barrier():
        for _, param, new_tensor in planned:
            param.data.copy_(new_tensor, non_blocking=True)
        torch.cuda.synchronize()

    # 2. Synchronize the optimizer state using the SOTA module
    with swap_barrier():
        report = sync_optimizer_state(optimizer, model, plan, name_map)
        print(f"Optimizer sync report: {report}")
        torch.cuda.synchronize()

    print("SOTA in-place swap complete.")

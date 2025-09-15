"""NCCL-based broadcast transport.

Uses torch.distributed with NCCL backend to broadcast bucket buffers.
If CUDA or torch.distributed is unavailable, falls back to a no-op path
compatible with tests (local-only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, List

import numpy as np

try:  # optional torch
    import torch
    import torch.distributed as dist
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    dist = None  # type: ignore

from .base import Transport
from ..utils.dist import ensure_pg, get_subgroup


Bucket = Tuple[int, np.ndarray]


def _maybe_init_dist() -> bool:
    ok, used = ensure_pg("nccl")
    return bool(ok)


@dataclass
class NCCLReplicator(Transport):
    chunk_bytes: int = 16 << 20  # 16 MiB

    def __post_init__(self) -> None:
        self._ok = _maybe_init_dist()
        try:
            self._rank = dist.get_rank() if (self._ok and dist.is_initialized()) else 0
        except Exception:
            self._rank = 0

    @property
    def world_rank(self) -> int:
        return getattr(self, "_rank", 0)

    def replicate(self, bucket_iter: Iterable[Bucket]) -> None:
        if not self._ok:
            # No-op fallback
            for _bid, _buf in bucket_iter:
                pass
            return
        for _bid, buf in bucket_iter:
            self._broadcast_into_numpy(buf)

    def replicate_stream(
        self,
        bucket_iter: Iterable[Tuple[int, np.ndarray] | Tuple[int, np.ndarray, List[int]]],
        on_complete: Callable[[int, np.ndarray], None],
    ) -> None:
        if not self._ok:
            for item in bucket_iter:
                bid, arr = item[0], item[1]
                on_complete(int(bid), arr)
            return

        # Optional subgroup caching
        group_cache: dict[frozenset[int], object] = {}

        def get_group(consumers: Optional[List[int]]):
            if not consumers:
                return dist.group.WORLD  # type: ignore[attr-defined]
            g = get_subgroup(consumers, group_cache)
            return g if g is not None else dist.group.WORLD  # type: ignore[attr-defined]

        for item in bucket_iter:
            if len(item) == 3:
                bid, arr, consumers = item  # type: ignore[misc]
            else:
                bid, arr = item  # type: ignore[misc]
                consumers = None
            # Skip non-consumers
            if consumers is not None and self.world_rank not in consumers:
                continue
            self._broadcast_into_numpy(arr, consumers=consumers)
            on_complete(int(bid), arr)

    # --- helpers ---
    def _broadcast_into_numpy(self, out: np.ndarray, consumers: Optional[List[int]] = None) -> None:
        """Broadcast the given numpy array into itself using NCCL via a GPU tensor.

        Root is the minimum rank in `consumers` or 0 if None.
        On root: copies CPU -> GPU, broadcasts, returns without modifying out.
        On non-root: broadcasts into GPU tensor and copies GPU -> CPU (into out).
        """
        if torch is None or dist is None or not dist.is_initialized():
            return
        rank = self.world_rank
        root = 0 if not consumers else min(int(x) for x in consumers)
        # Determine group for broadcast
        group = None
        try:
            if consumers is not None:
                # Try to fetch an existing group by ranks; not strictly necessary
                ranks = sorted(int(x) for x in consumers)
                group = dist.new_group(ranks=ranks)
        except Exception:
            group = None
        # Prepare device tensor
        device = torch.device("cuda", torch.cuda.current_device())
        n = int(out.nbytes)
        if rank == root:
            src = torch.from_numpy(out).to(device)
            if self.chunk_bytes and n > self.chunk_bytes:
                off = 0
                while off < n:
                    end = min(off + self.chunk_bytes, n)
                    view = src.view(torch.uint8)[off:end]
                    dist.broadcast(view, src=root, group=group)
                    off = end
            else:
                dist.broadcast(src, src=root, group=group)
        else:
            dst = torch.empty(n, dtype=torch.uint8, device=device)
            if self.chunk_bytes and n > self.chunk_bytes:
                off = 0
                while off < n:
                    end = min(off + self.chunk_bytes, n)
                    view = dst[off:end]
                    dist.broadcast(view, src=root, group=group)
                    off = end
            else:
                dist.broadcast(dst, src=root, group=group)
            # Copy back to CPU numpy
            out_view = torch.from_numpy(out)
            out_view.view(torch.uint8).copy_(dst, non_blocking=False)

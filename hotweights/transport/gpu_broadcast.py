"""Device broadcast transport for AMD/Intel/NVIDIA without CUDA IPC.

Broadcasts contiguous bucket buffers directly as device tensors using
torch.distributed backends:
- CUDA/ROCm: backend="nccl" (RCCL on ROCm)
- Intel XPU: backend="ccl" (oneCCL via oneccl_bindings_for_pytorch)

Root rank assembles a contiguous CPU buffer, performs an async H2D copy
into a device tensor, broadcasts it, and then all consumers scatter the
device buffer into per-tensor device allocations via a GenericGpuAgent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os

import numpy as np

try:  # optional torch
    import torch
    import torch.distributed as dist  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    dist = None  # type: ignore

from ..core.replicate import assemble_bucket as _assemble_bucket
from ..telemetry.logging import get_logger
import threading
import time
from ..telemetry.prom import Counter, Histogram
from ..utils.env import env_bool, env_int, env_list_int
from ..utils.dist import ensure_pg, get_subgroup


@dataclass
class GpuBroadcastTransport:
    agent: Any
    chunk_bytes_mb: int = 32
    backend: Optional[str] = None  # 'nccl'|'ccl'|None (auto)

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for GpuBroadcastTransport")
        self._log = get_logger("GpuBroadcast", {})
        # Allow env override for broadcast chunk size
        try:
            mb_env = int(os.getenv("HOTWEIGHTS_BCAST_CHUNK_MB", str(self.chunk_bytes_mb)))
        except Exception:
            mb_env = self.chunk_bytes_mb
        self._chunk_bytes = max(1, int(mb_env)) * (1 << 20)
        # Rank/world discovery
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # Group cache by frozenset of ranks
        self._groups: dict[frozenset[int], object] = {}
        # Ensure process group
        # Ensure or attempt to initialize the process group (best-effort)
        ok, used = ensure_pg(self.backend)
        if ok:
            try:
                self._log.info(f"initialized process group backend={used}")
            except Exception:
                pass
        try:
            self._log.info(f"bcast_chunk={int(self._chunk_bytes/(1<<20))}MiB")
        except Exception:
            pass
        # Optional microbenchmark autotune for chunk size
        try:
            if env_bool("HOTWEIGHTS_BCAST_AUTOTUNE", False):
                self._autotune_broadcast()
        except Exception:
            pass
        # Metrics
        self._buckets_c = Counter("hotweights_bcast_buckets_total", "GPU broadcast buckets completed")
        self._bytes_c = Counter("hotweights_bcast_bytes_total", "GPU broadcast bytes transferred")
        self._bcast_h = Histogram("hotweights_bcast_seconds", "GPU broadcast time per bucket")
        self._scatter_h = Histogram("hotweights_bcast_scatter_seconds", "GPU device scatter time per bucket")

    # --- init helpers ---
    def _ensure_pg(self) -> None:
        # Retained for backward compatibility; delegate to ensure_pg
        ensure_pg(self.backend)

    def _infer_backend(self) -> str:
        # Prefer nccl when CUDA/ROCm is available, else ccl for Intel XPU
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                return "nccl"
        except Exception:
            pass
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
                return "ccl"
        except Exception:
            pass
        # Fallback
        return "nccl"

    def _get_group(self, consumers: Optional[List[int]]) -> object | None:
        if dist is None or (hasattr(dist, "is_initialized") and not dist.is_initialized()):
            return None
        if not consumers:
            return None  # WORLD
        key = frozenset(int(x) for x in consumers)
        g = self._groups.get(key)
        if g is None:
            try:
                g = dist.new_group(ranks=sorted(key))
                self._groups[key] = g
            except Exception:
                g = None
        return g

    # --- main API ---
    def _prepare_dev_on_root(self, b: Dict[str, Any]) -> "torch.Tensor":  # noqa: ANN001
        size = int(b.get("size", 0))
        dev = torch.empty(size, dtype=torch.uint8, device=self.agent.device)
        if size <= 0:
            return dev
        cpu = _assemble_bucket(b["items"]) if size > 0 else np.empty(0, dtype=np.uint8)
        try:
            pinned = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            pinned.view(torch.uint8)[: size].copy_(torch.from_numpy(cpu))
            dev.copy_(pinned.to(dev.device, non_blocking=True))
        except Exception:
            dev.copy_(torch.from_numpy(cpu).to(dev.device, non_blocking=True))
        return dev

    def replicate(self, plan: Dict[str, Any]) -> None:
        # Single process fast-path
        if self.world_size <= 1 or dist is None or (hasattr(dist, "is_initialized") and not dist.is_initialized()):
            for b in plan.get("buckets", []):
                size = int(b.get("size", 0))
                # Root assembles and keeps device buffer locally
                cpu = _assemble_bucket(b["items"]) if size > 0 else np.empty(0, dtype=np.uint8)
                dev = torch.empty(size, dtype=torch.uint8, device=self.agent.device)
                dev.copy_(torch.from_numpy(cpu).to(dev.device, non_blocking=True))
                self.agent.share_from_gpu(int(b["bucket_id"]), dev)
                self.agent.scatter_from_shared(int(b["bucket_id"]), b["items"])
            return

        prefetch_ok = env_bool("HOTWEIGHTS_BCAST_PREFETCH", True)
        buckets = list(plan.get("buckets", []))
        next_thread: Optional[threading.Thread] = None
        next_ready: dict[str, Any] = {"dev": None}

        for i, b in enumerate(buckets):
            bucket_id = int(b["bucket_id"])
            size = int(b.get("size", 0))
            consumers = b.get("consumer_ranks")
            # If consumers set and this rank is not a consumer, skip
            if consumers is not None and self.rank not in [int(x) for x in consumers]:
                continue
            group = get_subgroup(consumers, self._groups)
            world_root = min(consumers) if consumers else 0
            # src index: world rank for WORLD, else index within group
            if group is None:
                src_rank = world_root
            else:
                src_rank = sorted(int(x) for x in (consumers or [0])).index(world_root)
            # Build device tensor
            is_root = (self.rank == world_root)
            if is_root and next_thread is not None:
                # Ensure previous prefetch completed
                try:
                    next_thread.join()
                except Exception:
                    pass
            if is_root and next_ready["dev"] is not None:
                dev = next_ready["dev"]  # type: ignore[assignment]
                next_ready["dev"] = None
            else:
                dev = torch.empty(size, dtype=torch.uint8, device=self.agent.device)
                if is_root:
                    # Prepare synchronously when no prefetch available
                    dev = self._prepare_dev_on_root(b)
            # Broadcast device buffer (chunked)
            t_b0 = time.perf_counter()
            if size > 0:
                if self._chunk_bytes and size > self._chunk_bytes:
                    off = 0
                    while off < size:
                        end = min(off + self._chunk_bytes, size)
                        view = dev.view(torch.uint8)[off:end]
                        dist.broadcast(view, src=src_rank, group=group)
                        off = end
                else:
                    dist.broadcast(dev, src=src_rank, group=group)
            t_b1 = time.perf_counter()
            # Share and scatter locally
            self.agent.share_from_gpu(bucket_id, dev)
            t_s0 = time.perf_counter()
            self.agent.scatter_from_shared(bucket_id, b["items"])
            t_s1 = time.perf_counter()
            # Metrics
            try:
                self._buckets_c.inc(1.0)
                self._bytes_c.inc(float(size))
                self._bcast_h.observe(max(0.0, t_b1 - t_b0))
                self._scatter_h.observe(max(0.0, t_s1 - t_s0))
            except Exception:
                pass
            # Kick off prefetch for next bucket on root
            if is_root and prefetch_ok and (i + 1) < len(buckets):
                b_next = buckets[i + 1]
                # If next bucket doesn't include this rank, skip prefetch
                cons_next = b_next.get("consumer_ranks")
                if cons_next is None or self.rank in [int(x) for x in cons_next]:
                    def _do_prefetch():  # noqa: ANN202
                        try:
                            next_ready["dev"] = self._prepare_dev_on_root(b_next)
                        except Exception:
                            next_ready["dev"] = None
                    next_thread = threading.Thread(target=_do_prefetch, daemon=True)
                    try:
                        next_thread.start()
                    except Exception:
                        next_thread = None
        # Ensure any background prefetch is finished
        if next_thread is not None:
            try:
                next_thread.join()
            except Exception:
                pass

    # --- autotune helpers ---
    def _autotune_broadcast(self) -> None:
        if dist is None or not hasattr(dist, "is_initialized") or not dist.is_initialized() or self.world_size <= 1:
            return
        # Measure world broadcast performance for various chunk sizes
        size_mb = 128
        size_mb = max(16, env_int("HOTWEIGHTS_BCAST_AUTOTUNE_MB", 128))
        total = size_mb * (1 << 20)
        dev = torch.empty(total, dtype=torch.uint8, device=self.agent.device)
        # Candidates
        chunks_mb = env_list_int("HOTWEIGHTS_BCAST_AUTOTUNE_CHUNKS_MB", [8, 16, 32, 64])
        # Root rank for WORLD
        root = 0
        best = (0.0, self._chunk_bytes)
        for mb in chunks_mb:
            chk = max(1, mb) * (1 << 20)
            # Warmup
            self._bcast_with_chunk(dev, chk, root)
            dist.barrier()
            t0 = time.perf_counter()
            self._bcast_with_chunk(dev, chk, root)
            dist.barrier()
            dt = time.perf_counter() - t0
            bw = (total / dt) / (1 << 30) if dt > 0 else 0.0
            if bw > best[0]:
                best = (bw, chk)
        _, best_chk = best
        self._chunk_bytes = int(best_chk)
        try:
            self._log.info(f"autotune.bcast_chunk={int(self._chunk_bytes/(1<<20))}MiB")
        except Exception:
            pass

    def _bcast_with_chunk(self, dev: "torch.Tensor", chunk: int, root: int) -> None:
        n = int(dev.numel())
        if chunk and n > chunk:
            off = 0
            while off < n:
                end = min(off + chunk, n)
                view = dev.view(torch.uint8)[off:end]
                dist.broadcast(view, src=root)
                off = end
        else:
            dist.broadcast(dev, src=root)

    def cleanup(self) -> None:
        try:
            self.agent.cleanup()
        except Exception:
            pass
        # Free groups
        try:
            for g in self._groups.values():
                try:
                    g.barrier()  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

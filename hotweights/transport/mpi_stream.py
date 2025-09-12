"""MPI-based replication (skeleton).

Uses mpi4py when available; otherwise provides a no-op fallback that simply
iterates the buckets on rank 0 to exercise interfaces in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple, Dict, FrozenSet, List

import numpy as np

try:  # optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    MPI = None  # type: ignore

from ..telemetry.prom import Counter, Gauge
from ..telemetry.logging import get_logger

Bucket = Tuple[int, np.ndarray]


@dataclass
class MPIReplicator:
    bucket_bytes: int = 512 << 20
    window: int = 2
    group_ranks: Optional[list[int]] = None
    chunk_bytes: int = 0  # 0 means no chunking

    def __post_init__(self) -> None:
        self._mpi = MPI
        self._comms_cache: Dict[FrozenSet[int], object] = {}
        if self._mpi is not None:
            self._base_comm = self._mpi.COMM_WORLD
            self._world_rank = self._base_comm.Get_rank()
            if self.group_ranks is not None:
                # Default communicator for fixed subgroup use-cases
                try:
                    world_group = self._base_comm.Get_group()
                    sub_group = world_group.Incl(self.group_ranks)
                    if self._world_rank in self.group_ranks:
                        self._comm = self._base_comm.Create(sub_group)
                    else:
                        self._comm = None
                except Exception:
                    self._comm = self._base_comm
            else:
                self._comm = self._base_comm
        else:
            self._base_comm = None  # type: ignore[assignment]
            self._world_rank = 0
            self._comm = None
        # Metrics and logger
        self._log = get_logger("MPIReplicator", {"rank": self.world_rank})
        self._buckets = Counter("hotweights_mpi_buckets_bcast_total", "MPI buckets broadcasted")
        self._bytes = Counter("hotweights_mpi_bytes_total", "MPI bytes transferred")

    @property
    def rank(self) -> int:
        # Backcompat: world rank (used by callers for root decisions)
        return getattr(self, "_world_rank", 0)

    @property
    def world_rank(self) -> int:
        return getattr(self, "_world_rank", 0)

    def replicate(self, bucket_iter: Iterable[Bucket]) -> None:
        if self._mpi is None or self._comm is None:  # no-op fallback
            for _bid, _buf in bucket_iter:
                pass
            return

        inflight = []
        for bucket_id, buf in bucket_iter:
            assert isinstance(buf, np.ndarray) and buf.flags["C_CONTIGUOUS"]
            if self.chunk_bytes and buf.nbytes > self.chunk_bytes:
                off = 0
                while off < buf.nbytes:
                    end = min(off + self.chunk_bytes, buf.nbytes)
                    view = buf[off:end]
                    req = self._comm.Ibcast(view, root=0)
                    req.Wait()
                    off = end
                continue
            req = self._comm.Ibcast(buf, root=0)
            inflight.append((bucket_id, req))
            if len(inflight) >= self.window:
                self._drain(inflight.pop(0))
        for item in inflight:
            self._drain(item)

    def _drain(self, item) -> None:
        _bucket_id, req = item
        req.Wait()
        try:
            self._buckets.inc(1.0)
            # Note: buf size unknown here; rely on caller context if needed
        except Exception:
            pass

    # Streaming API: call on_complete as each broadcast finishes to enable overlap
    def replicate_stream(self, bucket_iter: Iterable[Tuple[int, np.ndarray] | Tuple[int, np.ndarray, List[int]]], on_complete: Callable[[int, np.ndarray], None]) -> None:
        if self._mpi is None or self._base_comm is None:
            for item in bucket_iter:
                if isinstance(item, tuple) and len(item) >= 2:
                    bucket_id, buf = item[0], item[1]
                    on_complete(bucket_id, buf)
            return

        inflight: list[tuple[int, object, np.ndarray]] = []

        def get_comm_and_root(consumers: Optional[List[int]]):
            if not consumers:
                return self._comm or self._base_comm, 0  # type: ignore[return-value]
            # Cache communicator by frozenset of world ranks
            key = frozenset(int(x) for x in consumers)
            comm = self._comms_cache.get(key)
            if comm is None:
                world_group = self._base_comm.Get_group()
                sub_group = world_group.Incl(sorted(key))
                comm = self._base_comm.Create(sub_group)
                self._comms_cache[key] = comm
            # Define root as min world rank in group, mapped to comm rank
            root_world = min(consumers)
            # Map world rank to comm rank by finding index in sorted list
            comm_root = sorted(key).index(root_world)
            return comm, comm_root

        for item in bucket_iter:
            if isinstance(item, tuple) and len(item) == 3:
                bucket_id, buf, consumers = item  # type: ignore[misc]
            else:
                bucket_id, buf = item  # type: ignore[misc]
                consumers = None
            if consumers is not None and self._world_rank not in consumers:
                # Not a consumer for this bucket on this rank; skip
                continue
            assert isinstance(buf, np.ndarray) and buf.flags["C_CONTIGUOUS"]
            comm, root = get_comm_and_root(consumers)
            if self.chunk_bytes and buf.nbytes > self.chunk_bytes:
                off = 0
                while off < buf.nbytes:
                    end = min(off + self.chunk_bytes, buf.nbytes)
                    view = buf[off:end]
                    req = comm.Ibcast(view, root=root)
                    req.Wait()
                    try:
                        self._bytes.inc(float(view.nbytes))
                    except Exception:
                        pass
                    off = end
                on_complete(bucket_id, buf)
                continue
            req = comm.Ibcast(buf, root=root)
            inflight.append((bucket_id, req, buf))
            if len(inflight) >= self.window:
                b, r, buf0 = inflight.pop(0)
                r.Wait()
                on_complete(b, buf0)
        for b, r, buf0 in inflight:
            r.Wait()
            on_complete(b, buf0)

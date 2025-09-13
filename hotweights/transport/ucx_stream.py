"""UCX transport for bucket broadcast (best-effort).

Requires ucx-py. Uses a simple root-broadcast model based on environment:
  - WORLD_SIZE (int), RANK (int), MASTER_ADDR (str), MASTER_PORT (int)
Root (rank 0) listens; others connect. For each (bucket_id, buffer) pair,
root sends the buffer to all peers; others receive into the provided buffer.

This implementation is intentionally simple and synchronous per bucket. It can
be extended with pipelining and multiple lanes if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Tuple, List, Optional
import os
import asyncio

import numpy as np

try:  # optional
    import ucp  # type: ignore
except Exception:  # pragma: no cover
    ucp = None  # type: ignore

from ..telemetry.prom import Counter, Gauge
from .base import Transport
from ..telemetry.logging import get_logger


Bucket = Tuple[int, np.ndarray]


@dataclass
class UCXReplicator(Transport):
    window: int = 2
    chunk_bytes: int = 8 << 20  # 8 MiB chunks
    send_concurrency: int = 0  # 0 -> fanout to all peers per chunk
    send_retries: int = 0
    retry_delay_ms: int = 10

    def __post_init__(self) -> None:  # noqa: D401
        self._ucp = ucp
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        self.master_port = int(os.getenv("MASTER_PORT", "19999"))
        # NIC binding (mirror NCCL_IB_HCA semantics if present)
        try:
            hca = os.getenv("HOTWEIGHTS_UCX_NET_DEVICES") or os.getenv("NCCL_IB_HCA")
            if hca:
                os.environ.setdefault("UCX_NET_DEVICES", hca)
        except Exception:
            pass
        # env-tunable chunk size
        try:
            mb = int(os.getenv("HOTWEIGHTS_UCX_CHUNK_MB", str(self.chunk_bytes >> 20)))
            self.chunk_bytes = max(1, mb) << 20
        except Exception:
            pass
        # env-tunable send concurrency per chunk
        try:
            conc = int(os.getenv("HOTWEIGHTS_UCX_CONCURRENCY", "0"))
            self.send_concurrency = max(0, conc)
        except Exception:
            self.send_concurrency = 0
        # Autotune basic concurrency if not set
        if self.send_concurrency == 0:
            try:
                # Favor small fanout to reduce head-of-line blocking
                if self.world_size >= 8:
                    self.send_concurrency = 4
                elif self.world_size >= 4:
                    self.send_concurrency = 2
            except Exception:
                pass
        # inflight limit (MB) maps to max peers per chunk based on chunk size
        try:
            inflight_mb = int(os.getenv("HOTWEIGHTS_UCX_INFLIGHT_LIMIT_MB", "0"))
        except Exception:
            inflight_mb = 0
        if inflight_mb > 0:
            limit_bytes = max(1, inflight_mb) << 20
            max_peers = max(1, limit_bytes // self.chunk_bytes)
            if self.send_concurrency == 0 or self.send_concurrency > max_peers:
                self.send_concurrency = int(max_peers)
        # retries
        try:
            self.send_retries = max(0, int(os.getenv("HOTWEIGHTS_UCX_RETRIES", "0")))
        except Exception:
            self.send_retries = 0
        try:
            self.retry_delay_ms = max(0, int(os.getenv("HOTWEIGHTS_UCX_RETRY_DELAY_MS", "10")))
        except Exception:
            self.retry_delay_ms = 10
        # Simple autotune: adjust chunk size for larger worlds if requested
        try:
            if os.getenv("HOTWEIGHTS_UCX_AUTOTUNE", "0") in ("1", "true", "True"):
                if self.world_size >= 16 and (self.chunk_bytes >> 20) > 4:
                    self.chunk_bytes = 4 << 20
        except Exception:
            pass
        # Metrics and logging
        self._log = get_logger("UCXReplicator", {"rank": self.rank, "ws": self.world_size})
        self._retries = Counter("hotweights_ucx_send_retries_total", "UCX send retries")
        self._bytes_sent = Counter("hotweights_ucx_bytes_sent_total", "UCX bytes sent")
        self._bytes_recv = Counter("hotweights_ucx_bytes_received_total", "UCX bytes received")
        self._retry_delay_g = Gauge("hotweights_ucx_retry_delay_ms", "UCX retry delay ms")
        self._last_err = Gauge("hotweights_ucx_last_error", "1 if last operation error")
        self._last_err_ts = Gauge("hotweights_ucx_last_error_ts", "Unix timestamp of last error")
        try:
            self._retry_delay_g.set(float(self.retry_delay_ms))
            self._last_err.set(0.0)
            self._last_err_ts.set(0.0)
        except Exception:
            pass

    def replicate(self, bucket_iter: Iterable[Bucket]) -> None:
        if self._ucp is None or self.world_size <= 1:
            # single process or no UCX: nothing to do
            for _bid, _buf in bucket_iter:
                pass
            return
        asyncio.run(self._run(bucket_iter))

    async def _run(self, bucket_iter: Iterable[Bucket]) -> None:
        if self.rank == 0:
            listener = await ucp.create_listener(self._on_connect, port=self.master_port)
            # Wait for all peers to connect
            while len(self._peers) < self.world_size - 1:
                await asyncio.sleep(0.01)
            # Broadcast buckets
            for bucket_id, buf in bucket_iter:
                await self._broadcast(buf)
            # Close
            for ep in self._peers:
                try:
                    await ep.close()
                except Exception:
                    pass
            listener.close()
        else:
            ep = await ucp.create_endpoint(self.master_addr, self.master_port)
            # Receive each bucket into provided buffers in order
            for bucket_id, buf in bucket_iter:
                await self._recv_into(ep, buf)
            try:
                await ep.close()
            except Exception:
                pass

    # --- helpers ---
    _peers: List[object] = field(default_factory=list)

    async def _on_connect(self, ep):  # noqa: ANN001
        self._peers.append(ep)

    async def _send_with_retry(self, ep, payload: np.ndarray) -> None:  # noqa: ANN001
        attempts = self.send_retries + 1
        delay = self.retry_delay_ms / 1000.0
        for i in range(attempts):
            try:
                await ep.send(payload)
                self._bytes_sent.inc(float(payload.nbytes))
                return
            except Exception:
                if i == attempts - 1:
                    self._log.error(f"send failed after {attempts} attempts")
                    try:
                        import time as _t
                        self._last_err.set(1.0)
                        self._last_err_ts.set(float(_t.time()))
                    except Exception:
                        pass
                    raise
                self._retries.inc(1.0)
                self._log.warning(f"send retry {i+1}/{attempts-1}")
                await asyncio.sleep(delay)

    async def _broadcast(self, buf: np.ndarray) -> None:
        # Send size then buffer in chunks to each peer (parallelized with optional concurrency cap)
        n = np.array([buf.nbytes], dtype=np.int64)
        peers = self._peers or []
        async def _send_all_header():
            if self.send_concurrency and self.send_concurrency > 0:
                for i in range(0, len(peers), self.send_concurrency):
                    batch = peers[i : i + self.send_concurrency]
                    await asyncio.gather(*[self._send_with_retry(ep, n) for ep in batch])
            else:
                await asyncio.gather(*[self._send_with_retry(ep, n) for ep in peers])

        await _send_all_header()

        total = buf.nbytes
        off = 0
        while off < total:
            end = min(off + self.chunk_bytes, total)
            view = buf[off:end]
            if self.send_concurrency and self.send_concurrency > 0:
                for i in range(0, len(peers), self.send_concurrency):
                    batch = peers[i : i + self.send_concurrency]
                    await asyncio.gather(*[self._send_with_retry(ep, view) for ep in batch])
            else:
                await asyncio.gather(*[self._send_with_retry(ep, view) for ep in peers])
            off = end

    async def _recv_into(self, ep, buf: np.ndarray) -> None:  # noqa: ANN001
        # Receive size then buffer (chunked)
        n = np.empty(1, dtype=np.int64)
        await ep.recv(n)
        expected = int(n[0])
        if expected != buf.nbytes:
            # Resize if caller preallocated exact size; otherwise slice
            if buf.nbytes < expected:
                raise RuntimeError(f"buffer too small: {buf.nbytes} < {expected}")
        off = 0
        while off < expected:
            end = min(off + self.chunk_bytes, expected)
            view = buf[off:end]
            await ep.recv(view)
            self._bytes_recv.inc(float(view.nbytes))
            off = end

    # Streaming replicate: invoke callback after each bucket receives locally
    def replicate_stream(self, bucket_iter: Iterable[Tuple[int, np.ndarray] | Tuple[int, np.ndarray, list[int]]], on_complete: Callable[[int, np.ndarray], None]) -> None:
        if self._ucp is None or self.world_size <= 1:
            for item in bucket_iter:
                b, buf = item[0], item[1]
                on_complete(b, buf)
            return
        # Simpler sequential streaming for UCX baseline
        async def _run_stream():
            if self.rank == 0:
                listener = await ucp.create_listener(self._on_connect, port=self.master_port)
                while len(self._peers) < self.world_size - 1:
                    await asyncio.sleep(0.01)
                for item in bucket_iter:
                    if len(item) == 3:
                        bucket_id, buf, consumers = item  # type: ignore[misc]
                        if self.rank not in consumers:
                            # Root must broadcast regardless; fall through
                            pass
                    else:
                        bucket_id, buf = item  # type: ignore[misc]
                    await self._broadcast(buf)
                    on_complete(bucket_id, buf)
                for ep in self._peers:
                    try:
                        await ep.close()
                    except Exception:
                        pass
                listener.close()
            else:
                ep = await ucp.create_endpoint(self.master_addr, self.master_port)
                for item in bucket_iter:
                    if len(item) == 3:
                        bucket_id, buf, consumers = item  # type: ignore[misc]
                        if self.rank not in consumers:
                            # Skip receive for non-consumer ranks
                            continue
                    else:
                        bucket_id, buf = item  # type: ignore[misc]
                    await self._recv_into(ep, buf)
                    on_complete(bucket_id, buf)
                try:
                    await ep.close()
                except Exception:
                    pass
        asyncio.run(_run_stream())

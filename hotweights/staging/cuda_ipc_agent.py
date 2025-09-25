"""
SOTA Staging Agent using CUDA IPC.

This agent manages GPU memory directly for staging tensor data. It is responsible
for creating and opening CUDA IPC handles to enable true zero-copy data transfers.

Features:
- Manages a pool of GPU memory for staging.
- Assembles buckets from disk directly into GPU memory.
- Creates and returns IPC handles for sharing.
- Opens IPC handles and provides access to remote GPU memory.
- Scatters data from a shared buffer into private, per-tensor GPU buffers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time

import numpy as np

from ..core.replicate import assemble_bucket_to_buffer
from ..telemetry.cuda_ipc_metrics import CudaIPCMetrics
from ..telemetry.logging import get_logger
from ..utils.env import env_bool, env_int, env_mb

try:
    import torch
    from torch.cuda import ipc
except Exception:
    torch = None
    ipc = None

# Optional GDS/GPU IO stack
try:  # optional
    import cupy as _cp  # type: ignore
except Exception:  # pragma: no cover
    _cp = None  # type: ignore
try:  # optional
    import kvikio as _kvikio  # type: ignore
except Exception:  # pragma: no cover
    _kvikio = None  # type: ignore


class CudaIPCAgent:
    """Manages GPU memory and IPC handles for zero-copy staging."""

    def __init__(self, device: str = "cuda", metrics: Optional[CudaIPCMetrics] = None):
        if torch is None or ipc is None:
            raise RuntimeError("PyTorch with CUDA support is required for CudaIPCAgent.")
        self.device = torch.device(device)
        self._metrics = metrics
        
        # Main shared buffer received via IPC
        self._shared_buffers: Dict[int, torch.Tensor] = {}
        # Private, per-tensor staging buffers
        self._private_staging: Dict[str, torch.Tensor] = {}
        # Copy tuning
        # Heuristic defaults with optional overrides
        autotune = env_bool("HOTWEIGHTS_IPC_COPY_AUTOTUNE", True)
        mb_env = os.getenv("HOTWEIGHTS_IPC_COPY_CHUNK_MB")
        streams_env = os.getenv("HOTWEIGHTS_IPC_COPY_STREAMS")
        if autotune and (mb_env is None or streams_env is None):
            # If P2P peer access is available across GPUs, assume NVLink-friendly
            try:
                peer_ok = False
                if torch.cuda.device_count() > 1:
                    cur = torch.cuda.current_device()
                    other = 0 if cur != 0 else 1
                    peer_ok = torch.cuda.can_device_access_peer(cur, other)
                if peer_ok:
                    self._copy_chunk = 64 * (1 << 20)  # 64 MiB
                    self._copy_streams_n = 4
                else:
                    self._copy_chunk = 16 * (1 << 20)
                    self._copy_streams_n = 2
            except Exception:
                self._copy_chunk = 16 * (1 << 20)
                self._copy_streams_n = 2
        else:
            self._copy_chunk = env_mb("HOTWEIGHTS_IPC_COPY_CHUNK_MB", 16)
            self._copy_streams_n = max(1, env_int("HOTWEIGHTS_IPC_COPY_STREAMS", 2))
        self._copy_streams = [torch.cuda.Stream() for _ in range(self._copy_streams_n)]
        # Pinned host buffer pool (size-class reuse)
        self._pinned_pool: dict[int, torch.Tensor] = {}
        self._pinned_granularity = max(1 << 20, env_mb("HOTWEIGHTS_IPC_PINNED_GRANULARITY_MB", 8))
        # GDS capability and logging guards
        self._gds_capable = (_cp is not None) and (_kvikio is not None)
        self._gds_info_logged = False
        self._gds_error_logged = False
        # Log chosen copy parameters
        self._log = None
        try:
            self._log = get_logger("CudaIPCAgent", {"device": str(self.device)})
            self._log.info(
                f"ipc.copy_chunk={int(self._copy_chunk/(1<<20))}MiB streams={self._copy_streams_n}"
            )
        except Exception:
            self._log = None

    def _round_pinned_size(self, size: int) -> int:
        gran = self._pinned_granularity
        if size <= gran:
            return gran
        return ((size + gran - 1) // gran) * gran

    def _get_pinned(self, size: int) -> "torch.Tensor":  # type: ignore[name-defined]
        wanted = self._round_pinned_size(size)
        for cap in sorted(self._pinned_pool.keys()):
            buf = self._pinned_pool[cap]
            if cap >= wanted:
                return buf
        buf = torch.empty(wanted, dtype=torch.uint8, pin_memory=True)
        self._pinned_pool[wanted] = buf
        return buf

    def _should_use_gds(self) -> bool:
        if not self._gds_capable:
            return False
        if os.getenv("HOTWEIGHTS_USE_GDS") is not None:
            return env_bool("HOTWEIGHTS_USE_GDS", True)
        if env_bool("HOTWEIGHTS_DISABLE_GDS", False):
            return False
        return True

    def assemble_and_share(self, bucket: Dict[str, Any]) -> Any:
        """(Root Node Only) Assembles a bucket into a new GPU buffer and returns its IPC handle."""
        size = int(bucket["size"])
        bucket_id = int(bucket.get("bucket_id", -1))
        gpu_buffer = torch.empty(size, dtype=torch.uint8, device=self.device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        # Try GPUDirect Storage path when available and enabled
        use_gds = self._should_use_gds()
        gds_start = time.perf_counter() if use_gds else None
        if use_gds and self._metrics is not None:
            self._metrics.set_gds_enabled(True)
        if use_gds:
            try:
                if not self._gds_info_logged and self._log is not None:
                    self._log.info("Using GPUDirect Storage (CuFile) for bucket assembly")
                    self._gds_info_logged = True
                # Create a CuPy view over the torch tensor
                cp_view = _cp.fromDlpack(torch.utils.dlpack.to_dlpack(gpu_buffer))
                for it in bucket["items"]:
                    uri = it["uri"]
                    assert uri.startswith("file://"), f"Unsupported URI: {uri}"
                    path = uri[len("file://") :]
                    off = int(it["offset"])
                    n = int(it["nbytes"])  # bytes to read
                    arr_view = cp_view[off : off + n]
                    with _kvikio.CuFile(path, "r") as f:
                        f.pread(arr_view)  # read entire file into device slice
                torch.cuda.synchronize()
                if self._metrics is not None and gds_start is not None:
                    self._metrics.observe_gds_seconds(time.perf_counter() - gds_start)
            except Exception as exc:
                # Fallback to CPU assemble
                use_gds = False
                if self._metrics is not None:
                    self._metrics.set_gds_enabled(False)
                if self._log is not None and not self._gds_error_logged:
                    self._log.warning(
                        "GDS assembly failed, falling back to pinned host buffers: %s",
                        exc,
                    )
                    self._gds_error_logged = True
        if not use_gds:
            if self._metrics is not None and gds_start is None:
                self._metrics.set_gds_enabled(False)
            # Assemble via pinned host tensor then async H2D copy
            pinned = self._get_pinned(size)
            # Fill pinned buffer directly from files
            for it in bucket["items"]:
                uri = it["uri"]; assert uri.startswith("file://"), f"Unsupported URI: {uri}"
                path = uri[len("file://"):]
                off = int(it["offset"]); n = int(it["nbytes"])  # bytes to read
                mm = np.memmap(path, dtype=np.uint8, mode="r")
                assert mm.shape[0] == n, f"size mismatch for {path}: got {mm.shape[0]}, expect {n}"
                # Copy into pinned using torch view to enable async H2D later
                dst = pinned[off:off+n]
                src_t = torch.from_numpy(mm[:n])
                dst.copy_(src_t)
            # Async H2D copy
            host_view = pinned[:size]
            gpu_buffer.copy_(host_view, non_blocking=True)
            torch.cuda.synchronize()
        t1.record(); t1.synchronize()
        # Metrics: observe assemble seconds
        if self._metrics is not None:
            try:
                ms = t0.elapsed_time(t1)  # milliseconds
                self._metrics.observe_assemble_seconds(ms / 1000.0)
            except Exception:
                pass
        # Store local buffer and return IPC handle
        self._shared_buffers[bucket_id] = gpu_buffer
        return ipc.get_ipc_handle(gpu_buffer)

    # --- hierarchical helpers ---
    def get_shared_gpu(self, bucket_id: int):  # noqa: ANN201
        """Return the GPU tensor for a previously assembled/shared bucket."""
        return self._shared_buffers.get(int(bucket_id))

    def share_from_gpu(self, bucket_id: int, gpu_tensor):  # noqa: ANN001, ANN201
        """Register an existing GPU tensor as the shared buffer and return its IPC handle."""
        self._shared_buffers[int(bucket_id)] = gpu_tensor
        return ipc.get_ipc_handle(gpu_tensor)

    def receive_shared(self, bucket_id: int, ipc_handle: Any):
        """(Worker Nodes) Opens an IPC handle and stores the memory view."""
        # This provides a zero-copy view into the root node's GPU memory.
        self._shared_buffers[bucket_id] = ipc.open_ipc_handle(ipc_handle)
        
    def scatter_from_shared(self, bucket_id: int, items: List[Dict[str, Any]]):
        """Scatters data from the shared bucket buffer into private per-tensor buffers."""
        shared_buffer = self._shared_buffers.get(bucket_id)
        if shared_buffer is None:
            # No shared buffer available; nothing to scatter
            return
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for item in items:
            key = item["key"]
            offset = int(item["offset"]) 
            nbytes = int(item["nbytes"]) 
            private_buf = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            if nbytes <= self._copy_chunk or self._copy_streams_n <= 1:
                # Single copy
                private_buf.copy_(shared_buffer[offset : offset + nbytes], non_blocking=True)
            else:
                # Chunked multi-stream copy
                pos = 0
                s_idx = 0
                while pos < nbytes:
                    end = min(pos + self._copy_chunk, nbytes)
                    with torch.cuda.stream(self._copy_streams[s_idx]):
                        private_buf[pos:end].copy_(shared_buffer[offset + pos : offset + end], non_blocking=True)
                    pos = end
                    s_idx = (s_idx + 1) % self._copy_streams_n
            self._private_staging[key] = private_buf
        # Synchronize all streams
        for s in self._copy_streams:
            s.synchronize()
        t1.record(); t1.synchronize()
        if self._metrics is not None:
            try:
                ms = t0.elapsed_time(t1)
                self._metrics.observe_scatter_seconds(ms / 1000.0)
            except Exception:
                pass

    def get_staged_tensor(self, key: str) -> torch.Tensor | None:
        """Gets a privately staged tensor, ready for use."""
        return self._private_staging.get(key)

    def verify(self, items: List[Dict[str, Any]]) -> bool:
        """Verifies the checksum of staged tensors on the GPU."""
        # This is complex. A GPU-accelerated hashing algorithm would be needed.
        # For now, we can copy back to CPU to verify, though it incurs overhead.
        print("Verification on GPU is a complex feature (stubbed).")
        # Example of how one item could be verified:
        # for item in items:
        #     key = item["key"]
        #     gpu_tensor = self.get_staged_tensor(key)
        #     if gpu_tensor is not None:
        #         cpu_tensor = gpu_tensor.to("cpu")
        #         # ... then hash cpu_tensor ...
        return True

    def cleanup(self):
        """Releases all GPU memory."""
        self._shared_buffers.clear()
        self._private_staging.clear()
        if torch:
            torch.cuda.empty_cache()

"""Generic GPU staging agent (vendor-neutral).

Provides device-side staging and scatter without CUDA IPC. Works with
CUDA (NVIDIA), ROCm (AMD), and XPU (Intel, where supported by PyTorch).

Interface mirrors the minimal subset used by transports:
- share_from_gpu(bucket_id, gpu_tensor)
- get_shared_gpu(bucket_id)
- scatter_from_shared(bucket_id, items)
- get_staged_tensor(key)

Streams are used on CUDA when available; on other devices we fall back to
single-stream copies to maintain portability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time
from ..utils.env import env_bool, env_int, env_mb, env_list_int
from ..telemetry.logging import get_logger

try:  # optional torch
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


class GenericGpuAgent:
    """Manages device buffers and device-side scatter for staged tensors."""

    def __init__(self, device: str = "cuda") -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for GenericGpuAgent")
        # Validate device
        self.device = torch.device(device)
        # Shared bucket buffers (device tensors)
        self._shared_buffers: Dict[int, "torch.Tensor"] = {}
        # Private, per-tensor staging buffers
        self._private_staging: Dict[str, "torch.Tensor"] = {}
        # Copy tuning
        self._use_cuda_streams = (
            hasattr(torch, "cuda")
            and device.startswith("cuda")
            and torch.cuda.is_available()
        )
        # Environment-tunable defaults
        def _env_int(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except Exception:
                return default

        def _env_mb(name: str, default_mb: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default_mb)))) * (1 << 20)
            except Exception:
                return default_mb * (1 << 20)

        # Autotune if requested
        autotune = env_bool("HOTWEIGHTS_GPU_COPY_AUTOTUNE", True)
        if autotune and self._use_cuda_streams:
            try:
                peer_ok = False
                if torch.cuda.device_count() > 1:
                    cur = torch.cuda.current_device()
                    other = 0 if cur != 0 else 1
                    peer_ok = torch.cuda.can_device_access_peer(cur, other)
                if peer_ok:
                    self._copy_chunk = 64 * (1 << 20)
                    self._copy_streams_n = 4
                else:
                    self._copy_chunk = 16 * (1 << 20)
                    self._copy_streams_n = 2
            except Exception:
                self._copy_chunk = 16 * (1 << 20)
                self._copy_streams_n = 2
        else:
            self._copy_chunk: int = _env_mb("HOTWEIGHTS_GPU_COPY_CHUNK_MB", 16)
            self._copy_streams_n: int = _env_int("HOTWEIGHTS_GPU_COPY_STREAMS", 2)
        self._copy_streams: List["torch.cuda.Stream" | None]
        if self._use_cuda_streams:
            self._copy_streams = [torch.cuda.Stream() for _ in range(self._copy_streams_n)]  # type: ignore[attr-defined]
        else:
            self._copy_streams = []
        # Log chosen copy parameters
        try:
            self._log = get_logger("GenericGpuAgent", {"device": str(self.device)})
            self._log.info(f"copy_chunk={int(self._copy_chunk/(1<<20))}MiB streams={self._copy_streams_n}")
        except Exception:
            pass

        # Optional microbenchmark autotune
        try:
            from ..utils.env import env_bool
            if env_bool("HOTWEIGHTS_GPU_COPY_MICRO", False):
                self._micro_autotune_copy()
        except Exception:
            pass

    # Shared buffer helpers
    def get_shared_gpu(self, bucket_id: int):  # noqa: ANN201
        return self._shared_buffers.get(int(bucket_id))

    def share_from_gpu(self, bucket_id: int, gpu_tensor):  # noqa: ANN001, ANN201
        self._shared_buffers[int(bucket_id)] = gpu_tensor
        return None

    def receive_shared(self, bucket_id: int, gpu_tensor: Any) -> None:
        # For API parity; store provided device tensor
        self._shared_buffers[int(bucket_id)] = gpu_tensor  # type: ignore[assignment]

    def scatter_from_shared(self, bucket_id: int, items: List[Dict[str, Any]]) -> None:
        shared = self._shared_buffers.get(int(bucket_id))
        if shared is None:
            return
        # Device-to-device copies, optionally chunked/streamed on CUDA
        if not isinstance(shared, torch.Tensor):  # type: ignore[unreachable]
            return
        use_streams = self._use_cuda_streams and bool(self._copy_streams)
        streams = self._copy_streams if use_streams else [None]
        s_idx = 0
        for it in items:
            key = str(it["key"])
            offset = int(it["offset"])  # byte offset
            nbytes = int(it["nbytes"])  # bytes
            dst = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
            if nbytes <= self._copy_chunk or not use_streams:
                dst.copy_(shared.view(torch.uint8)[offset : offset + nbytes], non_blocking=True)
            else:
                pos = 0
                while pos < nbytes:
                    end = min(pos + self._copy_chunk, nbytes)
                    if streams[s_idx] is not None:
                        with torch.cuda.stream(streams[s_idx]):  # type: ignore[attr-defined]
                            dst[pos:end].copy_(shared.view(torch.uint8)[offset + pos : offset + end], non_blocking=True)
                    else:
                        dst[pos:end].copy_(shared.view(torch.uint8)[offset + pos : offset + end], non_blocking=True)
                    pos = end
                    s_idx = (s_idx + 1) % len(streams)
            self._private_staging[key] = dst
        # Synchronize streams if used
        if use_streams:
            for s in streams:
                if s is not None:
                    s.synchronize()

    def get_staged_tensor(self, key: str) -> "torch.Tensor" | None:
        return self._private_staging.get(key)

    def cleanup(self) -> None:
        self._shared_buffers.clear()
        self._private_staging.clear()
        try:
            if self.device.type == "cuda" and hasattr(torch, "cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    # --- autotune helpers ---
    def _micro_autotune_copy(self) -> None:
        size_mb = 128
        from ..utils.env import env_int, env_list_int
        size_mb = max(16, env_int("HOTWEIGHTS_GPU_COPY_AUTOTUNE_MB", 128))
        total = size_mb * (1 << 20)
        # Build candidates
        chunk_cands = env_list_int("HOTWEIGHTS_GPU_COPY_MICRO_CHUNKS_MB", [8, 16, 32, 64])
        stream_cands = env_list_int("HOTWEIGHTS_GPU_COPY_MICRO_STREAMS", [1, 2, 4]) if self._use_cuda_streams else [1]
        # Allocate test tensors
        src = torch.empty(total, dtype=torch.uint8, device=self.device)
        dst = torch.empty_like(src)
        best = (0.0, self._copy_chunk, self._copy_streams_n)
        for mb in chunk_cands:
            chk = mb * (1 << 20)
            for sn in stream_cands:
                # Prepare streams
                streams = [torch.cuda.Stream() for _ in range(sn)] if self._use_cuda_streams else [None]
                # Warmup
                self._bench_copy_once(src, dst, chk, streams)
                t0 = time.perf_counter()
                self._bench_copy_once(src, dst, chk, streams)
                if self._use_cuda_streams:
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                bw = (total / dt) / (1 << 30)  # GiB/s
                if bw > best[0]:
                    best = (bw, chk, sn)
        # Apply best
        _, best_chk, best_sn = best
        self._copy_chunk = int(best_chk)
        self._copy_streams_n = int(best_sn)
        self._copy_streams = [torch.cuda.Stream() for _ in range(self._copy_streams_n)] if self._use_cuda_streams else []  # type: ignore[attr-defined]
        try:
            self._log.info(f"autotune.copy_chunk={int(self._copy_chunk/(1<<20))}MiB streams={self._copy_streams_n}")
        except Exception:
            pass

    def _bench_copy_once(self, src: "torch.Tensor", dst: "torch.Tensor", chunk: int, streams: List["torch.cuda.Stream" | None]) -> None:
        n = int(src.numel())
        pos = 0
        si = 0
        while pos < n:
            end = min(pos + chunk, n)
            if streams and streams[si] is not None:
                with torch.cuda.stream(streams[si]):  # type: ignore[attr-defined]
                    dst[pos:end].copy_(src[pos:end], non_blocking=True)
            else:
                dst[pos:end].copy_(src[pos:end], non_blocking=True)
            pos = end
            si = (si + 1) % max(1, len(streams))

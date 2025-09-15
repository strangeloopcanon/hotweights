"""Transport selection helpers (shared by CLI and Worker).

Creates the best available transport based on device/vendor capabilities:
- NVIDIA with CUDA IPC → CudaIPCTransport
- ROCm/Intel or CUDA without IPC → GpuBroadcastTransport
- CPU fallback → TransportManager (MPI/UCX/local) wrapped in a shim with replicate(plan).

Returns (transport_obj, caps_dict) where caps_dict is safe to publish to coordinator.
"""
from __future__ import annotations

from typing import Any, Tuple
import os


def choose_transport(device: str, coord_endpoint: str | None = None) -> tuple[Any, dict]:  # noqa: ANN401
    caps = {"transport": "unknown", "rank": int(os.getenv("RANK", "0"))}
    # Optional log of selection
    try:
        from ..telemetry.logging import get_logger
        _log = get_logger("Selection", {"device": device, "rank": caps["rank"]})
    except Exception:
        _log = None  # type: ignore
    # Try GPU paths first
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    is_cuda = bool(torch) and device.startswith("cuda") and getattr(torch.cuda, "is_available", lambda: False)()
    is_xpu = bool(torch) and hasattr(torch, "xpu") and device.startswith("xpu") and getattr(getattr(torch, "xpu"), "is_available", lambda: False)()  # type: ignore[attr-defined]

    # NVIDIA CUDA-IPC fast path
    if is_cuda:
        try:
            from torch.cuda import ipc as _ipc  # type: ignore
            _ = _ipc
            from ..staging.cuda_ipc_agent import CudaIPCAgent
            from ..transport.cuda_ipc import CudaIPCTransport
            from ..telemetry.cuda_ipc_metrics import CudaIPCMetrics

            rank = int(os.getenv("RANK", "0"))
            metrics = CudaIPCMetrics(rank)
            agent = CudaIPCAgent(device=device, metrics=metrics)
            transport = CudaIPCTransport(agent=agent, metrics=metrics, coord_endpoint=coord_endpoint)
            caps["transport"] = "cuda_ipc"
            if _log is not None:
                _log.info("selected=cuda_ipc")
            return transport, caps
        except Exception:
            pass

    # GPU broadcast path (ROCm/Intel or CUDA without IPC)
    if is_cuda or is_xpu:
        try:
            from ..staging.gpu_agent import GenericGpuAgent
            from ..transport.gpu_broadcast import GpuBroadcastTransport

            agent = GenericGpuAgent(device=device)
            backend = "ccl" if is_xpu else "nccl"
            transport = GpuBroadcastTransport(agent=agent, backend=backend)
            caps["transport"] = "gpu_broadcast"
            if _log is not None:
                _log.info("selected=gpu_broadcast")
            return transport, caps
        except Exception:
            pass

    # CPU fallback via TransportManager shim
    from ..staging.host_agent import HostAgent
    import numpy as _np
    from ..core.replicate import assemble_bucket as _assemble_bucket
    from ..core.replicate import scatter_bucket as _scatter_bucket

    host = HostAgent(use_pinned=False)

    # Try to use TransportManager if available; otherwise fall back to local path
    replicator = None
    try:
        from ..transport.transport_manager import TransportManager

        tm = TransportManager(world_size=int(os.getenv("WORLD_SIZE", "1")), rank=int(os.getenv("RANK", "0")), auto_select=True)
        try:
            replicator = tm.get_replicator()
        except Exception:
            replicator = None
    except Exception:
        replicator = None

    class _Shim:
        def __init__(self, rep, host_agent):
            self._r = rep
            self._h = host_agent

        def replicate(self, plan):  # noqa: ANN001
            if self._r is None:
                # Pure local path: assemble and scatter bucket-by-bucket
                for b in plan.get("buckets", []):
                    items = b["items"]
                    buf = _assemble_bucket(items)
                    _scatter_bucket(self._h, items, buf)
                return
            rank_local = getattr(self._r, "world_rank", getattr(self._r, "rank", 0))
            bucket_bufs: list[tuple[dict, _np.ndarray]] = []

            def gen():
                for b in plan.get("buckets", []):
                    items = b["items"]
                    size = int(b["size"])  # precomputed
                    consumers = b.get("consumer_ranks")
                    if consumers is not None and rank_local not in consumers:
                        continue
                    group_root = 0 if not consumers else min(int(x) for x in consumers)
                    if rank_local == group_root:
                        buf = _assemble_bucket(items)
                    else:
                        buf = _np.empty(size, dtype=_np.uint8)
                    bucket_bufs.append((b, buf))
                    if consumers is None:
                        yield (int(b["bucket_id"]), buf)
                    else:
                        yield (int(b["bucket_id"]), buf, list(int(x) for x in consumers))

            def on_complete(_bid: int, _buf: _np.ndarray) -> None:
                b, buf = bucket_bufs.pop(0)
                _scatter_bucket(self._h, b["items"], buf)

            if hasattr(self._r, "replicate_stream"):
                self._r.replicate_stream(gen(), on_complete)  # type: ignore[attr-defined]
            else:
                self._r.replicate(gen())
                while bucket_bufs:
                    on_complete(0, bucket_bufs[0][1])

    caps["transport"] = "cpu_fallback"
    if _log is not None:
        _log.info("selected=cpu_fallback")
    return _Shim(replicator, host), caps

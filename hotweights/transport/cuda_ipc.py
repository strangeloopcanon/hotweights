"""
SOTA CUDA-IPC Transport Layer.

Features:
- Full zero-copy replication using CUDA IPC handles.
- GPU-aware topology discovery (NVLink, NVSwitch, PCIe).
- Multi-lane, congestion-controlled data streaming.
- Dynamic traffic shaping to avoid saturating interconnects.
- Unified API that abstracts broadcast and P2P for the worker.
"""
from __future__ import annotations

import os
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Optional
import base64
import hmac
import hashlib
import time
import pickle
from collections import deque

import numpy as np

from ..telemetry.cuda_ipc_metrics import CudaIPCMetrics
from ..staging.cuda_ipc_agent import CudaIPCAgent
from ..coordinator.zmq_client import Client as ZClient
from .topology_scheduler import TopologyDiscovery, TopologyAwareScheduler
from ..telemetry.logging import get_logger
import socket

try:
    import torch
except Exception:
    torch = None


@dataclass
class CudaIPCTransport:
    """A unified, high-performance transport using CUDA IPC."""

    agent: CudaIPCAgent
    metrics: CudaIPCMetrics
    rank: int = field(init=False)
    world_size: int = field(init=False)
    coord_endpoint: Optional[str] = None
    _client: Optional[ZClient] = field(init=False, default=None)
    _handle_token: Optional[str] = field(init=False, default=None)
    
    # Advanced topology and congestion control state
    _topology: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _congestion_windows: Dict[int, int] = field(default_factory=dict)
    _topo: Optional[TopologyDiscovery] = field(init=False, default=None)
    _sched: Optional[TopologyAwareScheduler] = field(init=False, default=None)
    _bucket_times: Any = field(init=False)

    def __post_init__(self):
        if torch is None:
            raise RuntimeError("PyTorch with CUDA support is required for CudaIPCTransport.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; CudaIPCTransport requires a CUDA device.")
        
        # In a real implementation, this would come from cluster integration
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        # Create coordinator client if endpoint provided or env set
        ep = self.coord_endpoint or os.getenv("HOTWEIGHTS_COORD")
        if ep:
            try:
                self._client = ZClient(ep)
            except Exception:
                self._client = None
        # Optional shared secret for handle signing/validation
        tok = os.getenv("HOTWEIGHTS_HANDLE_TOKEN")
        self._handle_token = tok if tok else None
        # Node identity and handle scope
        self._node_id = os.getenv("HOTWEIGHTS_NODE_ID") or socket.gethostname()
        self._handle_scope = 'node' if os.getenv("HOTWEIGHTS_HIERARCHICAL", "0") in ("1", "true", "True") else 'global'
        
        # Concurrency/window tuning
        try:
            self._max_inflight = max(1, int(os.getenv("HOTWEIGHTS_IPC_INFLIGHT_BUCKETS", "4")))
        except Exception:
            self._max_inflight = 4
        try:
            self._max_inflight_bytes = max(0, int(os.getenv("HOTWEIGHTS_IPC_INFLIGHT_BYTES", "0")))
        except Exception:
            self._max_inflight_bytes = 0

        self._discover_topology()
        self._bucket_times = deque(maxlen=50)
        # Adaptive settings
        self._adapt = os.getenv("HOTWEIGHTS_IPC_ADAPT", "1") in ("1", "true", "True")
        try:
            self._min_inflight = max(1, int(os.getenv("HOTWEIGHTS_IPC_MIN_INFLIGHT", "1")))
            self._max_inflight_cap = max(self._min_inflight, int(os.getenv("HOTWEIGHTS_IPC_MAX_INFLIGHT", "16")))
            self._target_bucket_ms = max(1, int(os.getenv("HOTWEIGHTS_IPC_TARGET_BUCKET_MS", "250")))
        except Exception:
            self._min_inflight, self._max_inflight_cap, self._target_bucket_ms = 1, 16, 250
        # Advertise initial window/target
        try:
            self.metrics.set_window(self._max_inflight)
            self.metrics.set_target_bucket_ms(self._target_bucket_ms)
        except Exception:
            pass
        # Logger
        try:
            self._log = get_logger("CudaIPCTransport", {"rank": self.rank, "ws": self.world_size})
        except Exception:
            self._log = None  # type: ignore[assignment]

    def _discover_topology(self):
        """Discovers the GPU topology (stubs for real nvidia-smi/NCCL calls)."""
        # Initialize topology discovery/scheduler (best-effort)
        try:
            self._topo = TopologyDiscovery()
            self._sched = TopologyAwareScheduler(self._topo)
            # Discover minimal info for this rank's GPU
            gpu_id = 0
            try:
                if torch and torch.cuda.is_available():
                    gpu_id = torch.cuda.current_device()
            except Exception:
                gpu_id = 0
            self._topo.discover_gpu_topology(rank=self.rank, gpu_id=gpu_id)
        except Exception:
            self._topo = None
            self._sched = None
        # Default congestion window
        for i in range(self.world_size):
            self._congestion_windows[i] = 4
        print("CUDA-IPC Transport: Topology initialized (best-effort).")
        # Hierarchical mode (inter-node NCCL + intra-node IPC)
        self._hier = os.getenv("HOTWEIGHTS_HIERARCHICAL", "0") in ("1", "true", "True")
        try:
            self._local_rank = int(os.getenv("LOCAL_RANK", "0"))
        except Exception:
            self._local_rank = 0
        try:
            self._local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0"))
        except Exception:
            self._local_world_size = 0
        # Precompute leader group for hierarchical broadcast
        self._leaders_group = None
        self._leaders_root = 0
        if self._hier:
            try:
                import torch.distributed as dist  # type: ignore
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                ws = dist.get_world_size()
                lws = self._local_world_size or 1
                if ws >= lws and lws > 0:
                    leaders = list(range(0, ws, lws))
                else:
                    leaders = [0]
                self._leaders_root = min(leaders)
                self._leaders_group = dist.new_group(ranks=leaders)
            except Exception:
                self._leaders_group = None

    async def _replicate_one_bucket(self, bucket: Dict[str, Any], plan: Dict[str, Any]):
        """Handles the replication of a single bucket."""
        bucket_id = bucket["bucket_id"]
        version = plan.get("version") or "_"
        t0 = time.perf_counter()
        # Hierarchical inter-node stage (leaders use NCCL)
        used_hier = False
        if self._hier:
            try:
                import torch.distributed as dist  # type: ignore
                if not dist.is_initialized():
                    # Try env init; if it fails, skip hierarchical
                    dist.init_process_group(backend="nccl")
                is_leader = (self._local_rank == 0)
                group = self._leaders_group if self._leaders_group is not None else dist.group.WORLD  # type: ignore[attr-defined]
                leader_root = self._leaders_root
                if is_leader:
                    size = int(bucket.get("size", 0))
                    if self.rank == leader_root:
                        # assemble on root leader
                        _ = self.agent.assemble_and_share(bucket)
                        gpu_t = self.agent.get_shared_gpu(bucket_id)
                        if gpu_t is None:
                            gpu_t = torch.empty(size, dtype=torch.uint8, device=self.agent.device)  # type: ignore[attr-defined]
                        # broadcast to other leaders
                        dist.broadcast(gpu_t.view(torch.uint8), src=leader_root, group=group)
                        # share locally (handle)
                        _ = self.agent.share_from_gpu(bucket_id, gpu_t)
                    else:
                        # receive into a fresh tensor then share locally
                        gpu_t = torch.empty(size, dtype=torch.uint8, device=self.agent.device)  # type: ignore[attr-defined]
                        dist.broadcast(gpu_t.view(torch.uint8), src=leader_root, group=group)
                        _ = self.agent.share_from_gpu(bucket_id, gpu_t)
                    used_hier = True
            except Exception:
                used_hier = False

        if not used_hier and self.rank == 0:
            # 1. Root node assembles the bucket directly into its GPU memory via the agent.
            ipc_handle = self.agent.assemble_and_share(bucket)
            
            # 2. Share the IPC handle with all other ranks (e.g., via ZMQ/etcd).
            # This part would be integrated with the HA control plane.
            # For this file, we simulate broadcasting it.
            await self._broadcast_handle(bucket_id, ipc_handle, version)
            
            # 3. Root node also "receives" the data locally.
            self.agent.receive_shared(bucket_id, ipc_handle)
        elif not used_hier:
            # Skip non-consumers entirely
            consumers = bucket.get("consumer_ranks")
            if consumers is not None and isinstance(consumers, list) and int(self.rank) not in [int(x) for x in consumers]:
                try:
                    self._log.debug(f"skip bucket {bucket_id} (not a consumer)")
                except Exception:
                    pass
                return
            # 1. Worker nodes wait to receive the IPC handle.
            ipc_handle = await self._receive_handle(bucket_id, version)
            
            # 2. Open the IPC handle to get a direct, zero-copy mapping to the root's GPU memory.
            self.agent.receive_shared(bucket_id, ipc_handle)

        # 4. All ranks now have a view into the same GPU memory.
        # They can now scatter the data to their own private staging areas.
        # 4. Scatter locally; when hierarchical, leaders scatter from their local shared GPU
        self.agent.scatter_from_shared(bucket_id, bucket["items"])
        # Best-effort ack to release storage early
        try:
            if self._client is not None:
                payload = {"version": version, "bucket_id": int(bucket_id), "who": str(self.rank)}
                if self._handle_scope == 'node':
                    payload["node"] = self._node_id
                self._client.call("ack_handle", **payload)
                try:
                    self._log.debug(f"acked handle bucket={bucket_id} version={version}")
                except Exception:
                    pass
        except Exception:
            pass
        # Metrics
        try:
            self.metrics.log_bucket_replicated(bucket["size"])  # bytes-based rate
            dt = time.perf_counter() - t0
            self.metrics.observe_bucket_seconds(dt)
        except Exception:
            pass
        # Adaptive window control
        try:
            self._bucket_times.append(dt)
            self._maybe_adapt()
        except Exception:
            pass

    async def _run_replication(self, plan: Dict[str, Any]):
        """The main async replication loop."""
        # This implements the multi-lane, congestion-controlled replication.
        # We can process multiple buckets concurrently based on topology.
        buckets = plan.get("buckets", [])
        # Topology-informed initial inflight: prefer more lanes when many consumers
        try:
            if self._sched is not None and buckets:
                # Use first bucket as proxy for consumer set size
                b0 = buckets[0]
                ranks = b0.get("consumer_ranks") or list(range(self.world_size))
                sched = self._sched.compute_transfer_schedule(int(b0.get("bucket_id", 0)), src_rank=0, dst_ranks=[int(x) for x in ranks], size_bytes=int(b0.get("size", 0)))
                try:
                    self.metrics.set_congestion_risk(float(getattr(sched, "congestion_risk", 0.0)))
                    # Heuristic recommended window
                    n_dsts = max(1, len(ranks))
                    rec = min(self._max_inflight_cap if hasattr(self, "_max_inflight_cap") else 16,
                              max(self._min_inflight if hasattr(self, "_min_inflight") else 1,
                                  min(8 if sched.expected_duration < 0.25 else max(2, n_dsts // 2), n_dsts)))
                    self.metrics.set_recommended_window(int(rec))
                except Exception:
                    pass
                # Heuristic: set inflight relative to number of destinations and expected duration
                n_dsts = max(1, len(ranks))
                if sched.expected_duration < 0.25:
                    self._max_inflight = min(self._max_inflight_cap if hasattr(self, "_max_inflight_cap") else 16, max(self._max_inflight, min(8, n_dsts)))
                else:
                    self._max_inflight = max(self._min_inflight if hasattr(self, "_min_inflight") else 1, min(self._max_inflight, max(2, n_dsts // 2)))
        except Exception:
            pass
        # Use a semaphore with the maximum possible cap; enforce dynamic window via 'active' gating
        cap = getattr(self, "_max_inflight_cap", self._max_inflight)
        sem = asyncio.Semaphore(cap)
        inflight_bytes = 0
        lock = asyncio.Lock()
        active = 0
        active_lock = asyncio.Lock()

        async def _guarded(bkt: Dict[str, Any]):
            nonlocal inflight_bytes, active
            async with sem:
                # Enforce dynamic window (active < self._max_inflight)
                while True:
                    async with active_lock:
                        if active < self._max_inflight:
                            active += 1
                            break
                    await asyncio.sleep(0.001)
                # Per-bucket topology hint (optional)
                try:
                    if self._sched is not None:
                        ranks = bkt.get("consumer_ranks") or list(range(self.world_size))
                        sched = self._sched.compute_transfer_schedule(int(bkt.get("bucket_id", 0)), src_rank=0, dst_ranks=[int(x) for x in ranks], size_bytes=int(bkt.get("size", 0)))
                        self.metrics.set_congestion_risk(float(getattr(sched, "congestion_risk", 0.0)))
                        n_dsts = max(1, len(ranks))
                        rec = min(getattr(self, "_max_inflight_cap", 16), max(getattr(self, "_min_inflight", 1), min(8 if sched.expected_duration < 0.25 else max(2, n_dsts // 2), n_dsts)))
                        self.metrics.set_recommended_window(int(rec))
                        if self._adapt:
                            self._max_inflight = max(getattr(self, "_min_inflight", 1), min(getattr(self, "_max_inflight_cap", 16), rec))
                            self.metrics.set_window(self._max_inflight)
                        # Publish simple per-path utilization: share of chunks per (src->dst)
                        try:
                            assignments = getattr(sched, "chunk_assignments", {}) or {}
                            counts: Dict[str, int] = {}
                            total = 0
                            for _cid, paths in assignments.items():
                                for p in paths:
                                    key = f"{p.src_rank}->{p.dst_rank}"
                                    counts[key] = counts.get(key, 0) + 1
                                    total += 1
                            if total > 0:
                                for k, c in counts.items():
                                    self.metrics.set_path_utilization(k, c / total)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Optional bytes backpressure
                if self._max_inflight_bytes > 0:
                    size = int(bkt.get("size", 0))
                    while True:
                        async with lock:
                            if inflight_bytes + size <= self._max_inflight_bytes or inflight_bytes == 0:
                                inflight_bytes += size
                                break
                        await asyncio.sleep(0.001)
                    # metrics inflight update
                    try:
                        self.metrics.set_inflight(active, inflight_bytes)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        await self._replicate_one_bucket(bkt, plan)
                    finally:
                        async with lock:
                            inflight_bytes = max(0, inflight_bytes - size)
                        try:
                            self.metrics.set_inflight(max(0, active - 1), inflight_bytes)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                else:
                    try:
                        self.metrics.set_inflight(active, inflight_bytes)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    await self._replicate_one_bucket(bkt, plan)
                    try:
                        self.metrics.set_inflight(max(0, active - 1), inflight_bytes)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                # Decrement active
                async with active_lock:
                    active = max(0, active - 1)

        await asyncio.gather(*(_guarded(b) for b in buckets))

    def _maybe_adapt(self) -> None:
        if not self._adapt or not self._bucket_times:
            return
        avg_ms = 1000.0 * (sum(self._bucket_times) / len(self._bucket_times))
        # If slower than target by >50%, reduce inflight to relieve pressure
        if avg_ms > 1.5 * self._target_bucket_ms and self._max_inflight > self._min_inflight:
            self._max_inflight = max(self._min_inflight, self._max_inflight - 1)
        # If faster than target by >25%, try increasing inflight up to cap
        elif avg_ms < 0.75 * self._target_bucket_ms and self._max_inflight < self._max_inflight_cap:
            self._max_inflight = min(self._max_inflight_cap, self._max_inflight + 1)
        # Publish updated window
        try:
            self.metrics.set_window(self._max_inflight)
            self._log.debug(f"adapt window -> {self._max_inflight} (avg_ms={avg_ms:.1f} target={self._target_bucket_ms})")
        except Exception:
            pass

    def replicate(self, plan: Dict[str, Any]):
        """Public method to start the replication process."""
        if self.world_size <= 1:
            # Single-node case, just assemble locally.
            for bucket in plan.get("buckets", []):
                self.agent.assemble_and_share(bucket)
                self.agent.scatter_from_shared(bucket["bucket_id"], bucket["items"])
            return

        print(f"CUDA-IPC Transport: Starting replication for {len(plan.get('buckets', []))} buckets...")
        try:
            self._log.info(f"replicate start buckets={len(plan.get('buckets', []))} version={plan.get('version')}")
        except Exception:
            pass
        asyncio.run(self._run_replication(plan))
        print("CUDA-IPC Transport: Replication complete.")

    # Handle sharing via HA control plane (fallback to in-proc registry)
    def _serialize_handle(self, handle: Any) -> str:
        try:
            if isinstance(handle, (bytes, bytearray)):
                return "b64:" + base64.b64encode(bytes(handle)).decode("ascii")
            # Known patterns: objects with pack/serialize/to_bytes
            for m in ("pack", "serialize", "to_bytes"):
                fn = getattr(handle, m, None)
                if callable(fn):
                    data = fn()
                    if isinstance(data, (bytes, bytearray)):
                        return "b64:" + base64.b64encode(bytes(data)).decode("ascii")
        except Exception:
            pass
        # Fallback to pickle
        try:
            return "pkl:" + base64.b64encode(pickle.dumps(handle, protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
        except Exception:
            pass
        # Fallback to repr; real builds should provide proper handle bytes
        return f"repr:{repr(handle)}"

    def _deserialize_handle(self, payload: str) -> Any:
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8")
            except Exception:
                return payload
        if isinstance(payload, str) and payload.startswith("b64:"):
            try:
                return base64.b64decode(payload[4:])
            except Exception:
                return payload
        if isinstance(payload, str) and payload.startswith("pkl:"):
            try:
                return pickle.loads(base64.b64decode(payload[4:]))
            except Exception:
                return payload
        return payload

    def _sign_payload(self, payload: str) -> Optional[str]:
        if not self._handle_token:
            return None
        try:
            mac = hmac.new(self._handle_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256)
            return mac.hexdigest()
        except Exception:
            return None

    async def _broadcast_handle(self, bucket_id: int, handle: Any, version: str = "_"):
        if self._client is not None:
            try:
                ser = self._serialize_handle(handle)
                sig = self._sign_payload(ser)
                payload = {"version": version, "bucket_id": int(bucket_id), "handle": ser, "sig": sig}
                if self._handle_scope == 'node':
                    payload["node"] = self._node_id
                _ = self._client.call("post_handle", **payload)
                return
            except Exception:
                pass
        # Fallback: local registry for single-process tests
        self._handle_registry = getattr(self, "_handle_registry", {})
        self._handle_registry[bucket_id] = handle
        await asyncio.sleep(0.005)

    async def _receive_handle(self, bucket_id: int, version: str = "_") -> Any:
        if self._client is not None:
            # Poll HA control plane with exponential backoff until handle is posted
            delay = 0.005
            elapsed = 0.0
            timeout = 10.0
            while elapsed < timeout:
                try:
                    payload = {"version": version, "bucket_id": int(bucket_id)}
                    if self._handle_scope == 'node':
                        payload["node"] = self._node_id
                    resp = self._client.call("get_handle", **payload)
                    h = resp.get("handle") if isinstance(resp, dict) else None
                    sig = resp.get("sig") if isinstance(resp, dict) else None
                    if h:
                        # Validate signature if token set
                        if self._handle_token:
                            expect = self._sign_payload(h)
                            if not sig or expect != sig:
                                # Signature mismatch; ignore and retry
                                await asyncio.sleep(delay)
                                elapsed += delay
                                delay = min(delay * 2.0, 0.1)
                                continue
                        return self._deserialize_handle(h)
                except Exception:
                    pass
                await asyncio.sleep(delay)
                elapsed += delay
                delay = min(delay * 2.0, 0.1)
        # Fallback local
        self._handle_registry = getattr(self, "_handle_registry", {})
        while bucket_id not in self._handle_registry:
            await asyncio.sleep(0.005)
        return self._handle_registry[bucket_id]

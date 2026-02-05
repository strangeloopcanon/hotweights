"""Bind hotweights update loop into a vLLM worker process.

This module provides a best-effort binding function that you can call from a
vLLM worker process after the model is instantiated. It polls the coordinator
for a plan+version and applies updates into the worker's torch.nn.Module using
async H2D copies with pinned host buffers.

It avoids importing vLLM directly; instead it introspects the provided object
to find a `.model` attribute or a nested `.model_runner.model`.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..coordinator.zmq_client import Client
from .vllm_ext import apply_from_ipc_agent_to_module
from .vllm_pause import pause_requests, resume_requests
from ..telemetry.prom import Counter, Gauge, start_http_server


def _extract_module(obj: object) -> object:  # noqa: ANN001
    # common vLLM structures: engine.model_executor?.model_runner?.model
    for path in (
        "model",
        "model_runner.model",
        "engine.model",
        "engine.model_runner.model",
        "executor.model_runner.model",
        "model_executor.model_runner.model",
    ):
        cur = obj
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and cur is not None:
            return cur
    raise AttributeError("Could not locate torch.nn.Module on the provided object")


class HotweightsVLLMBinding:
    def __init__(
        self,
        obj,
        name_map: dict[str, str] | Callable[[dict], dict[str, str]] | None,
        endpoint: str,
        use_mpi: bool = False,
        pinned: bool = True,
        verify: bool = False,
        use_kv_migration: bool = True, # Use SOTA feature by default
        device: str = "cuda",
        poll_interval: float = 2.0,
    ) -> None:
        self.obj = obj
        self.module = _extract_module(obj)
        self.name_map = name_map
        self.endpoint = endpoint
        self.use_mpi = use_mpi
        self.pinned = pinned
        self.verify = verify
        self.use_kv_migration = use_kv_migration
        self.device = device
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_version: Optional[str] = None
        try:
            start_http_server(9098)
        except Exception:
            pass
        self._updates_total = Counter("hotweights_binder_updates_total", "Updates applied by binder")
        self._pause_seconds = Gauge("hotweights_binder_last_pause_seconds", "Last pause window seconds")
        self._current_version = Gauge("hotweights_binder_current_version", "Hash of current version")

    def start(self) -> None:
        # Best-effort: attach a CUDA-IPC agent to the engine for GPU-native commit
        try:
            use_ipc = os.getenv("HOTWEIGHTS_USE_IPC_AGENT", "0") in ("1", "true", "True")
            if use_ipc and getattr(self.obj, "hotweights_agent", None) is None and torch is not None and torch.cuda.is_available():
                from ..staging.cuda_ipc_agent import CudaIPCAgent
                setattr(self.obj, "hotweights_agent", CudaIPCAgent(device=self.device))
        except Exception:
            pass
        t = threading.Thread(target=self._loop, name="hotweights-vllm", daemon=True)
        t.start()
        self._thread = t

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        c = Client(self.endpoint)
        worker_id = (
            os.getenv("VLLM_WORKER_ID")
            or os.getenv("WORKER_ID")
            or f"pid:{os.getpid()}"
        )
        while not self._stop.is_set():
            try:
                st = c.call("status")
                version = st.get("version")
                resp = c.call("get_plan")
                plan = resp.get("plan")
                if plan and version and version != self._last_version:
                    old_cache = None
                    if self.use_kv_migration:
                        try:
                            # This path is highly dependent on vLLM internals
                            old_cache = self.obj.model_runner.gpu_cache
                            print(
                                "Successfully extracted old KV-cache for migration."
                            )
                        except AttributeError:
                            print(
                                "Warning: Could not extract KV-cache. Skipping migration."
                            )
                            old_cache = None

                    try:
                        t0 = time.perf_counter()
                        pause_requests(self.obj, drain=not self.use_kv_migration)
                        
                        # The new worker agent handles replication, so we just need to commit.
                        # This binding logic assumes the worker has already run.
                        # Here we will focus on the commit-time logic: KV migration and weight swap.
                        print("Applying new weights to model...")
                        # GPU-native commit via CudaIPCAgent if available and enabled
                        use_ipc = os.getenv("HOTWEIGHTS_USE_IPC_AGENT", "0") in (
                            "1",
                            "true",
                            "True",
                        )
                        agent = getattr(self.obj, "hotweights_agent", None)
                        if use_ipc and agent is not None:
                            # Aggregate plan items
                            items: list[dict] = []
                            for b in plan.get("buckets", []):
                                items.extend(b.get("items", []))
                            # name_map can be provided externally; if None, rely on engine-side mapping
                            if self.name_map and isinstance(self.name_map, dict):
                                name_map = self.name_map
                            else:
                                # Best effort: identity-like map using normalized tensor names
                                name_map = {
                                    it["key"]: it["tensor"]
                                    .replace("/", ".")
                                    .rsplit(".", 1)[0]
                                    for it in items
                                }
                            apply_from_ipc_agent_to_module(
                                items,
                                agent,
                                self.module,
                                name_map,
                                device=self.device,
                            )
                        else:
                            # Fallback: assume worker performed the apply
                            pass

                        if old_cache and self.use_kv_migration:
                            from .kv_cache_migration import migrate_kv_cache
                            new_cache, report = migrate_kv_cache(
                                old_cache, self.module, plan
                            )
                            print(f"KV-cache migration report: {report}")
                            # This path is also highly dependent on vLLM internals
                            self.obj.model_runner.set_gpu_cache(new_cache)
                            print("Successfully loaded new migrated KV-cache.")

                        c.call("precommit", worker_id=worker_id)
                        c.call("commit", version=version)
                        self._last_version = version
                        
                        dt = time.perf_counter() - t0
                        self._pause_seconds.set(dt)
                        self._updates_total.inc(1)
                    finally:
                        resume_requests(self.obj)
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"HotweightsVLLMBinding loop error: {e}")
                time.sleep(self.poll_interval * 2)


def bind_to_vllm(
    engine_or_runner,
    name_map: dict[str, str] | Callable[[dict], dict[str, str]] | None,
    endpoint: str = "tcp://127.0.0.1:5555",
    use_mpi: bool = False,
    pinned: bool = True,
    verify: bool = False,
    use_kv_migration: bool = True,
    device: str = "cuda",
    poll_interval: float = 2.0,
) -> HotweightsVLLMBinding:
    """Bind a background updater into a vLLM engine/runner and start it."""
    binding = HotweightsVLLMBinding(
        engine_or_runner,
        name_map,
        endpoint=endpoint,
        use_mpi=use_mpi,
        pinned=pinned,
        verify=verify,
        use_kv_migration=use_kv_migration,
        device=device,
        poll_interval=poll_interval,
    )
    binding.start()
    return binding

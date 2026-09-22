"""Bind hotweights update loop into a vLLM worker process.

This module provides a best-effort binding function that you can call from a
vLLM worker process after the model is instantiated. It polls the coordinator
for a plan+version and applies updates into the worker's torch.nn.Module using
async H2D copies with pinned host buffers.

It avoids importing vLLM directly; instead it introspects the provided object
to find a `.model` attribute or a nested `.model_runner.model`.
"""
from __future__ import annotations

import hashlib
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
from .vllm_ext import (
    apply_from_ipc_agent_to_module,  # noqa: F401  (legacy in-place path)
    atomic_swap_params,
    read_ipc_staged_bytes,
    stage_from_ipc_agent,
    verify_staged_hashes,
)
from .vllm_pause import pause_requests, resume_requests
from ..telemetry.prom import Counter, Gauge, start_http_server


def _version_gauge_value(version: str) -> float:
    """Stable numeric fingerprint of a version string for the gauge."""
    digest = hashlib.sha256(version.encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16))


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
        event_token: Optional[str] = None,
    ) -> None:
        self.obj = obj
        self.module = _extract_module(obj)
        self.name_map = name_map
        self.endpoint = endpoint
        self.use_mpi = use_mpi  # reserved for the legacy host-buffer path
        self.pinned = pinned  # reserved for the legacy host-buffer path
        self.verify = verify
        self.use_kv_migration = use_kv_migration
        self.device = device
        self.poll_interval = poll_interval
        # Shared secret for coordinator mutating RPCs; falls back to env so
        # CLI-launched binders and coordinators agree without code changes.
        self.event_token = event_token or os.getenv("HOTWEIGHTS_COORD_TOKEN")
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_version: Optional[str] = None
        try:
            start_http_server(9098)
        except Exception:
            pass
        self._updates_total = Counter("hotweights_binder_updates_total", "Updates applied by binder")
        self._rejected_total = Counter("hotweights_binder_rejected_total", "Updates rejected by coordinator quorum")
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
        # Register so the coordinator's commit quorum accounts for this binder.
        try:
            c.call(
                "register",
                worker_id=worker_id,
                caps={"role": "vllm-binder", "transport": "cuda_ipc"},
            )
        except Exception as e:
            print(f"HotweightsVLLMBinding: coordinator register failed: {e}")
        while not self._stop.is_set():
            try:
                st = c.call("status")
                version = st.get("version")
                resp = c.call("get_plan")
                plan = resp.get("plan")
                if plan and version and version != self._last_version:
                    self._apply_update(c, worker_id, plan, version)
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"HotweightsVLLMBinding loop error: {e}")
                time.sleep(self.poll_interval * 2)

    def _resolve_name_map(self, plan: dict) -> dict:  # noqa: ANN202
        """Resolve the key -> module-parameter mapping for a plan."""
        if isinstance(self.name_map, dict):
            return self.name_map
        if callable(self.name_map):
            try:
                return dict(self.name_map(plan))
            except Exception:
                pass
        # Best effort: identity-like map using normalized tensor names
        items = [it for b in plan.get("buckets", []) for it in b.get("items", [])]
        return {
            it["key"]: it["tensor"].replace("/", ".").rsplit(".", 1)[0] for it in items
        }

    def _stage_update(self, plan: dict) -> dict:  # noqa: ANN202
        """Stage new weights WITHOUT touching the live module.

        Returns {param_path: new_tensor}. An empty dict means the legacy path:
        a worker process already applied the weights and the binder only
        participates in the commit protocol.
        """
        use_ipc = os.getenv("HOTWEIGHTS_USE_IPC_AGENT", "0") in ("1", "true", "True")
        agent = getattr(self.obj, "hotweights_agent", None)
        if not (use_ipc and agent is not None):
            return {}
        items = [it for b in plan.get("buckets", []) for it in b.get("items", [])]
        if not items:
            return {}
        name_map = self._resolve_name_map(plan)
        return stage_from_ipc_agent(items, agent, self.module, name_map, self.device)

    def _apply_update(self, c, worker_id: str, plan: dict, version: str) -> bool:  # noqa: ANN001, ANN202
        """Run one version update: stage -> verify -> precommit -> commit -> flip.

        Returns True only when the coordinator accepted the commit AND the new
        weights were installed. On any failure, or when the coordinator
        rejects the commit, the live module is left untouched, requests are
        resumed, and ``_last_version`` is NOT advanced (so the next poll
        retries instead of pretending the update landed).
        """
        old_cache = None
        if self.use_kv_migration:
            try:
                # This path is highly dependent on vLLM internals
                old_cache = self.obj.model_runner.gpu_cache
                print("Successfully extracted old KV-cache for migration.")
            except AttributeError:
                print("Warning: Could not extract KV-cache. Skipping migration.")
                old_cache = None

        try:
            t0 = time.perf_counter()
            pause_requests(self.obj, drain=not self.use_kv_migration)

            # 1) Stage new weights into shadow tensors; live params untouched.
            staged = self._stage_update(plan)

            # 2) Optional hash verification of staged tensors before precommit.
            if self.verify and staged:
                agent = getattr(self.obj, "hotweights_agent", None)
                items = [
                    it for b in plan.get("buckets", []) for it in b.get("items", [])
                ]
                nbytes = {it["key"]: int(it["nbytes"]) for it in items}
                verify_staged_hashes(
                    items,
                    lambda key: read_ipc_staged_bytes(agent, key, nbytes[key]),
                )

            # 3) KV-cache migration reads model config only (not weights), so it
            # can run pre-commit; the result is installed only on accept.
            new_cache = None
            if old_cache is not None and self.use_kv_migration:
                from .kv_cache_migration import migrate_kv_cache

                new_cache, report = migrate_kv_cache(old_cache, self.module, plan)
                print(f"KV-cache migration report: {report}")

            # 4) Two-phase commit: honor the coordinator's quorum decision.
            c.call("precommit", worker_id=worker_id, version=version,
                   token=self.event_token)
            resp = c.call("commit", version=version, token=self.event_token) or {}
            if not resp.get("accepted", False):
                print(
                    f"Commit for version {version} not accepted "
                    f"(waiting_for={resp.get('waiting_for')}); staged update discarded, "
                    "live weights untouched."
                )
                self._rejected_total.inc(1)
                return False

            # 5) Accepted: atomically flip to the staged weights, then install
            # the migrated KV-cache while requests are still paused.
            if staged:
                atomic_swap_params(self.module, staged)
                print(f"Swapped in {len(staged)} staged parameters for version {version}.")
            if new_cache is not None:
                try:
                    # This path is also highly dependent on vLLM internals
                    self.obj.model_runner.set_gpu_cache(new_cache)
                    print("Successfully loaded new migrated KV-cache.")
                except AttributeError:
                    print(
                        "Warning: model_runner.set_gpu_cache missing; "
                        "continuing with previous KV-cache."
                    )

            self._last_version = version
            self._current_version.set(_version_gauge_value(version))
            self._updates_total.inc(1)
            dt = time.perf_counter() - t0
            self._pause_seconds.set(dt)
            return True
        finally:
            resume_requests(self.obj)


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
    event_token: Optional[str] = None,
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
        event_token=event_token,
    )
    binding.start()
    return binding

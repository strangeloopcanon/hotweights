"""vLLM WorkerExtension implementation (best-effort).

This class matches the typical WorkerExtension shape used by vLLM and delegates
to hotweights internals. It should be configured with a coordinator endpoint
and an optional name_map function.

Usage:

  from hotweights.adapters.vllm_extension import HotweightsWorkerExtension
  ext = HotweightsWorkerExtension(endpoint="tcp://coord:5555", use_mpi=True, pinned=True, device="cuda")
  # Register `ext` with vLLM worker hooks
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from .vllm_plugin import update_weights_from_coordinator


class HotweightsWorkerExtension:
    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        name_map: Dict[str, str] | Callable[[dict], Dict[str, str]] | None = None,
        use_mpi: bool = False,
        pinned: bool = True,
        verify: bool = False,
        device: str = "cuda",
    ) -> None:
        self.endpoint = endpoint
        self.name_map = name_map
        self.use_mpi = use_mpi
        self.pinned = pinned
        self.verify = verify
        self.device = device
        self._module = None

    # vLLM hook: called when an update begins
    def begin_update(self, version: str, manifest: dict) -> None:  # noqa: ANN001
        _ = (version, manifest)

    # vLLM hook: return a target buffer or descriptor
    def request_buffer(self, tensor: str, shard_rank: int, nbytes: int):  # noqa: ANN001, ANN201
        return {"tensor": tensor, "shard_rank": shard_rank, "nbytes": nbytes}

    # vLLM hook: called when a shard is finalized
    def finalize_shard(self, tensor: str, shard_rank: int, hash_: str) -> None:  # noqa: ANN001
        _ = (tensor, shard_rank, hash_)

    # vLLM hook: called before commit
    def precommit(self) -> None:
        pass

    # vLLM hook: apply and commit; requires a reference to the model module
    def commit(self, version: str) -> None:  # noqa: ANN001
        _ = version

    # hotweights: set the torch.nn.Module target
    def bind_module(self, module) -> None:  # noqa: ANN001
        self._module = module

    # API for vLLM workers to trigger the update explicitly
    def apply_update(self) -> None:
        if self._module is None:
            raise RuntimeError("Module not bound; call bind_module(module) first")
        update_weights_from_coordinator(
            self._module,
            self.name_map,
            endpoint=self.endpoint,
            use_mpi=self.use_mpi,
            pinned=self.pinned,
            verify=self.verify,
            device=self.device,
        )


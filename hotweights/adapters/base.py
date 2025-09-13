"""Adapter protocol for server-plane integrations (vLLM, trainers)."""
from __future__ import annotations

from typing import Protocol, Dict, List


class HotweightsAdapter(Protocol):
    def begin_update(self, version: str, manifest: Dict) -> None:
        ...

    def request_buffer(self, tensor: str, shard_rank: int, nbytes: int):  # noqa: ANN201
        ...

    def finalize_shard(self, tensor: str, shard_rank: int, hash_: str) -> None:
        ...

    def precommit(self) -> None:
        ...

    def commit(self, version: str) -> None:
        ...


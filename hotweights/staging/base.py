"""Staging agent base interfaces."""
from __future__ import annotations

from typing import Protocol


class StagingAgent(Protocol):
    def reserve(self, key: str, nbytes: int) -> memoryview:
        ...

    def write(self, key: str, off: int, buf: memoryview) -> None:
        ...

    def seal(self, key: str) -> str:
        ...

    def read(self, key: str) -> memoryview:
        ...

    def torch_tensor(self, key: str):  # optional
        ...


"""Control-plane client (stub)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Client:
    endpoint: str = "inproc://stub"

    def request(self, method: str, **kwargs):  # noqa: ANN003
        # Placeholder for RPC.
        return {"method": method, "kwargs": kwargs}


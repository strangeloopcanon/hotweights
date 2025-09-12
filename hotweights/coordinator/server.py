"""Control-plane server (stub).

A real implementation would use ZeroMQ to coordinate begin/commit events.
This stub exists to anchor interfaces referenced in the design.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Coordinator:
    version: Optional[str] = None
    registered: Dict[str, dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.registered is None:
            self.registered = {}

    def register(self, worker_id: str, caps: dict) -> dict:
        self.registered[worker_id] = caps
        return {"ok": True, "current_version": self.version}

    def begin(self, version: str, manifest_digest: str) -> dict:
        self.version = version
        return {"event": "begin", "version": version, "digest": manifest_digest}

    def commit(self, version: str) -> dict:
        self.version = version
        return {"event": "commit", "version": version}

    def abort(self, reason: str) -> dict:
        return {"event": "abort", "reason": reason}


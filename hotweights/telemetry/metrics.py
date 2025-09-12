"""Minimal metrics helpers (stub)."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    name: str
    start: float | None = None
    elapsed: float = 0.0

    def __enter__(self):  # noqa: ANN001
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN201
        self.elapsed = time.perf_counter() - (self.start or time.perf_counter())
        return False


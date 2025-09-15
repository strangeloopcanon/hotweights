"""Environment parsing helpers.

Small helpers to consistently parse env vars with sane defaults.
"""
from __future__ import annotations

from typing import Iterable, List
import os


def env_bool(name: str, default: bool = False) -> bool:
    try:
        val = os.getenv(name)
        if val is None:
            return bool(default)
        return val in ("1", "true", "True", "YES", "yes", "on", "On")
    except Exception:
        return bool(default)


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    try:
        v = int(os.getenv(name, str(default)))
    except Exception:
        v = int(default)
    if minimum is not None:
        try:
            v = max(minimum, v)
        except Exception:
            pass
    return v


def env_mb(name: str, default_mb: int, minimum_mb: int | None = 1) -> int:
    v = env_int(name, default_mb, minimum=minimum_mb or 1)
    return int(v) * (1 << 20)


def env_list_int(name: str, default: Iterable[int]) -> List[int]:
    s = os.getenv(name, "")
    if not s:
        return list(default)
    out: List[int] = []
    try:
        for tok in s.split(","):
            tok = tok.strip()
            if not tok:
                continue
            out.append(int(tok))
        return out
    except Exception:
        return list(default)


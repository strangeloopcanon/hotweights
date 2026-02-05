"""Best-effort pause/resume helpers for vLLM engines.

This module avoids importing vLLM; it tries common method names and fields.
"""
from __future__ import annotations

import time
from typing import Any


def _get(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_engine(obj: Any) -> Any | None:
    for path in (
        "engine",
        "executor.engine",
        "model_executor.engine",
        "runner.engine",
    ):
        eng = _get(obj, path)
        if eng is not None:
            return eng
    return obj if hasattr(obj, "__dict__") else None


def pause_requests(obj: Any, timeout_s: float = 5.0, drain: bool = True) -> None:
    eng = find_engine(obj) or obj
    # Try explicit pause methods
    for name in ("pause", "pause_requests", "block_new_requests"):
        if hasattr(eng, name):
            try:
                getattr(eng, name)()
                break
            except Exception:
                pass
    if not drain:
        return
    # Wait for in-flight to drain if we can observe
    deadline = time.time() + timeout_s
    for attr in ("num_active_requests", "active_requests", "running"):
        if hasattr(eng, attr):
            while time.time() < deadline:
                val = getattr(eng, attr)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        break
                try:
                    n = len(val) if not isinstance(val, (int, float)) else int(val)
                except Exception:
                    break
                if n == 0:
                    return
                time.sleep(0.01)
            return


def resume_requests(obj: Any) -> None:
    eng = find_engine(obj) or obj
    for name in ("resume", "resume_requests", "unblock_new_requests"):
        if hasattr(eng, name):
            try:
                getattr(eng, name)()
                break
            except Exception:
                pass

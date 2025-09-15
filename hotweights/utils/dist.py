"""torch.distributed helpers (best-effort).

Provide small utilities to initialize a process group and to create/fetch
subgroups for consumer rank sets. These are defensive and safe to import
when torch.distributed is unavailable.
"""
from __future__ import annotations

from typing import Iterable, Optional


def ensure_pg(backend: Optional[str] = None) -> tuple[bool, Optional[str]]:
    try:
        import torch  # type: ignore
        import torch.distributed as dist  # type: ignore
    except Exception:
        return False, None
    try:
        if getattr(dist, "is_initialized", lambda: False)():
            return True, backend
        # Choose backend if not provided
        use = backend
        if use is None:
            try:
                if getattr(torch.cuda, "is_available", lambda: False)():
                    use = "nccl"
                elif hasattr(torch, "xpu") and getattr(torch.xpu, "is_available", lambda: False)():  # type: ignore[attr-defined]
                    use = "ccl"
                else:
                    use = "gloo"
            except Exception:
                use = "gloo"
        dist.init_process_group(backend=use)
        return True, use
    except Exception:
        return False, backend


def get_subgroup(consumers: Optional[Iterable[int]], cache: dict[frozenset[int], object]) -> object | None:
    try:
        import torch.distributed as dist  # type: ignore
    except Exception:
        return None
    if not consumers:
        return None
    key = frozenset(int(x) for x in consumers)
    g = cache.get(key)
    if g is None:
        try:
            g = dist.new_group(ranks=sorted(key))
            cache[key] = g
        except Exception:
            g = None
    return g


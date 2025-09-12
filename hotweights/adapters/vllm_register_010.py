"""vLLM 0.10.x registration helper.

This attempts to attach Hotweights to vLLM 0.10.x engines by:
  - Binding our WorkerExtension to the model
  - Starting the background binder as a fallback for pause/apply/commit

Usage:
    from hotweights.adapters.vllm_register_010 import register
    ext, binding = register(engine, name_map_fn, endpoint="tcp://coord:5555", use_mpi=True, pinned=True, device="cuda")
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple, Optional

from .vllm_extension import HotweightsWorkerExtension
from .vllm_bind import bind_to_vllm


def _find_model(engine):  # noqa: ANN001
    for path in ("model", "model_runner.model", "engine.model", "engine.model_runner.model"):
        cur = engine
        ok = True
        for p in path.split("."):
            if not hasattr(cur, p):
                ok = False
                break
            cur = getattr(cur, p)
        if ok and cur is not None:
            return cur
    raise AttributeError("Could not locate model module on engine")


def register(
    engine,
    name_map: Dict[str, str] | Callable[[dict], Dict[str, str]] | None,
    endpoint: str = "tcp://127.0.0.1:5555",
    use_mpi: bool = False,
    pinned: bool = True,
    verify: bool = False,
    device: str = "cuda",
) -> Tuple[HotweightsWorkerExtension, object]:
    """Attach both a WorkerExtension and a background binder to a vLLM engine.

    Returns (extension, binding). Either can be used depending on available hooks.
    """
    ext = HotweightsWorkerExtension(endpoint=endpoint, name_map=name_map, use_mpi=use_mpi, pinned=pinned, verify=verify, device=device)
    mod = _find_model(engine)
    ext.bind_module(mod)
    setattr(engine, "hotweights_extension", ext)

    # Always start binder as fallback to manage pause/apply/commit loop
    binding = bind_to_vllm(engine, name_map, endpoint=endpoint, use_mpi=use_mpi, pinned=pinned, verify=verify, device=device)
    return ext, binding


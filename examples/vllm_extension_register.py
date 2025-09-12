"""Example: registering HotweightsWorkerExtension with a vLLM worker.

Note: This is illustrative. Adjust imports to match your vLLM version.
Run inside the vLLM worker process after model construction.
"""
from __future__ import annotations

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine  # type: ignore
except Exception:
    AsyncLLMEngine = None  # type: ignore

from hotweights.adapters.vllm_extension import HotweightsWorkerExtension


def install_extension_on_engine(engine, endpoint: str = "tcp://127.0.0.1:5555"):  # noqa: ANN001
    ext = HotweightsWorkerExtension(endpoint=endpoint, use_mpi=True, pinned=True, device="cuda")
    # Bind module: try common attribute names
    mod = getattr(engine, "model", None) or getattr(getattr(engine, "model_runner", None), "model", None)
    if mod is None:
        raise RuntimeError("Could not locate model module on engine")
    ext.bind_module(mod)
    # Wire into lifecycle if your vLLM version exposes explicit hooks
    # Otherwise, trigger ext.apply_update() from your coordinator/binder when begin/commit occurs.
    setattr(engine, "hotweights_extension", ext)
    return ext


if __name__ == "__main__":
    print("This module contains an example function; import and call install_extension_on_engine(engine)")


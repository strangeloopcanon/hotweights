"""Auto-bind hotweights into vLLM engines by monkeypatching engine init.

This is optional and best-effort: if vLLM is not importable, it no-ops.
When vLLM is available, it patches AsyncLLMEngine and LLMEngine __init__ to
start a background HotweightsVLLMBinding after the engine is constructed.
"""
from __future__ import annotations

from collections.abc import Callable

from .vllm_bind import bind_to_vllm


def install_autobind(
    name_map: dict[str, str] | Callable[[dict], dict[str, str]] | None,
    endpoint: str = "tcp://127.0.0.1:5555",
    use_mpi: bool = False,
    pinned: bool = True,
    verify: bool = False,
    device: str = "cuda",
    poll_interval: float = 2.0,
) -> bool:
    """Install auto-binding into vLLM engines if available.

    Returns True if vLLM engines were patched, False otherwise.
    """
    try:
        from vllm.engine.async_llm_engine import AsyncLLMEngine as async_engine_cls  # type: ignore
    except Exception:
        async_engine_cls = None  # type: ignore
    try:
        from vllm.engine.llm_engine import LLMEngine as llm_engine_cls  # type: ignore
    except Exception:
        llm_engine_cls = None  # type: ignore

    patched = False

    def _patch(cls: object) -> None:
        nonlocal patched
        if cls is None:
            return
        if getattr(cls, "_hotweights_patched", False):
            patched = True
            return
        orig_init = cls.__init__  # type: ignore[attr-defined]

        def _init(self, *args: object, **kwargs: object) -> None:  # noqa: ANN001
            orig_init(self, *args, **kwargs)  # type: ignore[misc]
            try:
                # Avoid double-binding
                if getattr(self, "_hotweights_binding", None) is None:
                    b = bind_to_vllm(
                        self,
                        name_map,
                        endpoint=endpoint,
                        use_mpi=use_mpi,
                        pinned=pinned,
                        verify=verify,
                        device=device,
                        poll_interval=poll_interval,
                    )
                    self._hotweights_binding = b  # type: ignore[attr-defined]
            except Exception:
                # best-effort; never fail engine init
                pass

        cls.__init__ = _init  # type: ignore[assignment]
        cls._hotweights_patched = True  # type: ignore[attr-defined]
        patched = True

    _patch(async_engine_cls)
    _patch(llm_engine_cls)
    return patched

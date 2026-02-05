from __future__ import annotations

import sys
import types

import hotweights.adapters.vllm_bind as vllm_bind
from hotweights.adapters.vllm_auto import install_autobind


def test_install_autobind_patches_fake_engines(monkeypatch):
    # Create a fake vllm module hierarchy
    vllm = types.ModuleType("vllm")
    engine = types.ModuleType("vllm.engine")
    async_mod = types.ModuleType("vllm.engine.async_llm_engine")
    llm_mod = types.ModuleType("vllm.engine.llm_engine")

    class AsyncLLMEngine:
        def __init__(self):
            self.initialized = True

    class LLMEngine:
        def __init__(self):
            self.initialized = True

    async_mod.AsyncLLMEngine = AsyncLLMEngine
    llm_mod.LLMEngine = LLMEngine
    engine.async_llm_engine = async_mod
    engine.llm_engine = llm_mod
    vllm.engine = engine

    sys.modules["vllm"] = vllm
    sys.modules["vllm.engine"] = engine
    sys.modules["vllm.engine.async_llm_engine"] = async_mod
    sys.modules["vllm.engine.llm_engine"] = llm_mod

    patched = install_autobind(name_map=None, endpoint="tcp://127.0.0.1:5555")
    assert patched is True

    # instantiating should not raise and should set a binding attribute best-effort
    a = async_mod.AsyncLLMEngine()
    l = llm_mod.LLMEngine()
    # binding attribute may be missing due to underlying bind errors, but init should not fail
    assert getattr(a, "initialized", False) is True
    assert getattr(l, "initialized", False) is True


def test_bind_to_vllm_accepts_compat_kwargs(monkeypatch):
    class Dummy:
        model = object()

    monkeypatch.setattr(vllm_bind.HotweightsVLLMBinding, "start", lambda self: None)
    binding = vllm_bind.bind_to_vllm(
        Dummy(),
        None,
        endpoint="tcp://127.0.0.1:5555",
        use_mpi=True,
        pinned=False,
        verify=True,
    )
    assert isinstance(binding, vllm_bind.HotweightsVLLMBinding)

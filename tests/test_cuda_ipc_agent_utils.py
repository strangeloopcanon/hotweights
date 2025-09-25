from __future__ import annotations

from types import SimpleNamespace

import hotweights.staging.cuda_ipc_agent as cia


def _blank_agent() -> cia.CudaIPCAgent:
    agent = cia.CudaIPCAgent.__new__(cia.CudaIPCAgent)
    agent._gds_capable = True
    agent._pinned_granularity = 8
    agent._pinned_pool = {}
    return agent


def test_should_use_gds_defaults(monkeypatch):
    agent = _blank_agent()
    monkeypatch.delenv("HOTWEIGHTS_USE_GDS", raising=False)
    monkeypatch.delenv("HOTWEIGHTS_DISABLE_GDS", raising=False)
    assert agent._should_use_gds() is True

    agent._gds_capable = False
    assert agent._should_use_gds() is False


def test_should_use_gds_env_overrides(monkeypatch):
    agent = _blank_agent()
    monkeypatch.setenv("HOTWEIGHTS_DISABLE_GDS", "1")
    assert agent._should_use_gds() is False

    monkeypatch.delenv("HOTWEIGHTS_DISABLE_GDS", raising=False)
    monkeypatch.setenv("HOTWEIGHTS_USE_GDS", "0")
    assert agent._should_use_gds() is False

    monkeypatch.setenv("HOTWEIGHTS_USE_GDS", "true")
    assert agent._should_use_gds() is True


def test_get_pinned_reuses_aligned_buffers(monkeypatch):
    calls: list[int] = []

    class _FakeTensor:
        def __init__(self, size: int) -> None:
            self.size = size

        def __getitem__(self, _slice):  # noqa: ANN001
            return self

    class _FakeTorch(SimpleNamespace):
        def empty(self, size: int, **_kwargs):  # noqa: ANN003
            calls.append(size)
            return _FakeTensor(size)

    fake_torch = _FakeTorch(uint8="uint8")
    monkeypatch.setattr(cia, "torch", fake_torch)

    agent = _blank_agent()

    buf_small = agent._get_pinned(3)
    assert calls == [8]
    assert buf_small.size == 8

    buf_again = agent._get_pinned(5)
    assert calls == [8]
    assert buf_again is buf_small

    buf_large = agent._get_pinned(20)
    assert calls == [8, 24]
    assert buf_large.size == 24

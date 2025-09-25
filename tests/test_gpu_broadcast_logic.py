from __future__ import annotations

from types import SimpleNamespace

import hotweights.transport.gpu_broadcast as gb


class _FakeTensor:
    def __init__(self, size: int, device: str = "cuda") -> None:
        self.size = size
        self.device = device

    def view(self, *_args, **_kwargs):  # noqa: ANN002
        return self

    def copy_(self, *_args, **_kwargs):  # noqa: ANN002
        return self


class _FakeTorch:
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    cuda = _FakeCuda()
    uint8 = "uint8"

    @staticmethod
    def empty(size: int, **_kwargs) -> _FakeTensor:  # noqa: ANN003
        return _FakeTensor(size)


class _FakeDist:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int, object | None]] = []
        self.group = SimpleNamespace(WORLD=object())  # type: ignore[attr-defined]

    @staticmethod
    def is_initialized() -> bool:
        return True

    def broadcast(self, tensor: object, src: int, group: object | None = None) -> None:
        self.calls.append((tensor, src, group))


class _StubAgent:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.shared: list[tuple[int, object]] = []
        self.scattered: list[int] = []

    def share_from_gpu(self, bucket_id: int, tensor: object) -> None:
        self.shared.append((bucket_id, tensor))

    def scatter_from_shared(self, bucket_id: int, _items):  # noqa: ANN001
        self.scattered.append(bucket_id)

    def cleanup(self) -> None:
        return None


def _install_minimal_stubs(monkeypatch) -> _FakeDist:  # noqa: ANN001
    fake_dist = _FakeDist()
    monkeypatch.setenv("HOTWEIGHTS_BCAST_AUTOTUNE", "0")
    monkeypatch.setenv("HOTWEIGHTS_BCAST_PREFETCH", "0")
    monkeypatch.setattr(gb, "torch", _FakeTorch())
    monkeypatch.setattr(gb, "dist", fake_dist)
    monkeypatch.setattr(gb, "ensure_pg", lambda backend=None: (True, backend or "nccl"))
    return fake_dist


def test_gpu_broadcast_uses_world_rank_for_src(monkeypatch):
    fake_dist = _install_minimal_stubs(monkeypatch)
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "5")

    def fake_get_subgroup(consumers, cache):  # noqa: ANN001
        if consumers:
            key = frozenset(consumers)
            cache[key] = cache.get(key, object())
            return cache[key]
        return None

    monkeypatch.setattr(gb, "get_subgroup", fake_get_subgroup)
    monkeypatch.setattr(
        gb.GpuBroadcastTransport,
        "_prepare_dev_on_root",
        lambda self, bucket: _FakeTensor(int(bucket.get("size", 0)), device=self.agent.device),  # noqa: ANN401
    )

    agent = _StubAgent()
    transport = gb.GpuBroadcastTransport(agent=agent, backend="nccl")

    plan = {"buckets": [{"bucket_id": 1, "size": 16, "items": [], "consumer_ranks": [2, 4]}]}
    transport.replicate(plan)

    assert fake_dist.calls, "broadcast not invoked"
    _tensor, src, _group = fake_dist.calls[0]
    assert src == 2  # world rank of root


def test_gpu_broadcast_world_fallback_includes_all_ranks(monkeypatch):
    fake_dist = _install_minimal_stubs(monkeypatch)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "5")
    monkeypatch.setattr(gb, "get_subgroup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gb.GpuBroadcastTransport,
        "_prepare_dev_on_root",
        lambda self, bucket: _FakeTensor(int(bucket.get("size", 0)), device=self.agent.device),  # noqa: ANN401
    )

    agent = _StubAgent()
    transport = gb.GpuBroadcastTransport(agent=agent, backend="nccl")

    plan = {"buckets": [{"bucket_id": 2, "size": 8, "items": [], "consumer_ranks": [2, 4]}]}
    transport.replicate(plan)

    assert fake_dist.calls, "non-consumer rank must still participate when subgroup missing"
    assert agent.shared == []  # non-consumer should not stage data

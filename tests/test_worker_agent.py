from __future__ import annotations

from hotweights.worker.agent import WorkerConfig, run_worker


class _FakeTransport:
    def __init__(self) -> None:
        self.replicated = False

    def replicate(self, plan: dict) -> None:
        _ = plan
        self.replicated = True


class _FakeClientNoSub:
    def __init__(self) -> None:
        self._status_idx = 0
        self.calls: list[str] = []

    def call(self, method: str, **kwargs):  # noqa: ANN003, ANN201
        _ = kwargs
        self.calls.append(method)
        if method == "get_plan":
            return {"plan": {"version": "v1", "buckets": []}}
        if method == "status":
            states = [
                {"state": "precommit", "version": "v1"},
                {"state": "committed", "version": "v1"},
            ]
            out = states[min(self._status_idx, len(states) - 1)]
            self._status_idx += 1
            return out
        return {"ok": True}


class _FakeClientSub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, method: str, **kwargs):  # noqa: ANN003, ANN201
        _ = kwargs
        self.calls.append(method)
        if method == "get_plan":
            return {"plan": {"version": "v1", "buckets": []}}
        if method == "status":
            return {"state": "precommit", "version": "v1"}
        return {"ok": True}


class _FakeSubscriber:
    def __init__(self, events: list[tuple[str, dict]]) -> None:
        self._events = events
        self._idx = 0

    def recv(self):  # noqa: ANN201
        out = self._events[min(self._idx, len(self._events) - 1)]
        self._idx += 1
        return out


def _install_worker_dummies(monkeypatch, client, subscriber=None) -> _FakeTransport:  # noqa: ANN001
    import hotweights.worker.agent as wa

    tr = _FakeTransport()
    monkeypatch.setattr(wa, "Client", lambda endpoint: client)
    if subscriber is not None:
        monkeypatch.setattr(wa, "Subscriber", lambda endpoint: subscriber)
    monkeypatch.setattr(wa, "CudaIPCMetrics", lambda rank: object())
    monkeypatch.setattr(wa, "CudaIPCAgent", lambda device, metrics: object())
    monkeypatch.setattr(
        wa, "CudaIPCTransport", lambda agent, metrics, coord_endpoint: tr
    )
    monkeypatch.setattr(wa, "start_http_server", lambda port: None)
    monkeypatch.setattr(wa.time, "sleep", lambda _s: None)
    return tr


def test_worker_runs_without_subscriber(monkeypatch) -> None:
    client = _FakeClientNoSub()
    tr = _install_worker_dummies(monkeypatch, client)
    cfg = WorkerConfig(endpoint="tcp://127.0.0.1:5555", device="cuda", use_sub=False)
    rc = run_worker(cfg)
    assert rc == 0
    assert tr.replicated is True
    assert "status" in client.calls


def test_worker_validates_event_token(monkeypatch) -> None:
    client = _FakeClientSub()
    sub = _FakeSubscriber(
        [
            ("begin", {"tok": "bad"}),
            ("begin", {"tok": "secret"}),
            ("commit", {"version": "v1", "accepted": True, "tok": "bad"}),
            ("commit", {"version": "v1", "accepted": True, "tok": "secret"}),
        ]
    )
    tr = _install_worker_dummies(monkeypatch, client, subscriber=sub)
    cfg = WorkerConfig(
        endpoint="tcp://127.0.0.1:5555",
        device="cuda",
        use_sub=True,
        event_token="secret",
    )
    rc = run_worker(cfg)
    assert rc == 0
    assert tr.replicated is True

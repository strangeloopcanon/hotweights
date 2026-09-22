from __future__ import annotations

import json
from pathlib import Path

from hotweights.cli import _cmd_replicate


class _FakeClient:
    def __init__(self, endpoint: str):  # noqa: ANN001
        self.endpoint = endpoint
        self.calls = []

    def call(self, method: str, **kwargs):  # noqa: ANN003
        self.calls.append((method, kwargs))
        # return a dummy ok payload
        return {"ok": True, "method": method, "args": kwargs}


def test_cli_replicate_forwards_token(monkeypatch, tmp_path: Path):
    # Create a minimal plan
    plan = {"version": "v1", "bucket_bytes": 0, "total_bytes": 0, "buckets": []}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    # Monkeypatch Client to our fake
    import hotweights.coordinator.zmq_client as zc

    fake = _FakeClient("tcp://x")
    monkeypatch.setattr(zc, "Client", lambda ep: fake)

    # Prepare args namespace
    class A:  # simple shim
        plan = str(plan_path)
        commit = False
        device = "cpu"
        coord_endpoint = "tcp://127.0.0.1:5555"
        mpi = False
        ucx = False
        verify = False
        window = 2
        group = None
        mpi_chunk_mb = 0
        manifest_next = None

    # Set token and call
    monkeypatch.setenv("HOTWEIGHTS_COORD_TOKEN", "secret")
    assert _cmd_replicate(A()) == 0
    # Replicate now waits for precommit quorum between begin and commit.
    methods = [m for m, _ in fake.calls]
    assert methods == ["submit_plan", "begin", "status", "commit"]
    # Verify that token was forwarded on all mutating calls
    for m, kw in fake.calls:
        if m in ("submit_plan", "begin", "commit", "abort"):
            assert kw.get("token") == "secret", m



def test_cli_replicate_aborts_on_quorum_timeout(monkeypatch, tmp_path: Path):
    # Workers are registered but never precommit: replicate must abort
    # instead of firing a doomed commit.
    plan = {"version": "v1", "bucket_bytes": 0, "total_bytes": 0, "buckets": []}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    import hotweights.coordinator.zmq_client as zc

    class _QuorumFake(_FakeClient):
        def call(self, method: str, **kwargs):  # noqa: ANN003
            self.calls.append((method, kwargs))
            if method == "status":
                return {"workers": ["w1"], "precommit_acks": []}
            if method == "commit":
                return {"accepted": False, "waiting_for": ["w1"]}
            return {"ok": True}

    fake = _QuorumFake("tcp://x")
    monkeypatch.setattr(zc, "Client", lambda ep: fake)
    monkeypatch.setenv("HOTWEIGHTS_QUORUM_TIMEOUT", "0")

    class A:  # simple shim
        plan = str(plan_path)
        commit = False
        device = "cpu"
        coord_endpoint = "tcp://127.0.0.1:5555"
        mpi = False
        ucx = False
        verify = False
        window = 2
        group = None
        mpi_chunk_mb = 0
        manifest_next = None

    assert _cmd_replicate(A()) == 1
    methods = [m for m, _ in fake.calls]
    assert "commit" not in methods
    assert methods[-1] == "abort"
    assert fake.calls[-1][1].get("reason") == "quorum timeout"

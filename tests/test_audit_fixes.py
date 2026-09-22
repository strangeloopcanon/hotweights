"""Regression tests for the audit follow-up fixes (findings 3, 4, 7, 8).

- HA ``ack_handle`` must not hide the handle from other ranks on the node.
- HA control plane implements ``get_plan`` / ``heartbeat`` used by the
  worker agent, binder, and plugin; ``status`` reports precommit acks.
- The ZMQ coordinator evicts version handles on commit/abort/begin instead
  of leaking a full model copy per update.
- ``CudaIPCTransport._receive_handle`` raises ``TimeoutError`` instead of
  hanging forever when a handle never arrives.
"""
from __future__ import annotations

import asyncio

import pytest

from hotweights.coordinator.ha_control_plane import (
    DistributedKVStoreMock,
    HAControlPlane,
)
from hotweights.coordinator.zmq_server import (
    State,
    _evict_superseded_handles,
    _evict_version_handles,
)


def _make_plane() -> HAControlPlane:
    plane = HAControlPlane.__new__(HAControlPlane)
    plane.instance_id = "test-1"
    plane.kv = DistributedKVStoreMock()
    plane.is_leader = True
    plane.handle_ttl = 30.0
    plane._heartbeats = {}
    plane._publish = lambda topic, payload: None  # noqa: SLF001, E731
    plane._log = _Log()
    plane.m_handles_posted = _NoopMetric()
    plane.m_handles_fetched = _NoopMetric()
    plane.m_handles_acked = _NoopMetric()
    plane.m_handles_expired = _NoopMetric()
    plane.g_handles_active = _NoopMetric()
    return plane


class _NoopMetric:
    def inc(self, *_a, **_k) -> None:
        pass

    def set(self, *_a, **_k) -> None:
        pass


class _Log:
    def info(self, *_a, **_k) -> None:
        pass

    def warning(self, *_a, **_k) -> None:
        pass

    def debug(self, *_a, **_k) -> None:
        pass


def _post(plane: HAControlPlane, bucket: int = 0, node: str = "n1") -> None:
    resp = plane.handle_request(
        "post_handle",
        {
            "bucket_id": bucket,
            "handle": b"raw-handle-bytes",
            "version": "v1",
            "node": node,
        },
    )
    assert resp["ok"] is True


# --- finding 3: ack must not clobber the handle ---------------------------


def test_ack_does_not_hide_handle_from_other_ranks() -> None:
    plane = _make_plane()
    _post(plane)
    first = plane.handle_request(
        "get_handle", {"bucket_id": 0, "version": "v1", "node": "n1"}
    )
    assert first["handle"] is not None

    ack = plane.handle_request(
        "ack_handle",
        {"bucket_id": 0, "version": "v1", "node": "n1", "worker_id": "w1"},
    )
    assert ack["ok"] is True
    assert "w1" in ack["acked"]

    # A second rank on the same node must still fetch the handle.
    second = plane.handle_request(
        "get_handle", {"bucket_id": 0, "version": "v1", "node": "n1"}
    )
    assert second["handle"] is not None
    assert second["handle"] == first["handle"]

    # Acks accumulate per worker instead of replacing the handle entry.
    ack2 = plane.handle_request(
        "ack_handle",
        {"bucket_id": 0, "version": "v1", "node": "n1", "worker_id": "w2"},
    )
    assert sorted(ack2["acked"]) == ["w1", "w2"]


# --- finding 4: get_plan / heartbeat / precommit acks in status ------------


def test_get_plan_returns_submitted_plan() -> None:
    plane = _make_plane()
    plan = {"version": "v9", "buckets": [{"id": 0}]}
    assert plane.handle_request("submit_plan", {"plan": plan})["ok"] is True
    assert plane.handle_request("get_plan", {})["plan"] == plan


def test_heartbeat_is_recorded() -> None:
    plane = _make_plane()
    assert plane.handle_request("heartbeat", {"worker_id": "w3"}) == {"ok": True}
    assert "w3" in plane._heartbeats  # noqa: SLF001


def test_status_reports_precommit_acks() -> None:
    plane = _make_plane()
    plane.handle_request("register", {"worker_id": "w1", "caps": {}})
    plane.handle_request("register", {"worker_id": "w2", "caps": {}})
    plane.handle_request("begin", {"version": "v1"})
    plane.handle_request("precommit", {"worker_id": "w1", "version": "v1"})
    st = plane.handle_request("status", {})
    assert st["workers"] == ["w1", "w2"] or sorted(st["workers"]) == ["w1", "w2"]
    assert st["precommit_acks"] == ["w1"]


# --- finding 7: handle eviction ---------------------------------------------


def _state_with_handles() -> State:
    st = State()
    st.handles = {
        "v1": {0: {"n1": {"handle": "x"}}},
        "v2": {0: {"n1": {"handle": "y"}}},
    }
    return st


def test_evict_version_handles_drops_only_that_version() -> None:
    st = _state_with_handles()
    _evict_version_handles(st, "v1")
    assert "v1" not in st.handles
    assert "v2" in st.handles


def test_evict_version_handles_none_is_noop() -> None:
    st = _state_with_handles()
    _evict_version_handles(st, None)
    assert set(st.handles) == {"v1", "v2"}


def test_evict_superseded_handles_keeps_current_version() -> None:
    st = _state_with_handles()
    _evict_superseded_handles(st, "v2")
    assert set(st.handles) == {"v2"}


# --- finding 8: _receive_handle times out instead of hanging ---------------


def _make_transport():
    from hotweights.transport.cuda_ipc import CudaIPCTransport

    t = CudaIPCTransport.__new__(CudaIPCTransport)
    t._client = None
    t._handle_token = None
    t._handle_scope = "global"
    t._node_id = "test-node"
    t._handle_timeout = 0.05
    t._handle_registry = {}
    return t


def test_receive_handle_local_registry_hit() -> None:
    t = _make_transport()
    t._handle_registry[3] = b"handle-bytes"
    assert asyncio.run(t._receive_handle(3)) == b"handle-bytes"


def test_receive_handle_missing_raises_timeout() -> None:
    t = _make_transport()
    with pytest.raises(TimeoutError):
        asyncio.run(t._receive_handle(999))

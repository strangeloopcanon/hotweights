"""Tests for the HA control plane's commit protocol.

The HA backend must implement the same safety contract as the ZeroMQ
coordinator:

- ``commit`` is accepted only when every registered worker precommitted the
  version (quorum), and the advertised version never advances on rejection;
- ``status`` exposes the version/state keys the worker agent polls on;
- IPC handles are node-scoped so per-node leaders cannot clobber each other.
"""
from __future__ import annotations

import pytest

from hotweights.coordinator.ha_control_plane import (
    DistributedKVStoreMock,
    HAControlPlane,
)


def _make_plane() -> tuple[HAControlPlane, list]:
    plane = HAControlPlane.__new__(HAControlPlane)
    plane.instance_id = "test-1"
    plane.kv = DistributedKVStoreMock()
    plane.is_leader = True
    plane.handle_ttl = 30.0
    published: list = []
    plane._publish = lambda topic, payload: published.append((topic, payload))  # noqa: SLF001
    plane._log = types_log()
    plane.m_handles_posted = _NoopMetric()
    plane.m_handles_fetched = _NoopMetric()
    plane.m_handles_acked = _NoopMetric()
    plane.m_handles_expired = _NoopMetric()
    plane.g_handles_active = _NoopMetric()
    return plane, published


class _NoopMetric:
    def inc(self, *_a, **_k) -> None:
        pass

    def set(self, *_a, **_k) -> None:
        pass


def types_log():  # noqa: ANN201
    class _Log:
        def info(self, *_a, **_k) -> None:
            pass

        def warning(self, *_a, **_k) -> None:
            pass

        def debug(self, *_a, **_k) -> None:
            pass

    return _Log()


def _register(plane: HAControlPlane, *worker_ids: str) -> None:
    for wid in worker_ids:
        resp = plane.handle_request("register", {"worker_id": wid, "caps": {}})
        assert resp.get("ok") is True


def test_ha_commit_requires_full_quorum() -> None:
    plane, published = _make_plane()
    _register(plane, "w1", "w2")

    assert plane.handle_request("begin", {"version": "v1"})["event"] == "begin"
    assert plane.handle_request("precommit", {"worker_id": "w1", "version": "v1"})["acks"] == 1

    resp = plane.handle_request("commit", {"version": "v1"})
    assert resp["accepted"] is False
    assert resp["waiting_for"] == ["w2"]
    assert resp["acks"] == 1
    assert resp["total_workers"] == 2
    assert plane.handle_request("status", {})["state"] != "committed"

    assert plane.handle_request("precommit", {"worker_id": "w2", "version": "v1"})["acks"] == 2
    resp = plane.handle_request("commit", {"version": "v1"})
    assert resp["accepted"] is True
    assert resp["waiting_for"] == []
    st = plane.handle_request("status", {})
    assert st["state"] == "committed"
    assert st["version"] == "v1"
    topics = [t for t, _ in published]
    assert "commit" in topics


def test_ha_rejected_commit_does_not_advance_version() -> None:
    plane, _ = _make_plane()
    _register(plane, "w1", "w2")
    plane.handle_request("begin", {"version": "v1"})
    plane.handle_request("precommit", {"worker_id": "w1", "version": "v1"})

    # Stale/foreign commit for v2 while v1 never reached quorum.
    resp = plane.handle_request("commit", {"version": "v2"})
    assert resp["accepted"] is False
    st = plane.handle_request("status", {})
    assert st["version"] == "v1", "unaccepted commit must not clobber the version"
    assert st["state"] != "committed"


def test_ha_abort_is_supported() -> None:
    plane, published = _make_plane()
    _register(plane, "w1")
    plane.handle_request("begin", {"version": "v1"})
    resp = plane.handle_request("abort", {"reason": "bad weights"})
    assert resp["event"] == "abort"
    assert plane.handle_request("status", {})["state"] == "aborted"
    assert ("abort", resp) in published


def test_ha_status_exposes_worker_poll_keys() -> None:
    plane, _ = _make_plane()
    _register(plane, "w1")
    plane.handle_request("begin", {"version": "v7"})
    st = plane.handle_request("status", {})
    assert st["version"] == "v7"
    assert st["state"] == "begun"
    assert st["workers"] == ["w1"]


def test_ha_handles_are_node_scoped() -> None:
    plane, _ = _make_plane()
    assert plane.handle_request(
        "post_handle",
        {"version": "v1", "bucket_id": 0, "node": "node-a",
         "handle": b"handle-a", "sig": "sig-a"},
    )["ok"] is True
    assert plane.handle_request(
        "post_handle",
        {"version": "v1", "bucket_id": 0, "node": "node-b",
         "handle": b"handle-b", "sig": "sig-b"},
    )["ok"] is True

    got_a = plane.handle_request(
        "get_handle", {"version": "v1", "bucket_id": 0, "node": "node-a"}
    )
    got_b = plane.handle_request(
        "get_handle", {"version": "v1", "bucket_id": 0, "node": "node-b"}
    )
    assert got_a["handle"] == b"handle-a"
    assert got_b["handle"] == b"handle-b"

    # Acking node-a must not clear node-b's handle.
    assert plane.handle_request(
        "ack_handle", {"version": "v1", "bucket_id": 0, "node": "node-a"}
    )["ok"] is True
    still_b = plane.handle_request(
        "get_handle", {"version": "v1", "bucket_id": 0, "node": "node-b"}
    )
    assert still_b["handle"] == b"handle-b"


def test_ha_non_leader_rejects_writes() -> None:
    plane, _ = _make_plane()
    plane.is_leader = False
    resp = plane.handle_request("commit", {"version": "v1"})
    assert "error" in resp
    # Reads still work.
    assert plane.handle_request("status", {})["state"] == "idle"

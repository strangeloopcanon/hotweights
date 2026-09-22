"""Tests for coordinator commit acceptance semantics.

A commit must only advance the advertised version when every registered
worker has precommitted. A rejected commit leaves the current version (and
state) untouched so workers never observe a version that was not accepted.
"""
from __future__ import annotations

from hotweights.coordinator.zmq_server import State, _evaluate_commit


def _state(workers: list[str], acked: list[str], version: str | None = "v1") -> State:
    st = State()
    st.workers = {w: {} for w in workers}
    st.precommit_acks = {w: True for w in acked}
    st.version = version
    st.state = "precommit"
    return st


def test_commit_accepted_when_all_workers_acked() -> None:
    st = _state(["w1", "w2"], ["w1", "w2"])
    payload = _evaluate_commit(st, "v2")
    assert payload["accepted"] is True
    assert payload["waiting_for"] == []
    assert payload["version"] == "v2"
    assert st.version == "v2"
    assert st.state == "committed"


def test_commit_rejected_when_worker_missing() -> None:
    st = _state(["w1", "w2"], ["w1"])
    payload = _evaluate_commit(st, "v2")
    assert payload["accepted"] is False
    assert payload["waiting_for"] == ["w2"]
    # The advertised version must NOT advance on rejection.
    assert st.version == "v1"
    assert payload["version"] == "v1"
    assert st.state == "precommit"


def test_commit_with_no_registered_workers_is_vacuously_accepted() -> None:
    st = _state([], [])
    payload = _evaluate_commit(st, "v2")
    assert payload["accepted"] is True
    assert st.version == "v2"

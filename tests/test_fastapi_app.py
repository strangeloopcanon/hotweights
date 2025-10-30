from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from hotweights.coordinator.fastapi_app import build_app


def _make_client() -> TestClient:
    return TestClient(build_app())


def _manifests() -> tuple[dict, dict]:
    prev = {
        "model_id": "demo",
        "version": "v0",
        "tensors": [
            {
                "name": "layer0.weight",
                "dtype": "float32",
                "shape": [1, 1],
                "shards": [
                    {
                        "rank": 0,
                        "bytes": 4,
                        "hash": "prev-hash",
                        "uri": "file:///tmp/prev.bin",
                    }
                ],
            }
        ],
    }
    nxt = {
        "model_id": "demo",
        "version": "v1",
        "tensors": [
            {
                "name": "layer0.weight",
                "dtype": "float32",
                "shape": [1, 1],
                "shards": [
                    {
                        "rank": 0,
                        "bytes": 4,
                        "hash": "next-hash",
                        "uri": "file:///tmp/next.bin",
                    }
                ],
            }
        ],
    }
    return prev, nxt


def test_fastapi_health_and_coordinator_endpoints():
    client = _make_client()

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

    resp = client.post("/coordinator/register", json={"worker_id": "worker-1", "caps": {"gpu": 1}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    resp = client.post("/coordinator/begin", json={"version": "v1", "manifest_digest": "digest"})
    assert resp.status_code == 200
    assert resp.json()["version"] == "v1"

    resp = client.post("/coordinator/commit", json={"version": "v1"})
    assert resp.status_code == 200
    assert resp.json()["event"] == "commit"

    resp = client.get("/coordinator/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "v1"
    assert body["registered_count"] == 1
    assert "worker-1" in body["registered"]


def test_fastapi_plan_round_trip():
    client = _make_client()
    prev, nxt = _manifests()

    resp = client.post("/plan", json={"prev": prev, "next": nxt, "bucket_mb": 1})
    assert resp.status_code == 200
    plan = resp.json()["plan"]
    assert plan["version"] == "v1"
    assert plan["buckets"]

    verify_payload = {"plan": plan, "require_consumers": False}
    resp = client.post("/plan/verify", json=verify_payload)
    assert resp.status_code == 200
    report = resp.json()["report"]
    assert "problems" in report

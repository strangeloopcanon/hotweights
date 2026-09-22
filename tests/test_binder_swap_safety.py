"""Swap-safety tests for the vLLM binder.

The nightmare scenario is a weight update that is half-applied when the
coordinator rejects the commit: the model would serve with a mix of old and
new weights. These tests pin the safe behavior:

- new weights are staged without touching the live module,
- the coordinator's ``accepted`` flag is honored,
- a rejected commit (or a hash mismatch) leaves the module untouched and
  does not advance ``_last_version``,
- the event token is forwarded on mutating RPCs.
"""
from __future__ import annotations

import hashlib
import types

import pytest

import hotweights.adapters.vllm_bind as vb
from hotweights.adapters.vllm_ext import SwapError, verify_staged_hashes


class _FakeClient:
    def __init__(self, commit_resp: dict, token: str | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._commit_resp = commit_resp
        self._token = token

    def call(self, method: str, **kwargs):  # noqa: ANN003, ANN201
        self.calls.append((method, dict(kwargs)))
        if method == "status":
            return {"version": "v2", "state": "begun"}
        if method == "get_plan":
            return {"plan": _PLAN, "digest": "d"}
        if method == "commit":
            return dict(self._commit_resp)
        return {"ok": True}


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


_PAYLOAD = b"\x01\x02\x03\x04" * 64
_PLAN = {
    "version": "v2",
    "bucket_bytes": len(_PAYLOAD),
    "total_bytes": len(_PAYLOAD),
    "buckets": [
        {
            "bucket_id": 0,
            "size": len(_PAYLOAD),
            "items": [
                {
                    "tensor": "w",
                    "key": "w:0",
                    "nbytes": len(_PAYLOAD),
                    "hash": f"sha256:{_sha256_hex(_PAYLOAD)}",
                    "uri": "file:///tmp/w.bin",
                    "dtype": "uint8",
                    "shape": [len(_PAYLOAD)],
                    "offset": 0,
                }
            ],
        }
    ],
}


_USE_DEFAULT_MAP = object()


def _make_binding(monkeypatch, name_map=_USE_DEFAULT_MAP, **kw) -> vb.HotweightsVLLMBinding:
    monkeypatch.setattr(vb, "start_http_server", lambda port: None)
    obj = types.SimpleNamespace()
    obj.model = types.SimpleNamespace()  # _extract_module finds .model
    obj.model_runner = types.SimpleNamespace(
        gpu_cache=["old-cache"],
        set_gpu_cache=lambda c: setattr(obj.model_runner, "installed_cache", c),
    )
    b = vb.HotweightsVLLMBinding(
        obj,
        {"w:0": "weight"} if name_map is _USE_DEFAULT_MAP else name_map,
        endpoint="tcp://127.0.0.1:5555",
        use_kv_migration=False,  # keep cache logic out of these tests
        event_token="secret",
        **kw,
    )
    # Silence pause/resume; record calls instead.
    calls = {"paused": 0, "resumed": 0}
    monkeypatch.setattr(vb, "pause_requests", lambda *a, **k: calls.update(paused=1))
    monkeypatch.setattr(vb, "resume_requests", lambda *a, **k: calls.update(resumed=1))
    b._pause_calls = calls  # type: ignore[attr-defined]
    return b


def _install_staging(monkeypatch, staged: dict) -> dict:
    """Patch the torch-dependent staging seam; return the swap-call record."""
    monkeypatch.setenv("HOTWEIGHTS_USE_IPC_AGENT", "1")
    record: dict = {}

    def fake_stage(items, agent, module, name_map, device="cuda"):  # noqa: ANN001, ANN202
        record["staged"] = dict(staged)
        return dict(staged)

    def fake_swap(module, staged_map):  # noqa: ANN001
        record["swapped"] = dict(staged_map)

    monkeypatch.setattr(vb, "stage_from_ipc_agent", fake_stage)
    monkeypatch.setattr(vb, "atomic_swap_params", fake_swap)
    return record


def _precommit_kwargs(client: _FakeClient) -> dict:
    for method, kwargs in client.calls:
        if method == "precommit":
            return kwargs
    raise AssertionError("precommit was not called")


def test_accepted_commit_flips_weights_and_advances_version(monkeypatch) -> None:
    b = _make_binding(monkeypatch)
    b.obj.hotweights_agent = object()
    record = _install_staging(monkeypatch, {"weight": "NEW_TENSOR"})
    client = _FakeClient({"accepted": True, "waiting_for": []})

    ok = b._apply_update(client, "binder-1", _PLAN, "v2")

    assert ok is True
    assert record.get("swapped") == {"weight": "NEW_TENSOR"}
    assert b._last_version == "v2"
    assert b._pause_calls == {"paused": 1, "resumed": 1}
    # Token is forwarded on the mutating RPCs.
    assert _precommit_kwargs(client).get("token") == "secret"
    commit_kw = next(kw for m, kw in client.calls if m == "commit")
    assert commit_kw.get("token") == "secret"
    assert commit_kw.get("version") == "v2"
    assert _precommit_kwargs(client).get("version") == "v2"


def test_rejected_commit_leaves_module_untouched(monkeypatch) -> None:
    b = _make_binding(monkeypatch)
    b.obj.hotweights_agent = object()
    record = _install_staging(monkeypatch, {"weight": "NEW_TENSOR"})
    client = _FakeClient({"accepted": False, "waiting_for": ["worker-2"]})

    ok = b._apply_update(client, "binder-1", _PLAN, "v2")

    assert ok is False
    assert "swapped" not in record, "rejected commit must not flip weights"
    assert b._last_version is None, "version must not advance on rejection"
    assert b._pause_calls["resumed"] == 1, "requests must be resumed"


def test_unauthorized_commit_is_treated_as_rejected(monkeypatch) -> None:
    b = _make_binding(monkeypatch)
    b.obj.hotweights_agent = object()
    record = _install_staging(monkeypatch, {"weight": "NEW_TENSOR"})
    client = _FakeClient({"error": "unauthorized"})

    ok = b._apply_update(client, "binder-1", _PLAN, "v2")

    assert ok is False
    assert "swapped" not in record
    assert b._last_version is None


def test_verify_mismatch_aborts_before_precommit(monkeypatch) -> None:
    b = _make_binding(monkeypatch, verify=True)
    b.obj.hotweights_agent = object()
    _install_staging(monkeypatch, {"weight": "NEW_TENSOR"})

    def boom(items, read_bytes):  # noqa: ANN001
        raise SwapError("hash mismatch for w:0")

    monkeypatch.setattr(vb, "verify_staged_hashes", boom)
    client = _FakeClient({"accepted": True, "waiting_for": []})

    with pytest.raises(SwapError):
        b._apply_update(client, "binder-1", _PLAN, "v2")

    methods = [m for m, _ in client.calls]
    assert "precommit" not in methods, "must not precommit unverified weights"
    assert "commit" not in methods
    assert b._last_version is None
    assert b._pause_calls["resumed"] == 1


def test_loop_registers_binder_with_coordinator(monkeypatch) -> None:
    b = _make_binding(monkeypatch)
    b._last_version = "v9"  # already current -> no update this iteration
    client = _FakeClient({"accepted": True, "waiting_for": []})
    monkeypatch.setattr(vb, "Client", lambda endpoint: client)

    def one_sleep(_s):  # noqa: ANN001
        b._stop.set()

    monkeypatch.setattr(vb.time, "sleep", one_sleep)
    b._loop()

    register_calls = [kw for m, kw in client.calls if m == "register"]
    assert len(register_calls) == 1
    assert register_calls[0]["caps"]["role"] == "vllm-binder"


def test_verify_staged_hashes_ok() -> None:
    items = [{"key": "w:0", "hash": f"sha256:{_sha256_hex(_PAYLOAD)}"}]
    verify_staged_hashes(items, lambda key: _PAYLOAD)  # should not raise


def test_verify_staged_hashes_mismatch() -> None:
    items = [{"key": "w:0", "hash": "sha256:" + "0" * 64}]
    with pytest.raises(SwapError, match="hash mismatch"):
        verify_staged_hashes(items, lambda key: _PAYLOAD)


def test_binder_name_map_callable_is_honored(monkeypatch) -> None:
    b = _make_binding(monkeypatch, name_map=lambda plan: {"w:0": "weight"})
    assert b._resolve_name_map(_PLAN) == {"w:0": "weight"}


def test_binder_name_map_identity_fallback(monkeypatch) -> None:
    b = _make_binding(monkeypatch, name_map=None)
    assert b._resolve_name_map(_PLAN) == {"w:0": "w"}

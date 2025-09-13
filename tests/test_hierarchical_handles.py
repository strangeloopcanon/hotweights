from __future__ import annotations

import asyncio
import base64

import pytest

from hotweights.transport.cuda_ipc import CudaIPCTransport


class FakeHandleStore:
    def __init__(self) -> None:
        # version -> bucket_id -> node -> entry
        self.handles: dict[str, dict[int, dict[str, dict]]] = {}

    def post(self, version: str, bucket_id: int, node: str, handle: str, sig: str | None) -> None:
        self.handles.setdefault(version, {}).setdefault(bucket_id, {})[node] = {
            "handle": handle,
            "sig": sig,
            "acks": [],
        }

    def get(self, version: str, bucket_id: int, node: str | None) -> dict | None:
        v = self.handles.get(version, {})
        d = v.get(bucket_id, {})
        return d.get(node or "global") or d.get("global")

    def ack(self, version: str, bucket_id: int, node: str | None, who: str) -> int:
        e = self.get(version, bucket_id, node)
        if e is None:
            return 0
        acks = e.setdefault("acks", [])
        if who not in acks:
            acks.append(who)
        return len(acks)


class FakeClient:
    def __init__(self, store: FakeHandleStore):
        self.store = store

    def call(self, method: str, **kwargs):  # noqa: ANN003
        if method == "post_handle":
            self.store.post(
                str(kwargs.get("version") or "_"),
                int(kwargs.get("bucket_id", -1)),
                str(kwargs.get("node") or "global"),
                kwargs.get("handle"),
                kwargs.get("sig"),
            )
            return {"ok": True}
        if method == "get_handle":
            e = self.store.get(
                str(kwargs.get("version") or "_"),
                int(kwargs.get("bucket_id", -1)),
                str(kwargs.get("node") or "global"),
            )
            if e is None:
                return {"handle": None}
            return {"handle": e.get("handle"), "sig": e.get("sig")}
        if method == "ack_handle":
            n = self.store.ack(
                str(kwargs.get("version") or "_"),
                int(kwargs.get("bucket_id", -1)),
                str(kwargs.get("node") or "global"),
                str(kwargs.get("who") or "?")
            )
            return {"ok": True, "acks": n}
        raise RuntimeError(f"unknown method {method}")


def test_node_scoped_handle_post_get_ack(monkeypatch):
    store = FakeHandleStore()
    fc = FakeClient(store)

    class Dummy:
        pass

    d = Dummy()
    # minimal attributes used by the helpers
    d._client = fc
    d._handle_token = None
    d._handle_scope = "node"
    d._node_id = "nodeA"
    d._serialize_handle = lambda b: "b64:" + base64.b64encode(b).decode("ascii")  # type: ignore[attr-defined]
    d._sign_payload = lambda s: None  # type: ignore[attr-defined]
    d._deserialize_handle = lambda s: (base64.b64decode(s[4:].encode("ascii")) if isinstance(s, str) and s.startswith("b64:") else s)  # type: ignore[attr-defined]

    post = CudaIPCTransport._broadcast_handle.__get__(d, CudaIPCTransport)
    get = CudaIPCTransport._receive_handle.__get__(d, CudaIPCTransport)

    # Simulate posting a handle (use b64: payload for parity with transport)
    handle_bytes = b"HANDLE"
    ser = "b64:" + base64.b64encode(handle_bytes).decode("ascii")

    async def _post():
        await post(1, handle_bytes, version="v1")

    asyncio.run(_post())

    # Ensure store has node-scoped entry
    e = store.get("v1", 1, "nodeA")
    assert e is not None
    # Retrieve with node scope
    out = asyncio.run(get(1, version="v1"))
    assert out == handle_bytes or out == handle_bytes  # exact bytes returned

    # Ack increments count
    resp = fc.call("ack_handle", version="v1", bucket_id=1, node="nodeA", who="0")
    assert resp.get("acks") == 1

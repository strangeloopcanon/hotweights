from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from hotweights.adapters.vllm_plugin import update_weights_from_coordinator


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return f"sha256:{h.hexdigest()}"


def test_vllm_plugin_apply_updates_model(tmp_path: Path) -> None:
    # Skip if torch not available
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return

    # Create a simple target model
    m = nn.Linear(2, 2, bias=False)
    # Build a binary shard payload for the weight (float32, same shape)
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = arr.tobytes()
    shard_path = tmp_path / "w.bin"
    shard_path.write_bytes(payload)

    # Craft a minimal plan with one bucket and a single item
    item = {
        "tensor": "w.bin",
        "key": "w:0",
        "nbytes": len(payload),
        "hash": _sha256_bytes(payload),
        "uri": shard_path.as_uri(),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "offset": 0,
    }
    plan = {
        "version": "v1",
        "bucket_bytes": len(payload),
        "total_bytes": len(payload),
        "buckets": [{"bucket_id": 0, "size": len(payload), "items": [item]}],
    }

    # Fake coordinator client that returns our plan
    class _FakeClient:
        def __init__(self, endpoint: str):  # noqa: ANN001
            self.endpoint = endpoint
        def call(self, method: str, **kwargs):  # noqa: ANN003, ANN201
            if method == "get_plan":
                return {"plan": plan, "digest": "demo"}
            if method in ("precommit", "commit"):
                return {"ok": True}
            if method == "status":
                return {"version": plan["version"]}
            raise RuntimeError(f"unsupported method {method}")

    # Monkeypatch the coordinator client used by the plugin
    import hotweights.adapters.vllm_plugin as plugin
    plugin.Client = lambda endpoint: _FakeClient(endpoint)  # type: ignore

    # Provide a mapping function for the weight param name
    def name_map_fn(p):  # noqa: ANN001, ANN202
        return {"w:0": "weight"}

    # Execute plugin update (local path, no MPI)
    update_weights_from_coordinator(
        m,
        name_map_fn,
        endpoint="tcp://127.0.0.1:5555",
        use_mpi=False,
        pinned=False,
        verify=False,
        device="cpu",
    )

    got = m.weight.detach().cpu().numpy()
    assert np.allclose(got, arr)

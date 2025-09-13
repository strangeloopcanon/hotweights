from __future__ import annotations

"""
vLLM Integration Smoke Demo (CPU-friendly)

This script demonstrates how to use the Hotweights WorkerExtension path to
fetch a plan from the coordinator and apply it to a simple torch.nn.Module.

It avoids starting a real vLLM engine and does not require CUDA. If pyzmq is
installed, it can talk to the coordinator; otherwise, it uses a local fake
client with an in-process plan.

Usage (coordinator mode):
  1) Start coordinator in another shell:
     hotweights coord-serve --endpoint tcp://127.0.0.1:5555
  2) In this shell, run the demo (it will publish a toy plan and apply it):
     python examples/vllm_smoke_demo.py --use-coord --endpoint tcp://127.0.0.1:5555

Usage (local mode without coordinator):
     python examples/vllm_smoke_demo.py
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return f"sha256:{h.hexdigest()}"


def build_toy_plan(tmpdir: Path) -> dict:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    payload = arr.tobytes()
    shard_path = tmpdir / "w.bin"
    shard_path.write_bytes(payload)
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
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-coord", action="store_true", help="Use ZeroMQ coordinator")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    args = ap.parse_args()

    try:
        import torch.nn as nn
    except Exception as e:
        print(f"Torch is required for this demo: {e}")
        return

    from hotweights.adapters.vllm_extension import HotweightsWorkerExtension

    tmpdir = Path("./demo_vllm_smoke").resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    plan = build_toy_plan(tmpdir)

    # Option 1: coordinator mode — submit plan + begin; extension fetches it
    if args.use_coord:
        try:
            from hotweights.coordinator.zmq_client import Client
        except Exception as e:
            print(f"pyzmq required for coordinator mode: {e}")
            return
        c = Client(args.endpoint)
        digest = _sha256_bytes(bytes(str(plan), "utf-8"))
        print(c.call("submit_plan", plan=plan, digest=digest))
        print(c.call("begin", version=plan["version"], digest=digest))

    # Build a tiny model and bind extension
    m = nn.Linear(2, 2, bias=False)
    ext = HotweightsWorkerExtension(endpoint=args.endpoint)
    ext.bind_module(m)

    if args.use_coord:
        print("Applying update from coordinator...")
        ext.apply_update()
    else:
        # Local fallback: monkeypatch client to return the in-process plan
        import hotweights.adapters.vllm_plugin as plugin

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

        plugin.Client = lambda endpoint: _FakeClient(endpoint)  # type: ignore
        print("Applying update via local fake coordinator...")
        ext.apply_update()

    # Confirm applied
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    got = m.weight.detach().cpu().numpy()
    ok = np.allclose(got, expected)
    print("Applied weights match expected:", ok)


if __name__ == "__main__":
    main()

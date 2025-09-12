from __future__ import annotations

import os
import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

from hotweights.adapters.vllm_plugin import update_weights_from_coordinator
from hotweights.manifest import build_simple_manifest, dump_manifest
from hotweights.cli import _create_plan


def main():
    if torch is None:
        print("Torch not available; demo skipped")
        return
    # Create a toy checkpoint with a single .npy weight
    os.makedirs("demo_ckpt_v0", exist_ok=True)
    os.makedirs("demo_ckpt_v1", exist_ok=True)
    w0 = np.zeros((2, 2), dtype=np.float32)
    w1 = np.ones((2, 2), dtype=np.float32)
    np.save("demo_ckpt_v0/w.npy", w0)
    np.save("demo_ckpt_v1/w.npy", w1)

    m_prev = build_simple_manifest("toy", "v0", "demo_ckpt_v0")
    m_next = build_simple_manifest("toy", "v1", "demo_ckpt_v1")
    plan = _create_plan(m_prev, m_next, bucket_mb=1)

    # Start a local apply using the plugin API without a coordinator by faking get_plan
    class FakeClient:
        def call(self, method, **kw):  # noqa: ANN001
            if method == "get_plan":
                return {"plan": plan, "digest": "demo"}
            raise RuntimeError("not implemented")

    import hotweights.adapters.vllm_plugin as plugin
    plugin.Client = lambda endpoint: FakeClient()  # type: ignore

    model = nn.Linear(2, 2, bias=False)

    def name_map_fn(p):  # noqa: ANN001
        mapping = {}
        for b in p["buckets"]:
            for it in b["items"]:
                mapping[it["key"]] = "weight"
        return mapping

    update_weights_from_coordinator(model, name_map_fn, use_mpi=False, pinned=True, verify=True, device="cpu")
    print("Applied weights:", model.weight.detach().numpy())


if __name__ == "__main__":
    main()


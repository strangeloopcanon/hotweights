from __future__ import annotations

import numpy as np

from hotweights.adapters.vllm_ext import HotReloadExtension, torch
from hotweights.staging.host_agent import HostAgent


def test_adapter_bind_and_commit_cpu():
    host = HostAgent()
    # Create a 2x2 float32 weight as .npy-like bytes
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    data = arr.tobytes()
    items = [{"key": "w:0", "nbytes": len(data), "dtype": str(arr.dtype), "shape": list(arr.shape)}]
    host.write("w:0", 0, memoryview(data))
    host.seal("w:0")

    ext = HotReloadExtension()
    ext.begin_update("v1", {})
    ext.ingest_from_host(items, host)

    if torch is not None:
        # Bind a tiny module
        import torch.nn as nn
        m = nn.Linear(2, 2, bias=False)
        # Map staged key to module weight param
        ext.bind_module(m, {"w:0": "weight"})
        ext.precommit(); ext.commit("v1")
        # After commit, weights should match
        got = m.weight.detach().cpu().numpy()
        assert got.shape == arr.shape
        # Compare ignoring layout (linear weight is out_features x in_features)
        # our dummy mapping assumes matching shape; so direct compare.
        assert np.allclose(got, arr)


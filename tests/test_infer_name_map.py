from __future__ import annotations

import types
import numpy as np

from hotweights.adapters.vllm_plugin import infer_name_map
from hotweights.adapters.vllm_ext import torch


def test_infer_name_map_simple():
    if torch is None:
        return
    import torch.nn as nn

    m = nn.Linear(2, 2, bias=False)
    # Fake plan for a .npy weight named w.npy
    plan = {
        "version": "v1",
        "buckets": [
            {
                "bucket_id": 0,
                "size": 16,
                "items": [
                    {"tensor": "w.npy", "key": "w:0", "shape": list(m.weight.shape), "dtype": str(m.weight.dtype).replace("torch.", "")},
                ],
            }
        ],
    }
    mapping = infer_name_map(m, plan)
    assert mapping["w:0"].endswith("weight")


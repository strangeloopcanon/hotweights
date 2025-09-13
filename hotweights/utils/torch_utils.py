from __future__ import annotations

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

import numpy as np


def torch_dtype_from_numpy_str(dtype_str: str) -> object | None:
    if torch is None:
        return None
    try:
        dt = np.dtype(dtype_str)
    except Exception:
        return None
    # Build mapping defensively to avoid importing dtypes that may not exist
    mapping = {
        np.dtype("float32"): torch.float32,
        np.dtype("float64"): torch.float64,
        np.dtype("float16"): torch.float16,
        np.dtype("int8"): torch.int8,
        np.dtype("uint8"): torch.uint8,
        np.dtype("int16"): torch.int16,
        np.dtype("int32"): torch.int32,
        np.dtype("int64"): torch.int64,
        np.dtype("bool"): torch.bool,
    }
    # bfloat16 may not be available in all NumPy builds
    try:
        bf16 = np.dtype("bfloat16")
        mapping[bf16] = getattr(torch, "bfloat16", None)
    except Exception:  # pragma: no cover - depends on NumPy version
        pass
    return mapping.get(dt, None)

"""Planner functions for delta compute and bucket packing.

Implements pandas versions by default; when Bodo is available, JIT wraps
the same logic. The public API remains pandas DataFrame in/out.
"""
from __future__ import annotations

from typing import Optional
import os

import numpy as np
import pandas as pd

try:  # optional
    import bodo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bodo = None  # type: ignore


def _compute_delta_pd(prev_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    j = next_df.merge(
        prev_df,
        on=["tensor", "shard_rank"],
        how="left",
        suffixes=("", "_prev"),
    )
    mask = j["hash_prev"].isna() | (j["hash_prev"] != j["hash"])
    cols = ["tensor", "shard_rank", "nbytes", "hash", "path"]
    return j.loc[mask, cols].reset_index(drop=True)


def _pack_buckets_pd(delta_df: pd.DataFrame, bucket_bytes: int) -> pd.DataFrame:
    """First-fit decreasing bin pack.

    Sort by size desc; place each item into the first bucket that fits,
    otherwise open a new bucket. Returns a copy with a new ``bucket_id`` col.
    """
    df = delta_df.sort_values("nbytes", ascending=False).reset_index(drop=True)
    n = len(df)
    bucket_id = np.empty(n, dtype=np.int64)
    # Track remaining capacity per bucket
    remaining: list[int] = []
    for i in range(n):
        size = int(df["nbytes"].iat[i])
        placed = False
        for b_idx in range(len(remaining)):
            if remaining[b_idx] >= size:
                bucket_id[i] = b_idx
                remaining[b_idx] -= size
                placed = True
                break
        if not placed:
            # open a new bucket
            bucket_id[i] = len(remaining)
            remaining.append(bucket_bytes - size)
    df = df.copy()
    df["bucket_id"] = bucket_id
    return df


if bodo is not None and os.getenv("HOTWEIGHTS_FORCE_PANDAS", "0") not in ("1", "true", "True"):  # pragma: no cover - exercised when bodo present
    _compute_delta_jit = bodo.jit(cache=True)(_compute_delta_pd)
    _pack_buckets_jit = bodo.jit(cache=True)(_pack_buckets_pd)
    _bodo_warned = False

    def compute_delta(prev_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
        """Run Bodo-compiled delta when possible, else fallback to pandas.

        Some environments may import Bodo successfully but not support
        distributing specific pandas ops used here. To keep the public API
        reliable, catch runtime JIT errors and use the pure-pandas path.
        """
        global _bodo_warned  # type: ignore[global-variable-not-assigned]
        try:
            return _compute_delta_jit(prev_df, next_df)
        except Exception:
            if not _bodo_warned:
                print("[hotweights] Bodo JIT fallback to pandas for compute_delta")
                _bodo_warned = True
            return _compute_delta_pd(prev_df, next_df)

    def pack_buckets(delta_df: pd.DataFrame, bucket_bytes: int) -> pd.DataFrame:
        global _bodo_warned  # type: ignore[global-variable-not-assigned]
        try:
            return _pack_buckets_jit(delta_df, bucket_bytes)
        except Exception:
            if not _bodo_warned:
                print("[hotweights] Bodo JIT fallback to pandas for pack_buckets")
                _bodo_warned = True
            return _pack_buckets_pd(delta_df, bucket_bytes)
else:
    if bodo is not None:
        print("[hotweights] Using pandas implementations (HOTWEIGHTS_FORCE_PANDAS)")
    compute_delta = _compute_delta_pd
    pack_buckets = _pack_buckets_pd


def warmup() -> None:
    """Optional JIT warmup for planner functions.

    Compiles the Bodo JIT versions (when available) using tiny DataFrames to
    amortize first-call latency in long-lived processes.
    """
    try:
        import pandas as _pd
        prev = _pd.DataFrame([
            {"tensor": "t", "shard_rank": 0, "nbytes": 1, "hash": "h0", "path": "t"}
        ])
        nxt = _pd.DataFrame([
            {"tensor": "t", "shard_rank": 0, "nbytes": 1, "hash": "h1", "path": "t"}
        ])
        _ = compute_delta(prev, nxt)
        _ = pack_buckets(nxt, bucket_bytes=8)
    except Exception:
        pass

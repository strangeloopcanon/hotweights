from __future__ import annotations

import pandas as pd

from hotweights.planner_bodo import compute_delta, pack_buckets


def _mk_df(rows):
    return pd.DataFrame(rows, columns=["tensor", "shard_rank", "nbytes", "hash", "path"])  # fmt: off


def test_compute_delta_simple():
    prev = _mk_df([
        ("t0", 0, 10, "h0", "t0"),
        ("t1", 0, 20, "h1", "t1"),
    ])
    nxt = _mk_df([
        ("t0", 0, 10, "h0", "t0"),  # unchanged
        ("t1", 0, 20, "H1", "t1"),   # changed
        ("t2", 0, 5,  "h2", "t2"),    # added
    ])
    delta = compute_delta(prev, nxt)
    got = {(r.tensor, r.shard_rank, r.hash) for r in delta.itertuples()}
    assert got == {("t1", 0, "H1"), ("t2", 0, "h2")}


def test_pack_buckets_greedy():
    df = _mk_df([
        ("a", 0, 6, "ha", "a"),
        ("b", 0, 5, "hb", "b"),
        ("c", 0, 4, "hc", "c"),
        ("d", 0, 3, "hd", "d"),
    ])
    out = pack_buckets(df, bucket_bytes=8)
    # Greedy desc: 6|2, 5|3 -> expect [6]->b0, [5,3]->b1, [4]->b2
    sizes_by_bucket = {}
    for row in out.itertuples():
        sizes_by_bucket.setdefault(row.bucket_id, 0)
        sizes_by_bucket[row.bucket_id] += int(row.nbytes)
    assert all(v <= 8 for v in sizes_by_bucket.values())
    assert set(sizes_by_bucket.values()) == {6, 8, 4}


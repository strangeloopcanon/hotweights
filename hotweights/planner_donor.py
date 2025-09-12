"""Donor assignment for late-join P2P (simplified).

Given:
  - have_df: columns [bucket_id, rank]
  - need_df: columns [bucket_id, rank]

Return a DataFrame with columns [bucket_id, rank, donor_rank], assigning up to k
donors per (bucket_id, rank) by round-robin among available donors for that bucket.
"""
from __future__ import annotations

import pandas as pd


def assign_donors(have_df: pd.DataFrame, need_df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    out_rows: list[tuple[int, int, int]] = []
    # Group by bucket, choose donors from 'have', assign to each 'need' rank
    have_groups = {b: g["rank"].tolist() for b, g in have_df.groupby("bucket_id")}
    for bucket_id, g_need in need_df.groupby("bucket_id"):
        donors = have_groups.get(bucket_id, [])
        if not donors:
            continue
        for r in g_need["rank"].tolist():
            # pick k donors not equal to r, round-robin
            choices = [d for d in donors if d != r]
            if not choices:
                continue
            for i in range(min(k, len(choices))):
                out_rows.append((int(bucket_id), int(r), int(choices[i % len(choices)])))
    return pd.DataFrame(out_rows, columns=["bucket_id", "rank", "donor_rank"])


from __future__ import annotations

import pandas as pd

from hotweights.planner_donor import assign_donors


def test_assign_donors_round_robin():
    have = pd.DataFrame({"bucket_id": [0, 0, 0], "rank": [0, 1, 2]})
    need = pd.DataFrame({"bucket_id": [0, 0], "rank": [3, 1]})
    out = assign_donors(have, need, k=2)
    # For rank 3, donors can be 0,1,2 (two of them)
    # For rank 1, donors can be 0,2
    by_rank = {r: sorted(g["donor_rank"].tolist()) for r, g in out.groupby("rank")}
    assert set(by_rank.keys()) == {1, 3}
    assert set(by_rank[1]) == {0, 2}
    assert len(by_rank[3]) == 2


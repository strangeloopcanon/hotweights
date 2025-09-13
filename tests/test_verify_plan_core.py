from __future__ import annotations

from hotweights.core.replicate import verify_plan


def test_verify_plan_basic_missing_and_range():
    plan = {
        "buckets": [
            {"bucket_id": 0, "items": [], "size": 0},  # no consumer_ranks
            {"bucket_id": 1, "items": [], "size": 0, "consumer_ranks": [0, 1, 1]},  # dupes
            {"bucket_id": 2, "items": [], "size": 0, "consumer_ranks": [3, 4]},  # out of range for ws=4
        ]
    }
    report = verify_plan(plan, require_consumers=True, world_size=4)
    assert report["buckets"] == 3
    assert report["missing_consumer_ranks"] == 1
    assert report["buckets_with_duplicates"] == 1
    assert report["buckets_with_out_of_range"] == 1
    assert len(report["problems"]) >= 2


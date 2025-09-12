"""Replication helpers: plan creation and bucket staging.

Centralizes logic shared by CLI, worker, and adapters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import fnmatch
import json
import numpy as np

from ..planner_bodo import compute_delta, pack_buckets


def create_plan(prev: dict, nxt: dict, bucket_mb: int, consumer_map: Optional[Dict[str, List[int]]] = None) -> dict:
    import pandas as pd

    def to_df(m):
        rows = []
        for t in m["tensors"]:
            for s in t["shards"]:
                rows.append(
                    {
                        "tensor": t["name"],
                        "shard_rank": s["rank"],
                        "nbytes": int(s["bytes"]),
                        "hash": s["hash"],
                        "path": t["name"],
                        "uri": s["uri"],
                        "dtype": t.get("dtype"),
                        "shape": t.get("shape"),
                    }
                )
        return pd.DataFrame(rows)

    prev_df = to_df(prev)
    next_df = to_df(nxt)
    uri_map = {(r.tensor, int(r.shard_rank)): r.uri for r in next_df.itertuples()}
    dtype_map = {(r.tensor, int(r.shard_rank)): getattr(r, "dtype", None) for r in next_df.itertuples()}
    shape_map = {(r.tensor, int(r.shard_rank)): getattr(r, "shape", None) for r in next_df.itertuples()}
    delta = compute_delta(prev_df, next_df)
    bucket_bytes = int(bucket_mb * (1 << 20))
    packed = pack_buckets(delta, bucket_bytes=bucket_bytes).reset_index(drop=True)

    buckets: Dict[int, dict] = {}
    for row in packed.itertuples():
        b = int(row.bucket_id)
        buckets.setdefault(b, {"bucket_id": b, "items": [], "size": 0})
        off = buckets[b]["size"]
        key = (row.tensor, int(row.shard_rank))
        item = (
            {
                "tensor": row.tensor,
                "shard_rank": int(row.shard_rank),
                "nbytes": int(row.nbytes),
                "hash": row.hash,
                "uri": str(uri_map.get(key, "")),
                "dtype": dtype_map.get(key),
                "shape": shape_map.get(key),
                "key": f"{row.tensor}:{int(row.shard_rank)}",
                "offset": int(off),
            }
        )
        buckets[b]["items"].append(item)
        buckets[b]["size"] = off + int(row.nbytes)

    # Assign consumer ranks per bucket based on tensor name patterns, if provided
    if consumer_map:
        for b in buckets.values():
            ranks: set[int] = set()
            for it in b["items"]:
                name = it["tensor"]
                for pat, rlist in consumer_map.items():
                    try:
                        if fnmatch.fnmatch(name, pat):
                            ranks.update(int(x) for x in rlist)
                    except Exception:
                        continue
            if ranks:
                b["consumer_ranks"] = sorted(ranks)
    else:
        # Optional auto-derivation from manifest partitioning or env HOTWEIGHTS_TP/HOTWEIGHTS_TP_GROUPS
        try:
            import os

            auto_tp = os.getenv("HOTWEIGHTS_AUTO_TP", "0") in ("1", "true", "True")
            tp_env = int(os.getenv("HOTWEIGHTS_TP", "0"))
            tp_groups_s = os.getenv("HOTWEIGHTS_TP_GROUPS", "")
        except Exception:
            auto_tp, tp_env, tp_groups_s = False, 0, ""

        # Parse TP groups mapping if provided (JSON string or file path)
        tp_groups = None
        if tp_groups_s:
            import json as _json
            from pathlib import Path as _Path
            try:
                if tp_groups_s.strip().startswith("{"):
                    tp_groups = _json.loads(tp_groups_s)
                else:
                    tp_groups = _json.loads(_Path(tp_groups_s).read_text())
            except Exception:
                tp_groups = None

        tp = 0
        if tp_env > 0:
            tp = tp_env
        else:
            # Find max 'tp' in next manifest tensor partitioning, if present
            try:
                tps: list[int] = []
                for t in nxt.get("tensors", []):
                    part = t.get("partitioning") or {}
                    val = int(part.get("tp", 0)) if isinstance(part, dict) and ("tp" in part) else 0
                    if val:
                        tps.append(val)
                if tps:
                    tp = max(tps)
            except Exception:
                tp = 0

        # Build tensor->partitioning lookup
        part_map: dict[str, dict] = {}
        try:
            for t in nxt.get("tensors", []):
                part_map[t.get("name", "")] = t.get("partitioning") or {}
        except Exception:
            part_map = {}

        if tp_groups and isinstance(tp_groups, dict):
            # Map group ids to rank lists using partitioning info per item
            for b in buckets.values():
                ranks: set[int] = set()
                for it in b["items"]:
                    part = part_map.get(it["tensor"], {})
                    gid = part.get("tp_group") or part.get("group") or part.get("group_id")
                    if gid is None:
                        continue
                    rlist = tp_groups.get(str(gid))
                    if rlist:
                        try:
                            ranks.update(int(x) for x in rlist)
                        except Exception:
                            continue
                if ranks:
                    b["consumer_ranks"] = sorted(ranks)
        elif auto_tp and tp > 1:
            # Simple contiguous mapping: first 'tp' ranks consume all buckets
            ranks = list(range(tp))
            for b in buckets.values():
                b["consumer_ranks"] = ranks

    plan = {
        "version": nxt.get("version", "unknown"),
        "bucket_bytes": bucket_bytes,
        "total_bytes": int(packed["nbytes"].sum()) if len(packed) else 0,
        "buckets": [buckets[k] for k in sorted(buckets.keys())],
    }
    return plan


def assemble_bucket(items: List[dict]) -> np.ndarray:
    size = sum(int(x["nbytes"]) for x in items)
    buf = np.empty(size, dtype=np.uint8)
    for it in items:
        uri = it["uri"]
        assert uri.startswith("file://"), f"Unsupported URI: {uri}"
        path = Path(uri[len("file://") :])
        n = int(it["nbytes"])  # expected size
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert mm.shape[0] == n, f"size mismatch for {path}: got {mm.shape[0]}, expect {n}"
        off = int(it["offset"])  # computed in plan
        buf[off : off + n] = mm[:n]
    return buf


def assemble_bucket_to_buffer(items: List[dict], out: np.ndarray) -> None:
    """Assemble items into an existing contiguous uint8 numpy buffer.

    This avoids extra allocations and lets callers control buffer placement.
    """
    assert out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"], "out must be contiguous uint8"
    size = sum(int(x["nbytes"]) for x in items)
    assert out.size >= size, f"output buffer too small: {out.size} < {size}"
    for it in items:
        uri = it["uri"]
        assert uri.startswith("file://"), f"Unsupported URI: {uri}"
        path = Path(uri[len("file://") :])
        n = int(it["nbytes"])  # expected size
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert mm.shape[0] == n, f"size mismatch for {path}: got {mm.shape[0]}, expect {n}"
        off = int(it["offset"])  # computed in plan
        out[off : off + n] = mm[:n]


def scatter_bucket(host, items: List[dict], buf: np.ndarray) -> None:  # noqa: ANN001
    for it in items:
        key = it["key"]
        off = int(it["offset"]) 
        n = int(it["nbytes"]) 
        mv = memoryview(buf)[off : off + n]
        host.write(key, 0, mv)
        host.seal(key)


def verify_items(host, items: List[dict]) -> None:  # noqa: ANN001
    import hashlib

    for it in items:
        algo, expect = it["hash"].split(":", 1)
        h = hashlib.new(algo)
        h.update(bytes(host.read(it["key"]))[: int(it["nbytes"])])
        got = h.hexdigest()
        assert got == expect, f"hash mismatch for {it['key']}: got {got}, expect {expect}"


def plan_digest(plan: dict) -> str:
    import hashlib

    blob = json.dumps(plan, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"

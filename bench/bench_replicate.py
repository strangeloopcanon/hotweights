from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hotweights.cli import _assemble_bucket, _scatter_bucket, _verify_items
from hotweights.staging.host_agent import HostAgent


def run_once(plan_path: Path, verify: bool = True, pinned: bool = False) -> dict:
    plan = json.loads(plan_path.read_text())
    t0 = time.perf_counter()
    host = HostAgent(use_pinned=pinned)
    bytes_total = int(plan.get("total_bytes", 0))
    buckets = 0
    for b in plan.get("buckets", []):
        items = b["items"]
        buf = _assemble_bucket(items)
        _scatter_bucket(host, items, buf)
        if verify:
            _verify_items(host, items)
        buckets += 1
    dt = time.perf_counter() - t0
    return {
        "seconds": dt,
        "bytes": bytes_total,
        "buckets": buckets,
        "throughput_bps": (bytes_total / dt) if dt > 0 else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--pinned", action="store_true")
    args = ap.parse_args()
    results = []
    for _ in range(args.repeat):
        results.append(
            run_once(args.plan, verify=not args.no_verify, pinned=args.pinned)
        )
    out = {
        "runs": results,
        "avg_seconds": sum(r["seconds"] for r in results) / len(results),
        "avg_throughput_bps": sum(r["throughput_bps"] for r in results)
        / len(results),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

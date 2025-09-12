from __future__ import annotations

"""End-to-end benchmark harness for replication.

Measures assemble/broadcast/scatter end-to-end time for a given plan using either
CUDA-IPC (when CUDA is available) or the MPI/UCX fallback path.
"""

import argparse
import json
import os
import time
from pathlib import Path


def bench_cuda_ipc(plan_path: Path, device: str, coord_endpoint: str | None) -> dict:
    from hotweights.staging.cuda_ipc_agent import CudaIPCAgent
    from hotweights.transport.cuda_ipc import CudaIPCTransport
    from hotweights.telemetry.cuda_ipc_metrics import CudaIPCMetrics

    plan = json.loads(plan_path.read_text())
    rank = int(os.getenv("RANK", "0"))
    metrics = CudaIPCMetrics(rank)
    agent = CudaIPCAgent(device=device)
    transport = CudaIPCTransport(agent=agent, metrics=metrics, coord_endpoint=coord_endpoint)
    t0 = time.perf_counter()
    transport.replicate(plan)
    dt = time.perf_counter() - t0
    return {"mode": "cuda_ipc", "seconds": dt, "buckets": len(plan.get("buckets", [])), "bytes": int(plan.get("total_bytes", 0))}


def bench_fallback(plan_path: Path, use_mpi: bool, window: int, group: str | None, mpi_chunk_mb: int, verify: bool) -> dict:
    import numpy as np
    from hotweights.staging.host_agent import HostAgent
    from hotweights.cli import _assemble_bucket as assemble, _scatter_bucket as scatter, _verify_items as verify_items
    from hotweights.transport.mpi_stream import MPIReplicator
    from hotweights.transport.ucx_stream import UCXReplicator

    plan = json.loads(plan_path.read_text())
    host = HostAgent(use_pinned=False)
    t0 = time.perf_counter()
    all_items = []
    if use_mpi:
        ranks = None
        if group:
            try:
                ranks = [int(x) for x in group.split(",") if x.strip()]
            except Exception:
                ranks = None
        replicator = MPIReplicator(window=window, group_ranks=ranks, chunk_bytes=(mpi_chunk_mb << 20) if mpi_chunk_mb > 0 else 0)
        rank = getattr(replicator, "world_rank", getattr(replicator, "rank", 0))
        bucket_bufs: list[tuple[dict, np.ndarray]] = []

        def gen():
            for b in plan.get("buckets", []):
                items = b["items"]
                size = int(b["size"])  # precomputed
                consumers = b.get("consumer_ranks")
                if consumers is not None and rank not in consumers:
                    continue
                group_root = 0 if not consumers else min(int(x) for x in consumers)
                if rank == group_root:
                    buf = assemble(items)
                else:
                    buf = np.empty(size, dtype=np.uint8)
                bucket_bufs.append((b, buf))
                if consumers is None:
                    yield (int(b["bucket_id"]), buf)
                else:
                    yield (int(b["bucket_id"]), buf, list(int(x) for x in consumers))

        def on_complete(_bid: int, _buf: np.ndarray) -> None:
            b, buf = bucket_bufs.pop(0)
            items = b["items"]
            scatter(host, items, buf)
            if verify and rank == (min(int(x) for x in b.get("consumer_ranks", [0])) if b.get("consumer_ranks") else 0):
                verify_items(host, items)
            all_items.extend(items)

        replicator.replicate_stream(gen(), on_complete)
    else:
        # UCX or local fallback (UCX path omitted in simple bench)
        for b in plan.get("buckets", []):
            items = b["items"]
            buf = assemble(items)
            scatter(host, items, buf)
            if verify:
                verify_items(host, items)
            all_items.extend(items)
    dt = time.perf_counter() - t0
    return {"mode": "fallback_mpi" if use_mpi else "fallback_local", "seconds": dt, "buckets": len(plan.get("buckets", [])), "bytes": int(plan.get("total_bytes", 0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--coord-endpoint", default=None)
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--mpi", action="store_true")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--group", default=None)
    ap.add_argument("--mpi-chunk-mb", type=int, default=0)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    try:
        import torch  # type: ignore
        cuda_ok = torch.cuda.is_available() and args.device.startswith("cuda")
    except Exception:
        cuda_ok = False

    if not args.fallback and cuda_ok:
        out = bench_cuda_ipc(args.plan, args.device, args.coord_endpoint)
    else:
        out = bench_fallback(args.plan, args.mpi, args.window, args.group, args.mpi_chunk_mb, args.verify)
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

"""hotweights CLI."""
from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from .manifest import build_simple_manifest, dump_manifest, load_manifest
from .planner_bodo import compute_delta, pack_buckets
# Legacy imports replaced by SOTA modules
from .telemetry.metrics import Timer
from .telemetry.prom import Histogram, start_http_server
from .telemetry.nvml import min_free_vram_mib
import json
import numpy as np


def _derive_pub_endpoint(endpoint: str) -> str:
    """Derive a PUB endpoint from a REP endpoint by incrementing port."""
    try:
        if endpoint.startswith("tcp://") and ":" in endpoint.rsplit(":", 1)[-1]:
            host, port_s = endpoint[len("tcp://") :].rsplit(":", 1)
            port = int(port_s)
            return f"tcp://{host}:{port+1}"
    except Exception:
        pass
    return "tcp://127.0.0.1:5556"



def _cmd_publish(args: argparse.Namespace) -> int:
    manifest = build_simple_manifest(
        model_id=args.model_id, version=args.version, checkpoint_dir=args.checkpoint
    )
    out = Path(args.output or (Path(args.checkpoint) / "checkpoint.manifest.json"))
    dump_manifest(manifest, out)
    print(f"Wrote manifest: {out}")
    return 0


def _maybe_coord_client(endpoint: str | None):
    if endpoint is None:
        return None
    try:
        from .coordinator.zmq_client import Client

        return Client(endpoint)
    except Exception as e:  # pragma: no cover - optional
        print(f"Coordinator unavailable: {e}")
        return None


def _cmd_commit(args: argparse.Namespace) -> int:
    endpoint = args.endpoint or os.getenv("HOTWEIGHTS_COORD")
    if endpoint:
        c = _maybe_coord_client(endpoint)
        if c is None:
            return 1
        print(c.call("commit", version=args.version))
    else:
        print(f"Commit requested for version {args.version} (local stub)")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    endpoint = args.endpoint or os.getenv("HOTWEIGHTS_COORD")
    if endpoint:
        c = _maybe_coord_client(endpoint)
        if c is None:
            return 1
        print(c.call("status"))
        return 0
    print("Status: stub (no coordinator running)")
    return 0


def _create_plan(prev: dict, nxt: dict, bucket_mb: int, consumer_map: dict | None = None) -> dict:
    import pandas as pd
    import fnmatch

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
    # map for URIs from next manifest
    uri_map = { (r.tensor, int(r.shard_rank)): r.uri for r in next_df.itertuples() }
    dtype_map = { (r.tensor, int(r.shard_rank)): getattr(r, "dtype", None) for r in next_df.itertuples() }
    shape_map = { (r.tensor, int(r.shard_rank)): getattr(r, "shape", None) for r in next_df.itertuples() }
    delta = compute_delta(prev_df, next_df)
    bucket_bytes = int(bucket_mb * (1 << 20))
    packed = pack_buckets(delta, bucket_bytes=bucket_bytes).reset_index(drop=True)

    # Assign offsets within each bucket and materialize items
    buckets = {}
    for row in packed.itertuples():
        b = int(row.bucket_id)
        buckets.setdefault(b, {"bucket_id": b, "items": [], "size": 0})
        off = buckets[b]["size"]
        key = (row.tensor, int(row.shard_rank))
        item = {
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
        buckets[b]["items"].append(item)
        buckets[b]["size"] = off + int(row.nbytes)

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

    plan = {
        "version": nxt.get("version", "unknown"),
        "bucket_bytes": bucket_bytes,
        "total_bytes": int(packed["nbytes"].sum()) if len(packed) else 0,
        "buckets": [buckets[k] for k in sorted(buckets.keys())],
    }
    return plan


def _cmd_plan(args: argparse.Namespace) -> int:
    prev = load_manifest(args.prev)
    nxt = load_manifest(args.next)
    bucket_mb = args.bucket_mb
    consumer_map = None
    gmap = getattr(args, "group_map", None) or os.getenv("HOTWEIGHTS_GROUP_MAP")
    if gmap:
        try:
            consumer_map = json.loads(Path(gmap).read_text())
        except Exception as e:
            print(f"Failed to load group map {gmap}: {e}")
    if args.auto:
        free = min_free_vram_mib()
        if free:
            bucket_mb = max(1, int(free * args.alpha))
            print(f"Auto bucket size MiB: {bucket_mb} (free={free}MiB, alpha={args.alpha})")
    plan = _create_plan(prev, nxt, bucket_mb=bucket_mb, consumer_map=consumer_map)
    out = Path(args.output or "transfer.plan.json")
    out.write_text(json.dumps(plan, indent=2))
    print(f"Wrote plan: {out} ({plan['total_bytes']} bytes across {len(plan['buckets'])} buckets)")
    return 0


def _assemble_bucket(items: list[dict]) -> np.ndarray:
    size = sum(int(x["nbytes"]) for x in items)
    buf = np.empty(size, dtype=np.uint8)
    for it in items:
        uri = it["uri"]
        assert uri.startswith("file://"), f"Unsupported URI: {uri}"
        path = Path(uri[len("file://") :])
        n = int(it["nbytes"])  # expected size
        # Use memory map to avoid extra copies for large files
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        assert mm.shape[0] == n, f"size mismatch for {path}: got {mm.shape[0]}, expect {n}"
        off = int(it["offset"])  # computed in plan
        buf[off : off + n] = mm[:n]
    return buf


def _scatter_bucket(host: HostAgent, items: list[dict], buf: np.ndarray) -> None:
    for it in items:
        key = it["key"]
        off = int(it["offset"])
        n = int(it["nbytes"]) 
        mv = memoryview(buf)[off : off + n]
        host.write(key, 0, mv)
        host.seal(key)


def _verify_items(host: HostAgent, items: list[dict]) -> None:
    import hashlib

    for it in items:
        algo, expect = it["hash"].split(":", 1)
        h = hashlib.new(algo)
        h.update(bytes(host.read(it["key"]))[: int(it["nbytes"])])
        got = h.hexdigest()
        assert (
            got == expect
        ), f"hash mismatch for {it['key']}: got {got}, expect {expect}"


def _plan_digest(plan: dict) -> str:
    import hashlib

    blob = json.dumps(plan, sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def _cmd_replicate(args: argparse.Namespace) -> int:
    plan = json.loads(Path(args.plan).read_text())
    total_bytes = int(plan.get("total_bytes", 0))
    coord_ep = getattr(args, "coord_endpoint", None) or os.getenv("HOTWEIGHTS_COORD")
    # metrics server for CLI
    try:
        from .telemetry.prom import start_http_server as _start
        port = int(os.getenv("HOTWEIGHTS_CLIENT_METRICS_PORT", "9097"))
        _start(port)
    except Exception:
        pass

    # Try CUDA-IPC path, fallback to MPI/UCX if CUDA unavailable
    use_cuda = False
    try:
        import torch  # type: ignore

        use_cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    except Exception:
        use_cuda = False

    if use_cuda:
        from .staging.cuda_ipc_agent import CudaIPCAgent
        from .transport.cuda_ipc import CudaIPCTransport
        from .telemetry.cuda_ipc_metrics import CudaIPCMetrics

        rank = int(os.getenv("RANK", "0"))
        metrics = CudaIPCMetrics(rank)
        agent = CudaIPCAgent(device=args.device, metrics=metrics)
        transport = CudaIPCTransport(agent=agent, metrics=metrics, coord_endpoint=coord_ep)
        print("Using SOTA CUDA-IPC Transport Layer.")
        with Timer("total") as total_timer:
            transport.replicate(plan)
        print(f"Replicated {len(plan.get('buckets', []))} buckets, total {total_bytes} bytes in {total_timer.elapsed:.3f}s.")
        if args.commit:
            print("Commit hint: Use apply_from_ipc_agent_to_module(agent, items, module, name_map) in your worker for GPU-native commit.")
    else:
        # Fallback to MPI/UCX streaming overlap path
        print("CUDA not available; falling back to MPI/UCX path.")
        from .staging.host_agent import HostAgent
        from .transport.mpi_stream import MPIReplicator
        from .transport.ucx_stream import UCXReplicator

        host = HostAgent(use_pinned=False)
        all_items: list[dict] = []
        replicator = None
        if getattr(args, "mpi", False):
            try:
                win = int(getattr(args, "window", 2))
                group = None
                gstr = getattr(args, "group", None) or os.getenv("HOTWEIGHTS_GROUP")
                if gstr:
                    group = [int(x) for x in gstr.split(",") if x.strip()]
                chunk_mb = int(getattr(args, "mpi_chunk_mb", 0))
                replicator = MPIReplicator(window=win, group_ranks=group, chunk_bytes=(chunk_mb << 20) if chunk_mb > 0 else 0)
            except Exception as e:
                print(f"MPI unavailable: {e}")
                replicator = None
        elif getattr(args, "ucx", False):
            try:
                replicator = UCXReplicator()
            except Exception as e:
                print(f"UCX unavailable: {e}")
                replicator = None

        if replicator is None:
            # Local assemble + scatter
            for b in plan.get("buckets", []):
                items = b["items"]
                buf = _assemble_bucket(items)
                _scatter_bucket(host, items, buf)
                if getattr(args, "verify", False):
                    _verify_items(host, items)
                all_items.extend(items)
        else:
            # Streaming scatter with overlap
            rank = getattr(replicator, "world_rank", getattr(replicator, "rank", 0))
            bucket_bufs: list[tuple[dict, np.ndarray]] = []

            def gen():
                import numpy as _np
                for b in plan.get("buckets", []):
                    items = b["items"]
                    size = int(b["size"])  # precomputed
                    consumers = b.get("consumer_ranks")
                    if consumers is not None and rank not in consumers:
                        continue
                    group_root = 0 if not consumers else min(int(x) for x in consumers)
                    if rank == group_root:
                        buf = _assemble_bucket(items)
                    else:
                        buf = _np.empty(size, dtype=_np.uint8)
                    bucket_bufs.append((b, buf))
                    if consumers is None:
                        yield (int(b["bucket_id"]), buf)
                    else:
                        yield (int(b["bucket_id"]), buf, list(int(x) for x in consumers))

            def on_complete(_bid: int, _buf: np.ndarray) -> None:
                b, buf = bucket_bufs.pop(0)
                items = b["items"]
                _scatter_bucket(host, items, buf)
                if getattr(args, "verify", False) and rank == (min(int(x) for x in b.get("consumer_ranks", [0])) if b.get("consumer_ranks") else 0):
                    _verify_items(host, items)
                all_items.extend(items)

            if hasattr(replicator, "replicate_stream"):
                replicator.replicate_stream(gen(), on_complete)  # type: ignore[attr-defined]
            else:
                replicator.replicate(gen())
                while bucket_bufs:
                    on_complete(0, bucket_bufs[0][1])

    # Submit/begin/commit outline
    if coord_ep:
        try:
            from .coordinator.zmq_client import Client

            c = Client(coord_ep)
            digest = _plan_digest(plan)
            print(c.call("submit_plan", plan=plan, digest=digest))
            print(c.call("begin", version=plan.get("version", "unknown"), digest=digest))
            print(c.call("commit", version=plan.get("version", "unknown")))
        except Exception as e:
            print(f"Coordinator unavailable: {e}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hotweights")
    sub = p.add_subparsers(dest="cmd", required=True)

    # publish
    sp = sub.add_parser("publish", help="Create a manifest for a checkpoint dir")
    sp.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    sp.add_argument("--version", required=True, help="Version string")
    sp.add_argument("--model-id", default="model", help="Model identifier")
    sp.add_argument("--output", default=None, help="Output manifest path")
    sp.set_defaults(func=_cmd_publish)

    # begin (optional coordinator)
    sp = sub.add_parser("begin", help="Broadcast begin event via coordinator (if available)")
    sp.add_argument("--version", required=True, help="Version string")
    sp.add_argument("--digest", default="", help="Manifest digest (optional)")
    sp.add_argument("--endpoint", default=None, help="ZeroMQ endpoint or HOTWEIGHTS_COORD env")
    def _cmd_begin(a: argparse.Namespace) -> int:
        endpoint = a.endpoint or os.getenv("HOTWEIGHTS_COORD")
        c = _maybe_coord_client(endpoint)
        if c is None:
            print("Coordinator not configured; no-op begin")
            return 0
        print(c.call("begin", version=a.version, digest=a.digest))
        return 0
    sp.set_defaults(func=_cmd_begin)

    # commit
    sp = sub.add_parser("commit", help="Commit a prepared update (stub)")
    sp.add_argument("--version", required=True, help="Version string")
    sp.add_argument("--endpoint", default=None, help="ZeroMQ endpoint or HOTWEIGHTS_COORD env")
    sp.set_defaults(func=_cmd_commit)

    # precommit
    sp = sub.add_parser("precommit", help="Precommit (workers ready) via coordinator")
    sp.add_argument("--worker-id", required=True)
    sp.add_argument("--endpoint", default=None)
    def _cmd_precommit(a: argparse.Namespace) -> int:
        c = _maybe_coord_client(a.endpoint or os.getenv("HOTWEIGHTS_COORD"))
        if c is None:
            print("Coordinator not configured")
            return 1
        print(c.call("precommit", worker_id=a.worker_id))
        return 0
    sp.set_defaults(func=_cmd_precommit)

    # abort
    sp = sub.add_parser("abort", help="Abort active update via coordinator")
    sp.add_argument("--reason", default="operator abort")
    sp.add_argument("--endpoint", default=None)
    def _cmd_abort(a: argparse.Namespace) -> int:
        c = _maybe_coord_client(a.endpoint or os.getenv("HOTWEIGHTS_COORD"))
        if c is None:
            print("Coordinator not configured")
            return 1
        print(c.call("abort", reason=a.reason))
        return 0
    sp.set_defaults(func=_cmd_abort)

    # status
    sp = sub.add_parser("status", help="Show system status (stub)")
    sp.add_argument("--endpoint", default=None, help="ZeroMQ endpoint or HOTWEIGHTS_COORD env")
    sp.set_defaults(func=_cmd_status)

    # plan
    sp = sub.add_parser("plan", help="Diff manifests and pack into buckets")
    sp.add_argument("--prev", required=True, help="Previous manifest path")
    sp.add_argument("--next", required=True, help="Next manifest path")
    sp.add_argument("--bucket-mb", type=int, default=512, help="Bucket size in MiB")
    sp.add_argument("--auto", action="store_true", help="Auto size bucket using min free VRAM * alpha")
    sp.add_argument("--alpha", type=float, default=0.25, help="Fraction for auto sizing")
    sp.add_argument("--group-map", default=None, help="JSON file mapping tensor patterns to consumer rank lists")
    sp.add_argument("--output", default=None, help="Output plan path")
    sp.set_defaults(func=_cmd_plan)

    # replicate (local)
    # replicate (local)
    sp = sub.add_parser("replicate", help="Replicate a plan using CUDA-IPC (fallback to MPI/UCX if needed).")
    sp.add_argument("--plan", required=True, help="Path to plan JSON")
    sp.add_argument("--commit", action="store_true", help="Ingest into adapter and commit (offline mode)")
    sp.add_argument("--device", default="cuda", help="Target device for staging and commit")
    sp.add_argument("--coord-endpoint", default=None, help="ZeroMQ coordinator endpoint for begin/commit")
    # Fallback options
    sp.add_argument("--mpi", action="store_true", help="Use MPI broadcast if available (fallback path)")
    sp.add_argument("--ucx", action="store_true", help="Use UCX broadcast if available (fallback path)")
    sp.add_argument("--verify", action="store_true", help="Verify hashes after staging (fallback path)")
    sp.add_argument("--window", type=int, default=2, help="MPI in-flight buckets window (fallback path)")
    sp.add_argument("--group", default=None, help="Comma-separated ranks for subgroup broadcast (MPI fallback)")
    sp.add_argument("--mpi-chunk-mb", type=int, default=0, help="MPI chunk size in MiB (fallback)")
    sp.set_defaults(func=_cmd_replicate)

    # bench (end-to-end)
    sp = sub.add_parser("bench", help="Run an end-to-end replication benchmark")
    sp.add_argument("--plan", required=True, help="Path to plan JSON")
    sp.add_argument("--device", default="cuda")
    sp.add_argument("--coord-endpoint", default=None)
    sp.add_argument("--fallback", action="store_true")
    sp.add_argument("--mpi", action="store_true")
    sp.add_argument("--window", type=int, default=2)
    sp.add_argument("--group", default=None)
    sp.add_argument("--mpi-chunk-mb", type=int, default=0)
    sp.add_argument("--verify", action="store_true")
    sp.add_argument("--output", default=None, help="Write JSON results to this file")
    def _cmd_bench(a: argparse.Namespace) -> int:
        import json as _json
        try:
            import torch  # type: ignore
            cuda_ok = torch.cuda.is_available() and a.device.startswith("cuda")
        except Exception:
            cuda_ok = False
        if not a.fallback and cuda_ok:
            from .staging.cuda_ipc_agent import CudaIPCAgent
            from .transport.cuda_ipc import CudaIPCTransport
            from .telemetry.cuda_ipc_metrics import CudaIPCMetrics
            plan = _json.loads(Path(a.plan).read_text())
            rank = int(os.getenv("RANK", "0"))
            metrics = CudaIPCMetrics(rank)
            agent = CudaIPCAgent(device=a.device, metrics=metrics)
            transport = CudaIPCTransport(agent=agent, metrics=metrics, coord_endpoint=a.coord_endpoint)
            t0 = time.perf_counter()
            transport.replicate(plan)
            dt = time.perf_counter() - t0
            res = {"mode": "cuda_ipc", "seconds": dt, "buckets": len(plan.get("buckets", [])), "bytes": int(plan.get("total_bytes", 0))}
            if a.output:
                Path(a.output).write_text(_json.dumps(res, indent=2))
            print(_json.dumps(res, indent=2))
            return 0
        else:
            # fallback path via MPI
            import numpy as _np
            from .staging.host_agent import HostAgent
            from .transport.mpi_stream import MPIReplicator
            from .transport.ucx_stream import UCXReplicator
            plan = _json.loads(Path(a.plan).read_text())
            host = HostAgent(use_pinned=False)
            all_items = []
            t0 = time.perf_counter()
            if a.mpi:
                ranks = None
                if a.group:
                    try:
                        ranks = [int(x) for x in a.group.split(",") if x.strip()]
                    except Exception:
                        ranks = None
                replicator = MPIReplicator(window=a.window, group_ranks=ranks, chunk_bytes=(a.mpi_chunk_mb << 20) if a.mpi_chunk_mb > 0 else 0)
                rank = getattr(replicator, "world_rank", getattr(replicator, "rank", 0))
                bucket_bufs: list[tuple[dict, _np.ndarray]] = []
                def gen():
                    for b in plan.get("buckets", []):
                        items = b["items"]
                        size = int(b["size"])  # precomputed
                        consumers = b.get("consumer_ranks")
                        if consumers is not None and rank not in consumers:
                            continue
                        group_root = 0 if not consumers else min(int(x) for x in consumers)
                        if rank == group_root:
                            buf = _assemble_bucket(items)
                        else:
                            buf = _np.empty(size, dtype=_np.uint8)
                        bucket_bufs.append((b, buf))
                        if consumers is None:
                            yield (int(b["bucket_id"]), buf)
                        else:
                            yield (int(b["bucket_id"]), buf, list(int(x) for x in consumers))
                def on_complete(_bid: int, _buf: _np.ndarray) -> None:
                    b, buf = bucket_bufs.pop(0)
                    items = b["items"]
                    _scatter_bucket(host, items, buf)
                    if a.verify and rank == (min(int(x) for x in b.get("consumer_ranks", [0])) if b.get("consumer_ranks") else 0):
                        _verify_items(host, items)
                    all_items.extend(items)
                replicator.replicate_stream(gen(), on_complete)
            else:
                for b in plan.get("buckets", []):
                    items = b["items"]
                    buf = _assemble_bucket(items)
                    _scatter_bucket(host, items, buf)
                    if a.verify:
                        _verify_items(host, items)
                    all_items.extend(items)
            dt = time.perf_counter() - t0
            res = {"mode": "fallback_mpi" if a.mpi else "fallback_local", "seconds": dt, "buckets": len(plan.get("buckets", [])), "bytes": int(plan.get("total_bytes", 0))}
            if a.output:
                Path(a.output).write_text(_json.dumps(res, indent=2))
            print(_json.dumps(res, indent=2))
            return 0
    sp.set_defaults(func=_cmd_bench)

    # metrics (start or print info)
    sp = sub.add_parser("metrics", help="Start or inspect metrics endpoints")
    sp.add_argument("--start", action="store_true", help="Start a local metrics server")
    sp.add_argument("--port", type=int, default=None, help="Port for local metrics server (default 9097)")
    sp.add_argument("--print", action="store_true", help="Print recommended ports and env vars")
    def _cmd_metrics(a: argparse.Namespace) -> int:
        if a.print:
            print("Worker metrics: HOTWEIGHTS_METRICS_PORT (default 9099)")
            print("Coordinator metrics: HOTWEIGHTS_COORD_METRICS_PORT (default 9100)")
            print("CLI client metrics: HOTWEIGHTS_CLIENT_METRICS_PORT (default 9097)")
            print("Handle token (optional): HOTWEIGHTS_HANDLE_TOKEN (HMAC-SHA256 of handle payload)")
        if a.start:
            try:
                from .telemetry.prom import start_http_server as _start
                port = a.port or int(os.getenv("HOTWEIGHTS_CLIENT_METRICS_PORT", "9097"))
                _start(port)
                print(f"Metrics server started on port {port}")
            except Exception as e:
                print(f"Failed to start metrics server: {e}")
                return 1
        return 0
    sp.set_defaults(func=_cmd_metrics)

    # tp-groups: generate a simple contiguous TP group mapping
    sp = sub.add_parser("tp-groups", help="Generate a contiguous TP group mapping JSON")
    sp.add_argument("--world-size", type=int, required=True)
    sp.add_argument("--tp", type=int, required=True)
    sp.add_argument("--output", default=None, help="Write mapping JSON to this file")
    def _cmd_tpg(a: argparse.Namespace) -> int:
        import json as _json
        ws, tp = int(a.world_size), int(a.tp)
        if tp <= 0 or ws <= 0 or ws % tp != 0:
            print("Invalid parameters: world-size must be divisible by tp")
            return 1
        groups = ws // tp
        mapping = {}
        r = 0
        for g in range(groups):
            ranks = list(range(r, r + tp))
            mapping[str(g)] = ranks
            r += tp
        js = _json.dumps(mapping, indent=2)
        if a.output:
            Path(a.output).write_text(js)
        print(js)
        return 0
    sp.set_defaults(func=_cmd_tpg)

    # coordinator (optional ZeroMQ)
    sp = sub.add_parser("coord-serve", help="Run HA coordinator (requires pyzmq)")
    sp.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    sp.add_argument("--pub-endpoint", default=None, help="PUB endpoint (default: endpoint port+1)")
    sp.add_argument("--instance-id", default=None, help="Unique ID for this coordinator instance")
    def _coord_serve(a: argparse.Namespace) -> int:
        try:
            from .coordinator.ha_control_plane import serve
            import uuid
        except Exception as e:  # pragma: no cover - optional path
            print(f"Coordinator unavailable: {e}")
            return 1
        instance_id = a.instance_id or f"coord-{uuid.uuid4().hex[:6]}"
        pub_ep = a.pub_endpoint or _derive_pub_endpoint(a.endpoint)
        print(f"Serving HA coordinator instance {instance_id} on {a.endpoint} (Ctrl+C to stop)")
        serve(a.endpoint, pub_ep, instance_id)
        return 0
    sp.set_defaults(func=_coord_serve)

    sp = sub.add_parser("coord-status", help="Query coordinator status (requires pyzmq)")
    sp.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    def _coord_status(a: argparse.Namespace) -> int:
        try:
            from .coordinator.zmq_client import Client
            c = Client(a.endpoint)
            print(c.call("status"))
            return 0
        except Exception as e:  # pragma: no cover - optional path
            print(f"Coordinator unavailable: {e}")
            return 1
    sp.set_defaults(func=_coord_status)

    sp = sub.add_parser("coord-submit-plan", help="Submit a plan to coordinator (requires pyzmq)")
    sp.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    sp.add_argument("--plan", required=True)
    def _coord_submit(a: argparse.Namespace) -> int:
        try:
            from .coordinator.zmq_client import Client
            import json as _json
            c = Client(a.endpoint)
            plan = _json.loads(Path(a.plan).read_text())
            digest = _plan_digest(plan)
            print(c.call("submit_plan", plan=plan, digest=digest))
            return 0
        except Exception as e:  # pragma: no cover
            print(f"Coordinator unavailable: {e}")
            return 1
    sp.set_defaults(func=_coord_submit)

    # coordinator subscribe (PUB/SUB)
    sp = sub.add_parser("coord-sub", help="Subscribe to coordinator events (requires pyzmq)")
    sp.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    sp.add_argument("--topics", nargs="*", default=None, help="Topics to subscribe (e.g., begin commit have)")
    sp.add_argument("--print-last", action="store_true", help="Print last snapshot of key events via status and exit")
    def _coord_sub(a: argparse.Namespace) -> int:
        try:
            if a.print_last:
                from .coordinator.zmq_client import Client as _C
                c = _C(a.pub_endpoint.replace(":5556", ":5555")) if ":" in a.pub_endpoint else _C()
                st = c.call("status")
                print(st.get("last_events", {}))
                return 0
            from .coordinator.zmq_client import Subscriber
            sub = Subscriber(a.pub_endpoint, topics=a.topics)
            print(f"Subscribed to {a.pub_endpoint} topics={a.topics or ['*']}")
            while True:
                topic, payload = sub.recv()
                print(topic, payload)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            return 0
        except Exception as e:  # pragma: no cover - optional path
            print(f"Subscriber unavailable: {e}")
            return 1
    sp.set_defaults(func=_coord_sub)

    # worker agent (ZeroMQ + CUDA-IPC)
    sp = sub.add_parser("worker", help="Run worker agent (requires pyzmq)")
    sp.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    sp.add_argument("--device", default="cuda", help="Target device for staging (cuda or cuda:N)")
    sp.add_argument("--no-sub", action="store_true", help="Disable SUB listener; use REQ/REP only")
    sp.add_argument("--pub-endpoint", default=None, help="PUB endpoint for events (default: endpoint port+1)")
    sp.add_argument("--event-token", default=None, help="Shared secret token for event validation")
    def _cmd_worker(a: argparse.Namespace) -> int:
        try:
            from .worker.agent import run_worker, WorkerConfig
            cfg = WorkerConfig(
                endpoint=a.endpoint,
                device=a.device,
                use_sub=not a.no_sub,
                pub_endpoint=a.pub_endpoint,
                event_token=a.event_token,
            )
            return run_worker(cfg)
        except Exception as e:  # pragma: no cover
            print(f"Worker failed: {e}")
            return 1
    sp.set_defaults(func=_cmd_worker)

    # bucket size suggestion (telemetry)
    sp = sub.add_parser("bucket-suggest", help="Suggest bucket size from free VRAM")
    sp.add_argument("--alpha", type=float, default=0.25, help="Fraction of min free VRAM")
    def _cmd_bucket(a: argparse.Namespace) -> int:
        free = min_free_vram_mib()
        if free is None:
            print("Unknown (no GPU or NVML)")
            return 1
        print(int(free * a.alpha))
        return 0
    sp.set_defaults(func=_cmd_bucket)

    # SOTA Tensor Mapping Wizard
    sp = sub.add_parser("map-wizard", help="Run interactive tensor mapping wizard.")
    sp.add_argument("--manifest", required=True, help="Path to the new manifest file.")
    sp.add_argument("--model", required=True, help="Path to the model file (or name for lookup).")
    sp.add_argument("--output", required=True, help="Path to save the final mapping JSON file.")
    def _cmd_map_wizard(a: argparse.Namespace) -> int:
        try:
            from .cli_wizards.mapping_wizard import run_wizard
        except ImportError as e:
            print(f"Fatal: Could not load mapping wizard. Error: {e}")
            return 1
        run_wizard(a.manifest, a.model, a.output)
        return 0
    sp.set_defaults(func=_cmd_map_wizard)

    # kv-check: derive/validate KV head map
    sp = sub.add_parser("kv-check", help="Inspect and derive KV head remapping for GQA")
    sp.add_argument("--heads", type=int, required=True, help="Total attention heads (H)")
    sp.add_argument("--kv-heads", type=int, default=0, help="Key/Value heads (KvH)")
    sp.add_argument("--order", choices=["grouped", "interleaved"], default="grouped", help="Derived head order")
    sp.add_argument("--map-file", default=None, help="Optional JSON file with explicit head map")
    sp.add_argument("--map", default=None, help="Optional inline JSON head map (overrides file)")
    sp.add_argument("--simulate", action="store_true", help="Simulate applying the map on dummy tensors (if torch available)")
    def _cmd_kv_check(a: argparse.Namespace) -> int:
        from .adapters.kv_cache_migration import derive_head_map
        try:
            from .adapters.kv_cache_migration import _validate_head_map as _val
        except Exception:
            _val = None
        H = int(a.heads)
        KvH = int(a.kv_heads) if a.kv_heads else None
        explicit = None
        try:
            if a.map:
                explicit = json.loads(a.map)
            elif a.map_file:
                explicit = json.loads(Path(a.map_file).read_text())
        except Exception as e:
            print(json.dumps({"error": f"failed to parse map: {e}"}, indent=2))
            return 1
        if explicit is not None:
            compat = isinstance(explicit, list) and len(explicit) == H and all(isinstance(x, int) for x in explicit)
            perm_ok = False
            if _val and compat:
                perm_ok, _ = _val(explicit, H)
            out = {
                "heads": H,
                "kv_heads": KvH,
                "order": a.order,
                "explicit_map": explicit,
                "compatible": bool(compat),
                "is_permutation": bool(perm_ok) if compat else False,
                "note": "explicit map provided" if compat else "invalid explicit map length or types",
            }
        else:
            derived = derive_head_map(H, KvH, order=a.order)
            compat = (KvH is None) or (H % KvH == 0)
            perm_ok = False
            if _val:
                perm_ok, _ = _val(derived, H)
            out = {
                "heads": H,
                "kv_heads": KvH,
                "order": a.order,
                "derived_map": derived,
                "compatible": bool(compat),
                "is_permutation": bool(perm_ok),
                "note": "identity map (incomplete params)" if KvH in (None, 0) else ("derived grouped map" if a.order == "grouped" else "derived interleaved map"),
            }
        # Optional simulation
        if a.simulate:
            try:
                import torch as _torch
                L, D = 16, 64
                if explicit is not None and compat:
                    m = explicit
                else:
                    m = out.get("derived_map")
                if not (isinstance(m, list) and len(m) == H):
                    out["simulate"] = {"applied": False, "reason": "no valid map"}
                else:
                    K = _torch.randn(H, L, D)
                    V = _torch.randn(H, L, D)
                    idx = _torch.as_tensor(m)
                    K2 = _torch.index_select(K, 0, idx)
                    V2 = _torch.index_select(V, 0, idx)
                    out["simulate"] = {"applied": True, "k_shape": list(K2.shape), "v_shape": list(V2.shape)}
            except Exception as e:
                out["simulate"] = {"applied": False, "reason": f"simulation unavailable: {e}"}
        print(json.dumps(out, indent=2))
        return 0
    sp.set_defaults(func=_cmd_kv_check)

    # opt-check: preview optimizer policy effects
    sp = sub.add_parser("opt-check", help="Preview optimizer state policy effects for updated/unchanged params")
    sp.add_argument("--updated", type=int, default=0, help="Count of updated parameters")
    sp.add_argument("--unchanged", type=int, default=0, help="Count of unchanged parameters")
    sp.add_argument("--policy", choices=["preserve", "reset", "attenuate"], default=None, help="Override HOTWEIGHTS_OPT_POLICY")
    sp.add_argument("--attenuation", type=float, default=None, help="Override HOTWEIGHTS_OPT_ATTENUATION")
    def _cmd_opt_check(a: argparse.Namespace) -> int:
        import os as _os
        pol = (a.policy or _os.getenv("HOTWEIGHTS_OPT_POLICY", "preserve")).lower()
        try:
            att = float(a.attenuation if a.attenuation is not None else _os.getenv("HOTWEIGHTS_OPT_ATTENUATION", "1.0"))
        except Exception:
            att = 1.0
        upd = max(0, int(a.updated)); unc = max(0, int(a.unchanged))
        preserved = unc + (upd if pol == "preserve" else 0)
        attenuated = upd if pol == "attenuate" else 0
        reinitialized = upd if pol == "reset" else 0
        out = {
            "policy": pol,
            "attenuation": att,
            "updated": upd,
            "unchanged": unc,
            "expected": {
                "preserved": preserved,
                "attenuated": attenuated,
                "reinitialized": reinitialized,
            },
            "notes": "Counts are approximate; actual behavior respects param shapes and shards.",
        }
        print(json.dumps(out, indent=2))
        return 0
    sp.set_defaults(func=_cmd_opt_check)

    # verify-plan: sanity checks on plan JSON
    sp = sub.add_parser("verify-plan", help="Validate a plan for consumer ranks, world-size bounds, and digest")
    sp.add_argument("--plan", required=True, help="Path to plan JSON")
    sp.add_argument("--world-size", type=int, default=None, help="World size for rank validation")
    sp.add_argument("--tp-groups", default=None, help="JSON string or file with TP group mapping (optional)")
    sp.add_argument("--require-consumers", action="store_true", help="Require consumer_ranks to be present on all buckets")
    sp.add_argument("--manifest-next", default=None, help="Path to next manifest (optional, to validate TP group superset)")
    sp.add_argument("--enforce-tp-superset", action="store_true", help="If manifest has tp_group and tp-groups provided, require consumer_ranks to be a superset of the group's ranks")
    def _cmd_verify_plan(a: argparse.Namespace) -> int:
        import json as _json
        from pathlib import Path as _Path
        plan = _json.loads(_Path(a.plan).read_text())
        ws = a.world_size
        # Parse TP groups mapping if provided or via env
        tp_groups = None
        cfg = a.tp_groups or os.getenv("HOTWEIGHTS_TP_GROUPS", "")
        if cfg:
            try:
                if cfg.strip().startswith("{"):
                    tp_groups = _json.loads(cfg)
                else:
                    tp_groups = _json.loads(_Path(cfg).read_text())
            except Exception:
                tp_groups = None
        problems = []
        warnings_ = []
        total = 0
        missing = 0
        out_of_range = 0
        empty = 0
        dupes = 0
        # Build manifest tensor->group map if provided
        tensor_group: dict[str, int] = {}
        if a.manifest_next:
            try:
                man = _json.loads(_Path(a.manifest_next).read_text())
                for t in man.get("tensors", []):
                    part = t.get("partitioning") or {}
                    gid = part.get("tp_group") or part.get("group") or part.get("group_id")
                    if gid is not None:
                        tensor_group[str(t.get("name"))] = int(gid)
            except Exception:
                pass
        for b in plan.get("buckets", []):
            total += 1
            ranks = b.get("consumer_ranks")
            if ranks is None:
                if a.require_consumers:
                    missing += 1
                    problems.append({"bucket_id": b.get("bucket_id"), "error": "missing consumer_ranks"})
                continue
            if isinstance(ranks, list):
                if len(ranks) == 0:
                    empty += 1
                    warnings_.append({"bucket_id": b.get("bucket_id"), "warning": "empty consumer_ranks"})
                if len(set(int(x) for x in ranks)) != len(ranks):
                    dupes += 1
                    warnings_.append({"bucket_id": b.get("bucket_id"), "warning": "duplicate ranks in consumer_ranks"})
                if ws is not None:
                    bad = [int(x) for x in ranks if int(x) < 0 or int(x) >= ws]
                    if bad:
                        out_of_range += 1
                        problems.append({"bucket_id": b.get("bucket_id"), "error": f"ranks out of range [0,{ws-1}]", "bad": bad})
                if tp_groups and isinstance(tp_groups, dict):
                    try:
                        valid = {int(x) for v in tp_groups.values() for x in v}
                        extra = [int(x) for x in ranks if int(x) not in valid]
                        if extra:
                            warnings_.append({"bucket_id": b.get("bucket_id"), "warning": "ranks not in TP groups union", "extra": extra})
                        # Superset enforcement per tensor group if requested
                        if a.enforce_tp_superset and tensor_group:
                            # Find any tensor groups present in this bucket
                            groups_in_bucket = set()
                            for it in b.get("items", []):
                                gid = tensor_group.get(it.get("tensor"))
                                if gid is not None:
                                    groups_in_bucket.add(int(gid))
                            for gid in groups_in_bucket:
                                group_ranks = set(int(x) for x in tp_groups.get(str(gid), []))
                                if group_ranks and not group_ranks.issubset(set(int(x) for x in ranks)):
                                    problems.append({"bucket_id": b.get("bucket_id"), "error": "consumer_ranks not superset of TP group", "group": gid})
                    except Exception:
                        pass
        digest = _plan_digest(plan)
        result = {
            "buckets": total,
            "missing_consumer_ranks": missing,
            "empty_consumer_ranks": empty,
            "buckets_with_out_of_range": out_of_range,
            "buckets_with_duplicates": dupes,
            "problems": problems,
            "warnings": warnings_,
            "digest": digest,
        }
        print(_json.dumps(result, indent=2))
        return 0 if not problems else 1
    sp.set_defaults(func=_cmd_verify_plan)

    # verify-tp: validate TP groups mapping against world size
    sp = sub.add_parser("verify-tp", help="Validate TP group mapping against world size")
    sp.add_argument("--tp-groups", required=True, help="JSON string or file with TP group mapping")
    sp.add_argument("--world-size", type=int, required=True)
    def _cmd_verify_tp(a: argparse.Namespace) -> int:
        import json as _json
        from pathlib import Path as _Path
        cfg = a.tp_groups
        if not cfg:
            print(_json.dumps({"error": "--tp-groups required"}, indent=2))
            return 1
        try:
            if cfg.strip().startswith("{"):
                groups = _json.loads(cfg)
            else:
                groups = _json.loads(_Path(cfg).read_text())
        except Exception as e:
            print(_json.dumps({"error": f"failed to parse tp-groups: {e}"}, indent=2))
            return 1
        ws = int(a.world_size)
        problems = []
        warnings_ = []
        seen = set()
        overlap = []
        for gid, ranks in groups.items():
            try:
                r = [int(x) for x in ranks]
            except Exception:
                problems.append({"group": gid, "error": "non-integer rank values"})
                continue
            # range check
            bad = [x for x in r if x < 0 or x >= ws]
            if bad:
                problems.append({"group": gid, "error": f"ranks out of range [0,{ws-1}]", "bad": bad})
            # duplicates within group
            if len(set(r)) != len(r):
                warnings_.append({"group": gid, "warning": "duplicate ranks within group"})
            # overlaps with previous groups
            dup = [x for x in r if x in seen]
            if dup:
                overlap.append({"group": gid, "overlap": dup})
            seen.update(r)
        out = {
            "groups": len(groups),
            "world_size": ws,
            "problems": problems,
            "overlaps": overlap,
            "warnings": warnings_,
        }
        print(_json.dumps(out, indent=2))
        return 0 if not problems else 1
    sp.set_defaults(func=_cmd_verify_tp)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""hotweights CLI."""
from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

from .manifest import build_simple_manifest, dump_manifest, load_manifest
from .core.replicate import (
    create_plan as _create_plan_core,
    create_plan_from_current as _create_plan_from_current,
    assemble_bucket as _assemble_bucket,
    scatter_bucket as _scatter_bucket,
    verify_items as _verify_items,
    plan_digest as _plan_digest,
    verify_plan as _verify_plan_core,
)
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


def _load_consumer_map(arg_value: str | None) -> dict | None:
    if not arg_value:
        return None
    try:
        return json.loads(Path(arg_value).read_text())
    except Exception as e:
        print(f"Failed to load group map {arg_value}: {e}")
        return None


def _cmd_plan(args: argparse.Namespace) -> int:
    # Optional warmup of planner JIT to reduce first-call latency
    try:
        if os.getenv("HOTWEIGHTS_NO_WARMUP", "0") not in ("1", "true", "True"):
            from . import planner_bodo as _pb

            _pb.warmup()
    except Exception:
        pass
    bucket_mb = args.bucket_mb
    consumer_map = _load_consumer_map(getattr(args, "group_map", None) or os.getenv("HOTWEIGHTS_GROUP_MAP"))
    if args.auto:
        free = min_free_vram_mib()
        if free:
            bucket_mb = max(1, int(free * args.alpha))
            print(f"Auto bucket size MiB: {bucket_mb} (free={free}MiB, alpha={args.alpha})")

    nxt = load_manifest(args.next)

    # Determine prev manifest: explicit --prev, coordinator current, or local cache
    prev: dict | None = None
    if getattr(args, "prev", None):
        prev = load_manifest(args.prev)
    else:
        endpoint = getattr(args, "coord_endpoint", None) or os.getenv("HOTWEIGHTS_COORD")
        c = _maybe_coord_client(endpoint)
        if c is not None:
            try:
                resp = c.call("get_current_manifest")
                prev = resp.get("manifest") if isinstance(resp, dict) else None
            except Exception:
                prev = None
        if prev is None:
            # Fallback local cache
            cache_path = Path(os.path.expanduser("~/.cache/hotweights/current_manifest.json"))
            if cache_path.exists():
                try:
                    prev = json.loads(cache_path.read_text())
                except Exception:
                    prev = None
    if prev is None:
        print("No previous manifest available. Provide --prev or run a coordinator with a current manifest.")
        return 1

    plan = _create_plan_core(prev, nxt, bucket_mb=bucket_mb, consumer_map=consumer_map)

    # Built-in verification (default on; use --no-verify to skip, --strict to fail on problems)
    if not getattr(args, "no_verify", False):
        ws_env = os.getenv("WORLD_SIZE")
        world_size = int(ws_env) if (ws_env and ws_env.isdigit()) else None
        tp_groups = None
        tp_cfg = os.getenv("HOTWEIGHTS_TP_GROUPS")
        if tp_cfg:
            try:
                if tp_cfg.strip().startswith("{"):
                    tp_groups = json.loads(tp_cfg)
                else:
                    tp_groups = json.loads(Path(tp_cfg).read_text())
            except Exception:
                tp_groups = None
        report = _verify_plan_core(plan, require_consumers=getattr(args, "require_consumers", False), world_size=world_size, tp_groups=tp_groups, enforce_tp_superset=getattr(args, "enforce_tp_superset", False))
        plan["verification"] = report
        if report.get("problems") and getattr(args, "strict", False):
            print(json.dumps(report, indent=2))
            return 1

    out = Path(args.output or "transfer.plan.json")
    out.write_text(json.dumps(plan, indent=2))
    print(f"Wrote plan: {out} ({plan['total_bytes']} bytes across {len(plan['buckets'])} buckets)")
    return 0


# Note: bucket assemble/scatter/verify utilities are imported from core.replicate


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
        # Fallback to auto-selected CPU transport (MPI/UCX) or local
        print("CUDA not available; selecting CPU transport (auto).")
        from .staging.host_agent import HostAgent
        from .transport.transport_manager import TransportManager

        host = HostAgent(use_pinned=False)
        all_items: list[dict] = []

        # Discover world size and rank for informative behavior
        world_size = 1
        rank = 0
        try:
            from mpi4py import MPI as _MPI  # type: ignore

            world_size = _MPI.COMM_WORLD.Get_size()
            rank = _MPI.COMM_WORLD.Get_rank()
        except Exception:
            try:
                ws_env = int(os.getenv("WORLD_SIZE", "1"))
                rk_env = int(os.getenv("RANK", "0"))
                world_size, rank = max(1, ws_env), max(0, rk_env)
            except Exception:
                world_size, rank = 1, 0

        replicator = None
        try:
            preferred = None
            if getattr(args, "mpi", False):
                print("[deprecated] --mpi flag: transport is auto-selected now; honoring preference for this run.")
                preferred = 'mpi'
            elif getattr(args, "ucx", False):
                print("[deprecated] --ucx flag: transport is auto-selected now; honoring preference for this run.")
                preferred = 'ucx'
            tm = TransportManager(world_size=world_size, rank=rank, auto_select=True, preferred_transport=preferred)
            replicator = tm.get_replicator()
        except Exception as e:
            print(f"Transport auto-selection failed: {e}")
            replicator = None

        if not replicator or world_size <= 1:
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
            # Forward optional token via env HOTWEIGHTS_COORD_TOKEN
            token = os.getenv("HOTWEIGHTS_COORD_TOKEN")
            print(c.call("submit_plan", plan=plan, digest=digest, token=token))
            print(c.call("begin", version=plan.get("version", "unknown"), digest=digest, token=token))
            print(c.call("commit", version=plan.get("version", "unknown"), token=token))
            # Optionally persist current manifest
            if getattr(args, "manifest_next", None):
                try:
                    nxt_manifest = json.loads(Path(args.manifest_next).read_text())
                    print(c.call("set_current_manifest", manifest=nxt_manifest, token=os.getenv("HOTWEIGHTS_COORD_TOKEN")))
                    cache_dir = Path(os.path.expanduser("~/.cache/hotweights"))
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    (cache_dir / "current_manifest.json").write_text(json.dumps(nxt_manifest))
                except Exception as e:
                    print(f"Failed to persist current manifest: {e}")
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
    sp = sub.add_parser("plan", help="Diff manifests and pack into buckets (prev optional via coordinator)")
    sp.add_argument("--prev", required=False, help="Previous manifest path (optional)")
    sp.add_argument("--next", required=True, help="Next manifest path")
    sp.add_argument("--bucket-mb", type=int, default=512, help="Bucket size in MiB")
    sp.add_argument("--auto", action="store_true", help="Auto size bucket using min free VRAM * alpha")
    sp.add_argument("--alpha", type=float, default=0.25, help="Fraction for auto sizing")
    sp.add_argument("--group-map", default=None, help="JSON file mapping tensor patterns to consumer rank lists")
    sp.add_argument("--coord-endpoint", default=None, help="ZeroMQ coordinator endpoint for current manifest lookup")
    sp.add_argument("--no-verify", action="store_true", help="Skip built-in verification")
    sp.add_argument("--strict", action="store_true", help="Fail if verification finds problems")
    sp.add_argument("--output", default=None, help="Output plan path")
    sp.set_defaults(func=_cmd_plan)

    # replicate (local)
    # replicate (local)
    sp = sub.add_parser("replicate", help="Replicate a plan using CUDA-IPC (fallback to MPI/UCX if needed).")
    sp.add_argument("--plan", required=True, help="Path to plan JSON")
    sp.add_argument("--commit", action="store_true", help="Ingest into adapter and commit (offline mode)")
    sp.add_argument("--device", default="cuda", help="Target device for staging and commit")
    sp.add_argument("--coord-endpoint", default=None, help="ZeroMQ coordinator endpoint for begin/commit")
    sp.add_argument("--manifest-next", default=None, help="Path to next manifest (persist as current after commit)")
    # Fallback options
    sp.add_argument("--mpi", action="store_true", help=argparse.SUPPRESS)
    sp.add_argument("--ucx", action="store_true", help=argparse.SUPPRESS)
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

    # coordinator: set current manifest explicitly
    sp = sub.add_parser("coord-set-current", help="Set current manifest on coordinator (requires pyzmq)")
    sp.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    sp.add_argument("--manifest", required=True, help="Path to manifest JSON to set as current")
    def _coord_set_current(a: argparse.Namespace) -> int:
        try:
            from .coordinator.zmq_client import Client
            import json as _json
            token = os.getenv("HOTWEIGHTS_COORD_TOKEN")
            c = Client(a.endpoint)
            m = _json.loads(Path(a.manifest).read_text())
            print(c.call("set_current_manifest", manifest=m, token=token))
            return 0
        except Exception as e:  # pragma: no cover
            print(f"Coordinator unavailable: {e}")
            return 1
    sp.set_defaults(func=_coord_set_current)

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
        report = _verify_plan_core(
            plan,
            require_consumers=a.require_consumers,
            world_size=ws,
            tp_groups=tp_groups,
            enforce_tp_superset=a.enforce_tp_superset,
        )
        print(_json.dumps(report, indent=2))
        return 0 if not report.get("problems") else 1
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

    # bench-hierarchical: synthetic stage timing for NCCL+IPC pipeline (leaders only)
    sp = sub.add_parser("bench-hierarchical", help="Measure H2D/NCCL/intra-node device copy timings (leaders only)")
    sp.add_argument("--size-mb", type=int, default=256, help="Total bytes per bucket (MiB)")
    sp.add_argument("--repeat", type=int, default=3)
    def _cmd_bench_hier(a: argparse.Namespace) -> int:
        try:
            import torch  # type: ignore
            import torch.distributed as dist  # type: ignore
        except Exception as e:
            print(f"Torch with CUDA required: {e}")
            return 1
        if not torch.cuda.is_available():
            print("CUDA not available")
            return 1
        # Init dist if needed
        try:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        except Exception as e:
            print(f"Failed to init process group: {e}")
            return 1
        ws = dist.get_world_size()
        rk = dist.get_rank()
        try:
            lrank = int(os.getenv("LOCAL_RANK", "0"))
            lws = int(os.getenv("LOCAL_WORLD_SIZE", "0"))
        except Exception:
            lrank, lws = 0, 0
        # Compute leaders
        leaders = [0]
        if lws and ws % lws == 0:
            leaders = list(range(0, ws, lws))
        root = min(leaders)
        group = dist.new_group(ranks=leaders) if len(leaders) > 1 else dist.group.WORLD  # type: ignore[attr-defined]
        is_leader = (lrank == 0)
        dev = torch.device("cuda", torch.cuda.current_device())
        size = max(1, int(a.size_mb)) * (1 << 20)
        results = []
        for _ in range(int(a.repeat)):
            out = {"rank": rk, "is_leader": is_leader, "size_bytes": size}
            if is_leader:
                # H2D
                cpu = torch.empty(size, dtype=torch.uint8, pin_memory=True)
                gpu = torch.empty(size, dtype=torch.uint8, device=dev)
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record(); gpu.copy_(cpu, non_blocking=True); torch.cuda.synchronize(); t1.record(); t1.synchronize()
                out["h2d_ms"] = float(t0.elapsed_time(t1))
                # Inter-node (leaders)
                t2 = torch.cuda.Event(enable_timing=True)
                t3 = torch.cuda.Event(enable_timing=True)
                t2.record()
                if rk == root:
                    dist.broadcast(gpu, src=root, group=group)
                else:
                    dst = torch.empty(size, dtype=torch.uint8, device=dev)
                    dist.broadcast(dst, src=root, group=group)
                    gpu = dst
                torch.cuda.synchronize(); t3.record(); t3.synchronize()
                out["inter_ms"] = float(t2.elapsed_time(t3))
                # Intra-node (proxy: device->device copy) and reload (another copy)
                dst1 = torch.empty_like(gpu)
                t4 = torch.cuda.Event(enable_timing=True); t5 = torch.cuda.Event(enable_timing=True)
                t4.record(); dst1.copy_(gpu, non_blocking=True); torch.cuda.synchronize(); t5.record(); t5.synchronize()
                out["intra_ms"] = float(t4.elapsed_time(t5))
                dst2 = torch.empty_like(gpu)
                t6 = torch.cuda.Event(enable_timing=True); t7 = torch.cuda.Event(enable_timing=True)
                t6.record(); dst2.copy_(dst1, non_blocking=True); torch.cuda.synchronize(); t7.record(); t7.synchronize()
                out["reload_ms"] = float(t6.elapsed_time(t7))
            else:
                out["skipped"] = True
            results.append(out)
        import json as _json
        print(_json.dumps({"world_size": ws, "results": results}, indent=2))
        return 0
    sp.set_defaults(func=_cmd_bench_hier)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

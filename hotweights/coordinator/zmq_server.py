"""ZeroMQ-based coordinator server (REP + PUB).

Adds a PUB socket to broadcast events like begin/commit/abort and progress
updates so workers can subscribe without polling.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time
import socket
from ..telemetry.prom import Counter, Gauge, start_http_server
from ..telemetry.logging import get_logger

try:  # optional
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional
    zmq = None  # type: ignore


@dataclass
class State:
    version: str | None = None
    workers: Dict[str, dict] = field(default_factory=dict)
    plan: Dict[str, Any] | None = None
    plan_digest: str | None = None
    current_manifest: Dict[str, Any] | None = None
    have: Dict[int, list[int]] = field(default_factory=dict)  # bucket_id -> ranks
    precommit_acks: Dict[str, bool] = field(default_factory=dict)
    heartbeats: Dict[str, float] = field(default_factory=dict)
    rpc_total: Dict[str, int] = field(default_factory=dict)
    last_events: Dict[str, Any] = field(default_factory=dict)
    state: str = "idle"  # idle|begun|precommit|committed|aborted
    # Handle registry: version -> bucket_id -> node_id -> {handle, sig, ts, acks}
    handles: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = field(default_factory=dict)


def _derive_pub_endpoint(endpoint: str) -> str:
    """Derive a PUB endpoint from a REP endpoint by incrementing port.

    Only handles tcp://host:port. Falls back to tcp://127.0.0.1:5556.
    """
    try:
        if endpoint.startswith("tcp://") and ":" in endpoint.rsplit(":", 1)[-1]:
            host, port_s = endpoint[len("tcp://") :].rsplit(":", 1)
            port = int(port_s)
            return f"tcp://{host}:{port+1}"
    except Exception:
        pass
    return "tcp://127.0.0.1:5556"


def serve(
    endpoint: str = "tcp://127.0.0.1:5555",
    pub_endpoint: Optional[str] = None,
    event_token: Optional[str] = None,
) -> None:
    if zmq is None:
        raise RuntimeError("pyzmq not installed; install with 'pip install .[extras]'")
    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(endpoint)
    # PUB socket for broadcasting events
    pub_ep = pub_endpoint or _derive_pub_endpoint(endpoint)
    pub = ctx.socket(zmq.PUB)
    pub.bind(pub_ep)
    st = State()
    # Metrics
    try:
        start_http_server(9100)
    except Exception:
        pass
    g_workers = Gauge("hotweights_coord_workers", "Registered workers")
    g_acks = Gauge("hotweights_coord_precommit_acks", "Precommit acknowledgements")
    g_have_buckets = Gauge("hotweights_coord_have_buckets", "Buckets with HAVE reports")
    def _pub(topic: str, payload: Dict[str, Any]) -> None:
        try:
            if event_token:
                payload = dict(payload)
                payload["tok"] = event_token
            pub.send_multipart([topic.encode("utf-8"), json.dumps(payload).encode("utf-8")])
            # keep last snapshot for key topics
            if topic in ("plan", "begin", "commit", "abort"):
                st.last_events[topic] = payload
        except Exception:
            pass

    log = get_logger("ZMQServer")
    while True:  # simple loop, Ctrl+C to stop
        msg = rep.recv_json()
        method = msg.get("method")
        args = msg.get("args", {})
        # metrics: rpc counter
        st.rpc_total[method] = st.rpc_total.get(method, 0) + 1

        # Helper: require token for mutating RPCs when configured
        def _check_token() -> bool:
            if not event_token:
                return True
            tok = args.get("token")
            return bool(tok == event_token)

        if method == "register":
            wid = args.get("worker_id", "?")
            caps = args.get("caps", {})
            # enrich with hostname if missing
            caps.setdefault("host", socket.gethostname())
            st.workers[wid] = caps
            rep.send_json({"ok": True, "current_version": st.version, "workers": st.workers})
            try:
                log.info(f"register worker_id={wid}")
            except Exception:
                pass
            _pub("register", {"worker_id": wid, "caps": caps})
        elif method == "heartbeat":
            wid = args.get("worker_id", "?")
            st.heartbeats[wid] = time.time()
            rep.send_json({"ok": True})
        elif method == "begin":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            st.version = args.get("version")
            st.plan_digest = args.get("digest") or st.plan_digest
            # reset precommit and have tracking for the new version
            st.precommit_acks.clear()
            st.have.clear()
            st.state = "begun"
            payload = {"event": "begin", "version": st.version, "digest": st.plan_digest}
            rep.send_json(payload)
            try:
                log.info(f"begin version={st.version}")
            except Exception:
                pass
            _pub("begin", payload)
        elif method == "precommit":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            wid = args.get("worker_id", "?")
            st.precommit_acks[wid] = True
            st.state = "precommit"
            rep.send_json({"ok": True, "acks": len(st.precommit_acks)})
            try:
                log.info(f"precommit worker_id={wid} acks={len(st.precommit_acks)}")
            except Exception:
                pass
            _pub("precommit", {"worker_id": wid, "acks": len(st.precommit_acks)})
        elif method == "commit":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            st.version = args.get("version")
            missing = [wid for wid in st.workers.keys() if wid not in st.precommit_acks]
            payload = {
                "event": "commit",
                "version": st.version,
                "acks": len(st.precommit_acks),
                "total_workers": len(st.workers),
                "accepted": len(missing) == 0,
                "waiting_for": missing,
                "digest": st.plan_digest,
            }
            st.state = "committed" if len(missing) == 0 else st.state
            rep.send_json(payload)
            try:
                log.info(f"commit version={st.version} accepted={payload['accepted']} waiting={missing}")
            except Exception:
                pass
            _pub("commit", payload)
        elif method == "abort":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            payload = {"event": "abort", "reason": args.get("reason")}
            st.state = "aborted"
            rep.send_json(payload)
            _pub("abort", payload)
        elif method == "submit_plan":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            st.plan = args.get("plan")
            st.plan_digest = args.get("digest")
            rep.send_json({"ok": True, "digest": st.plan_digest})
            _pub("plan", {"digest": st.plan_digest, "buckets": len((st.plan or {}).get("buckets", []))})
        elif method == "post_handle":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            version = str(args.get("version") or "_")
            bucket_id = int(args.get("bucket_id", -1))
            handle = args.get("handle")
            sig = args.get("sig")
            node = str(args.get("node") or "global")
            ts = time.time()
            st.handles.setdefault(version, {}).setdefault(bucket_id, {})[node] = {"handle": handle, "sig": sig, "ts": ts, "acks": []}
            rep.send_json({"ok": True})
            _pub("handle", {"version": version, "bucket_id": bucket_id, "node": node})
        elif method == "get_handle":
            version = str(args.get("version") or "_")
            bucket_id = int(args.get("bucket_id", -1))
            node = str(args.get("node") or "global")
            d = st.handles.get(version, {}).get(bucket_id, {})
            # prefer node-specific; fallback to global
            entry = d.get(node) or d.get("global")
            if entry:
                rep.send_json({"handle": entry.get("handle"), "sig": entry.get("sig"), "ts": entry.get("ts")})
            else:
                rep.send_json({"handle": None})
        elif method == "ack_handle":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            version = str(args.get("version") or "_")
            bucket_id = int(args.get("bucket_id", -1))
            node = str(args.get("node") or "global")
            who = str(args.get("who") or "?")
            d = st.handles.get(version, {}).get(bucket_id, {})
            entry = d.get(node) or d.get("global")
            if entry is not None:
                acks = entry.setdefault("acks", [])
                if who not in acks:
                    acks.append(who)
                rep.send_json({"ok": True, "acks": len(acks)})
            else:
                rep.send_json({"ok": False, "acks": 0})
        elif method == "set_current_manifest":
            if not _check_token():
                rep.send_json({"error": "unauthorized"}); continue
            # Persist the current (running) manifest; used for delta planning without explicit prev
            st.current_manifest = args.get("manifest")
            # Optionally set version from manifest
            try:
                st.version = (st.current_manifest or {}).get("version")
            except Exception:
                pass
            rep.send_json({"ok": True, "version": st.version})
            _pub("status", {"event": "current_manifest_updated", "version": st.version})
        elif method == "get_current_manifest":
            rep.send_json({"manifest": st.current_manifest, "version": st.version})
        elif method == "get_plan":
            rep.send_json({"plan": st.plan, "digest": st.plan_digest})
        elif method == "report_have":
            b = int(args.get("bucket_id", -1))
            r = int(args.get("rank", -1))
            st.have.setdefault(b, [])
            if r not in st.have[b]:
                st.have[b].append(r)
            rep.send_json({"ok": True})
            _pub("have", {"bucket_id": b, "ranks": st.have.get(b, [])})
        elif method == "who_has":
            b = int(args.get("bucket_id", -1))
            rep.send_json({"bucket_id": b, "ranks": st.have.get(b, [])})
        elif method == "status":
            # update gauges each status call
            try:
                g_workers.set(float(len(st.workers)))
                g_acks.set(float(len(st.precommit_acks)))
                g_have_buckets.set(float(len(st.have)))
            except Exception:
                pass
            rep.send_json({
                "workers": list(st.workers.keys()),
                "version": st.version,
                "current_manifest": bool(st.current_manifest),
                "plan_digest": st.plan_digest,
                "handles_versions": list(st.handles.keys()),
                "have": {str(k): v for k, v in st.have.items()},
                "precommit_acks": list(st.precommit_acks.keys()),
                "heartbeats": st.heartbeats,
                "rpc_total": st.rpc_total,
                "last_events": st.last_events,
                "state": st.state,
            })
        elif method == "list_workers":
            rep.send_json({"workers": st.workers})
        else:
            rep.send_json({"error": f"unknown method {method}"})

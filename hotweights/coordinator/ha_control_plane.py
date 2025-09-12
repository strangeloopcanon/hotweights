"""
High-Availability (HA) Control Plane with optional Redis backend.

- Stores version-scoped state for workers, plans, and CUDA-IPC handles.
- Leader election via a simple lock to avoid split brain.
- TTL-based cleanup of posted handles; acks shorten TTLs.
- Exposes the same RPCs as the basic ZeroMQ server so it can be swapped in.
"""
from __future__ import annotations

import json
import threading
import time
import os
from typing import Optional

try:  # optional
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional
    zmq = None  # type: ignore

from ..telemetry.prom import Counter, Gauge, start_http_server
from ..telemetry.logging import get_logger

try:  # optional
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional
    redis = None  # type: ignore


class DistributedKVStoreMock:
    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, value: bytes, ttl: float | None = None) -> bool:  # noqa: ARG002 - ttl unused
        with self._lock:
            self._store[key] = value
            return True

    def get_prefix(self, prefix: str) -> list[tuple[str, bytes]]:
        with self._lock:
            return [(k, v) for k, v in self._store.items() if k.startswith(prefix)]

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            keys_to_del = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_del:
                del self._store[k]

    def acquire_lock(self, lock_key: str, ttl: int = 10) -> bool:
        # Simplified leader election: first one to acquire wins and keeps it for 'ttl' seconds
        with self._lock:
            lock_raw = self._store.get(lock_key)
            if lock_raw is not None:
                try:
                    payload = json.loads(lock_raw.decode("utf-8"))
                    if time.time() - float(payload.get("ts", 0.0)) < ttl:
                        return False
                except Exception:
                    return False
            self._store[lock_key] = json.dumps({"ts": time.time()}).encode("utf-8")
            return True


class RedisKVStore:
    """Redis-backed KV with basic primitives and TTL support."""

    def __init__(self, url: str | None = None) -> None:
        if redis is None:
            raise RuntimeError("redis-py not installed; install with 'pip install redis'")
        url = url or os.getenv("HOTWEIGHTS_REDIS_URL", "redis://127.0.0.1:6379/0")
        self._r = redis.Redis.from_url(url)

    def get(self, key: str) -> bytes | None:
        return self._r.get(key)

    def put(self, key: str, value: bytes, ttl: float | None = None) -> bool:
        if ttl and ttl > 0:
            return bool(self._r.set(key, value, ex=max(1, int(ttl))))
        return bool(self._r.set(key, value))

    def get_prefix(self, prefix: str) -> list[tuple[str, bytes]]:
        out: list[tuple[str, bytes]] = []
        patt = prefix + "*"
        cursor = 0
        while True:
            cursor, keys = self._r.scan(cursor=cursor, match=patt, count=200)
            if keys:
                vals = self._r.mget(keys)
                for k, v in zip(keys, vals):
                    if v is not None:
                        out.append((k.decode("utf-8"), v))
            if cursor == 0:
                break
        return out

    def delete_prefix(self, prefix: str) -> None:
        patt = prefix + "*"
        cursor = 0
        todel: list[bytes] = []
        while True:
            cursor, keys = self._r.scan(cursor=cursor, match=patt, count=200)
            todel.extend(keys)
            if cursor == 0:
                break
        if todel:
            pipe = self._r.pipeline()
            for k in todel:
                pipe.delete(k)
            pipe.execute()

    def acquire_lock(self, lock_key: str, ttl: int = 10) -> bool:
        return bool(self._r.set(lock_key, str(time.time()), nx=True, ex=max(1, int(ttl))))


class HAControlPlane:
    def __init__(self, endpoint: str, pub_endpoint: str, instance_id: str):
        if zmq is None:
            raise RuntimeError("pyzmq not installed; install with 'pip install .[extras]'")
        self.instance_id = instance_id
        # Backend selection
        backend = os.getenv("HOTWEIGHTS_COORD_BACKEND", "auto").lower()
        if backend in ("redis", "auto"):
            try:
                self.kv = RedisKVStore()
                print(f"[{self.instance_id}] Using RedisKVStore for HA state")
            except Exception as e:
                if backend == "redis":
                    raise
                print(f"[{self.instance_id}] Redis unavailable ({e}); using in-memory KV")
                self.kv = DistributedKVStoreMock()
        else:
            self.kv = DistributedKVStoreMock()

        self.leader_key = "hotweights/leader"
        self.is_leader = False
        self.handle_ttl = float(os.getenv("HOTWEIGHTS_HANDLE_TTL", "30"))
        # Metrics
        self.m_handles_posted = Counter("hotweights_handles_posted_total", "Total handles posted")
        self.m_handles_fetched = Counter("hotweights_handles_fetched_total", "Total handles fetched")
        self.m_handles_acked = Counter("hotweights_handles_acked_total", "Total handles acked")
        self.m_handles_expired = Counter("hotweights_handles_expired_total", "Total handles expired")
        self.g_handles_active = Gauge("hotweights_handles_active", "Active (unacked,unexpired) handles")

        self.ctx = zmq.Context.instance()
        self.rep_sock = self.ctx.socket(zmq.REP)
        self.rep_sock.bind(endpoint)
        self.pub_sock = self.ctx.socket(zmq.PUB)
        self.pub_sock.bind(pub_endpoint)
        self._log = get_logger("HAControlPlane", {"id": instance_id})

        print(f"[{self.instance_id}] HA Control Plane instance started.")
        # Metrics endpoint (best-effort)
        try:
            port = int(os.getenv("HOTWEIGHTS_COORD_METRICS_PORT", "9100"))
            start_http_server(port)
        except Exception:
            pass

    def _publish(self, topic: str, payload: dict) -> None:
        self.pub_sock.send_multipart([topic.encode("utf-8"), json.dumps(payload).encode("utf-8")])

    def _update_active_handles_gauge(self) -> None:
        try:
            now = time.time()
            entries = self.kv.get_prefix("hotweights/handles/")
            active = 0
            for _k, v in entries:
                try:
                    data = json.loads(v.decode("utf-8"))
                    if data.get("handle") and (now - float(data.get("ts", 0.0)) < self.handle_ttl):
                        active += 1
                except Exception:
                    continue
            self.g_handles_active.set(float(active))
        except Exception:
            pass

    def _leader_election_loop(self) -> None:
        while True:
            try:
                if self.kv.acquire_lock(self.leader_key):
                    if not self.is_leader:
                        self.is_leader = True
                        print(f"[{self.instance_id}] Acquired leadership.")
                        try:
                            self._log.info("acquired leadership")
                        except Exception:
                            pass
                else:
                    if self.is_leader:
                        self.is_leader = False
                        print(f"[{self.instance_id}] Lost leadership.")
                        try:
                            self._log.warning("lost leadership")
                        except Exception:
                            pass
            except Exception:
                pass
            time.sleep(5)

    def _cleanup_handles_loop(self) -> None:
        # Periodically mark expired handle records to trigger gauge updates; Redis TTL handles actual deletion.
        while True:
            try:
                now = time.time()
                for k, v in self.kv.get_prefix("hotweights/handles/"):
                    try:
                        data = json.loads(v.decode("utf-8"))
                        ts = float(data.get("ts", 0.0))
                        if now - ts > self.handle_ttl and data.get("handle") is not None:
                            self.kv.put(k, json.dumps({"handle": None, "ts": now}).encode("utf-8"), ttl=5.0)
                            self.m_handles_expired.inc(1.0)
                    except Exception:
                        continue
                self._update_active_handles_gauge()
            except Exception:
                pass
            time.sleep(5)

    def run(self) -> None:
        leader_thread = threading.Thread(target=self._leader_election_loop, daemon=True)
        leader_thread.start()
        cleaner_thread = threading.Thread(target=self._cleanup_handles_loop, daemon=True)
        cleaner_thread.start()

        while True:
            msg = self.rep_sock.recv_json()
            method = msg.get("method")
            args = msg.get("args", {})
            response = self.handle_request(method, args)
            self.rep_sock.send_json(response)

    def handle_request(self, method: str, args: dict) -> dict:
        # Read operations can be handled by any instance.
        if method == "status":
            workers = [json.loads(v) for _k, v in self.kv.get_prefix("hotweights/workers/")]
            plan_digest = self.kv.get("hotweights/plan/digest")
            return {
                "workers": [w.get("id", "?") for w in workers],
                "plan_digest": plan_digest.decode() if plan_digest else None,
            }

        if not self.is_leader:
            return {"error": "Not the leader. Please retry.", "leader_hint": "..."}

        # Write operations below
        if method == "register":
            worker_id = args["worker_id"]
            key = f"hotweights/workers/{worker_id}"
            self.kv.put(key, json.dumps({"id": worker_id, "caps": args.get("caps"), "ts": time.time()}).encode("utf-8"))
            self._publish("register", {"worker_id": worker_id})
            return {"ok": True}

        elif method == "submit_plan":
            plan = args.get("plan")
            digest = args.get("digest")
            self.kv.put("hotweights/plan/current", json.dumps(plan).encode("utf-8"))
            self.kv.put("hotweights/plan/digest", (digest or "").encode("utf-8"))
            self._publish("plan", {"digest": digest})
            return {"ok": True, "digest": digest}

        elif method == "begin":
            version = args["version"]
            self.kv.delete_prefix(f"hotweights/versions/{version}/")
            self.kv.delete_prefix(f"hotweights/handles/{version}/")
            self.kv.put(f"hotweights/versions/{version}/active", b"1")
            self._publish("begin", {"version": version})
            return {"event": "begin", "version": version}

        elif method == "commit":
            version = args["version"]
            self.kv.delete_prefix(f"hotweights/handles/{version}/")
            self._publish("commit", {"version": version, "accepted": True})
            return {"event": "commit", "version": version, "accepted": True}

        elif method == "post_handle":
            b = int(args.get("bucket_id", -1))
            handle = args.get("handle")
            version = args.get("version") or "_"
            sig = args.get("sig")
            if b < 0 or handle is None:
                return {"ok": False, "error": "invalid args"}
            key = f"hotweights/handles/{version}/{b}"
            payload = json.dumps({"handle": handle, "sig": sig, "ts": time.time()}).encode("utf-8")
            self.kv.put(key, payload, ttl=self.handle_ttl)
            self._publish("handle", {"bucket_id": b})
            self.m_handles_posted.inc(1.0)
            self._update_active_handles_gauge()
            try:
                self._log.debug(f"post_handle version={version} bucket={b}")
            except Exception:
                pass
            return {"ok": True}

        elif method == "get_handle":
            b = int(args.get("bucket_id", -1))
            version = args.get("version") or "_"
            if b < 0:
                return {"handle": None}
            key = f"hotweights/handles/{version}/{b}"
            raw = self.kv.get(key)
            if not raw:
                return {"handle": None}
            try:
                data = json.loads(raw.decode("utf-8"))
                h = data.get("handle")
                s = data.get("sig")
                if h:
                    self.m_handles_fetched.inc(1.0)
                    try:
                        self._log.debug(f"get_handle version={version} bucket={b}")
                    except Exception:
                        pass
                return {"handle": h, "sig": s}
            except Exception:
                return {"handle": None, "sig": None}

        elif method == "ack_handle":
            b = int(args.get("bucket_id", -1))
            version = args.get("version") or "_"
            if b < 0:
                return {"ok": False}
            key = f"hotweights/handles/{version}/{b}"
            payload = json.dumps({"handle": None, "ts": time.time()}).encode("utf-8")
            # Short TTL to ensure fast cleanup
            self.kv.put(key, payload, ttl=5.0)
            self.m_handles_acked.inc(1.0)
            self._update_active_handles_gauge()
            try:
                self._log.debug(f"ack_handle version={version} bucket={b}")
            except Exception:
                pass
            return {"ok": True}

        else:
            return {"error": f"Unknown method {method}"}


def serve(endpoint: str, pub_endpoint: str, instance_id: str) -> None:
    server = HAControlPlane(endpoint, pub_endpoint, instance_id)
    server.run()

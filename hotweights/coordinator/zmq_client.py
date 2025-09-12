"""ZeroMQ client for the coordinator server.

Provides both a REQ client and a SUB subscriber for event streams.
"""
from __future__ import annotations

try:  # optional
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional
    zmq = None  # type: ignore


class Client:
    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555") -> None:
        if zmq is None:
            raise RuntimeError("pyzmq not installed; install with 'pip install .[extras]'")
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.connect(endpoint)

    def call(self, method: str, **args):  # noqa: ANN003
        self._sock.send_json({"method": method, "args": args})
        return self._sock.recv_json()


class Subscriber:
    """Simple SUB client receiving (topic, payload) JSON events."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556", topics: list[str] | None = None) -> None:
        if zmq is None:
            raise RuntimeError("pyzmq not installed; install with 'pip install .[extras]'")
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(endpoint)
        if not topics:
            self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        else:
            for t in topics:
                self._sock.setsockopt_string(zmq.SUBSCRIBE, t)

    def recv(self):  # noqa: ANN201
        topic, blob = self._sock.recv_multipart()
        import json as _json

        return topic.decode("utf-8"), _json.loads(blob.decode("utf-8"))

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

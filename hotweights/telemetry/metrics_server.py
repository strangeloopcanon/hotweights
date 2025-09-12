"""Tiny metrics HTTP server exposing counters and histograms.

This is intentionally minimal; for production use, replace with Prometheus client.
"""
from __future__ import annotations

import http.server
import threading
from dataclasses import dataclass, field
from typing import Dict


_REGISTRY: Dict[str, float] = {}


def inc(key: str, amt: float = 1.0) -> None:
    _REGISTRY[key] = _REGISTRY.get(key, 0.0) + amt


def set_gauge(key: str, val: float) -> None:
    _REGISTRY[key] = val


class Handler(http.server.BaseHTTPRequestHandler):  # type: ignore[misc]
    def do_GET(self):  # noqa: N802, ANN001
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = []
        for k, v in sorted(_REGISTRY.items()):
            body.append(f"{k} {v}")
        blob = ("\n".join(body) + "\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)


@dataclass
class MetricsServer:
    host: str = "0.0.0.0"
    port: int = 9099
    _server: http.server.HTTPServer | None = field(init=False, default=None)
    _thread: threading.Thread | None = field(init=False, default=None)

    def start(self) -> None:
        self._server = http.server.ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

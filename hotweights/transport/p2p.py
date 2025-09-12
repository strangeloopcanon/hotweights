"""Simple P2P item server/client for late-join fetch.

Protocol: a single-line request 'GET <key> <nbytes>\n' and raw bytes response.
Not authenticated; intended for trusted cluster networks.
"""
from __future__ import annotations

import socket
import socketserver
import threading
from dataclasses import dataclass


@dataclass
class P2PServer:
    host_agent: object
    host: str = "0.0.0.0"
    port: int = 0  # 0 -> ephemeral

    def start(self) -> tuple[str, int]:
        handler_agent = self.host_agent

        class Handler(socketserver.BaseRequestHandler):  # type: ignore[misc]
            def handle(self):  # noqa: ANN001
                line = b""
                while not line.endswith(b"\n"):
                    chunk = self.request.recv(1)
                    if not chunk:
                        return
                    line += chunk
                parts = line.decode("utf-8").strip().split()
                if len(parts) != 3 or parts[0] != "GET":
                    return
                _, key, nbytes_s = parts
                n = int(nbytes_s)
                mv = handler_agent.read(key)  # type: ignore[attr-defined]
                self.request.sendall(bytes(mv)[:n])

        class Threaded(socketserver.ThreadingTCPServer):  # type: ignore[misc]
            allow_reuse_address = True

        srv = Threaded((self.host, self.port), Handler)
        ip, port = srv.server_address
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.port = int(port)
        return ip, int(port)


def fetch_item(addr: str, port: int, key: str, nbytes: int) -> bytes:
    with socket.create_connection((addr, port), timeout=10) as s:
        s.sendall(f"GET {key} {nbytes}\n".encode("utf-8"))
        remaining = nbytes
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = s.recv(min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) != nbytes:
            raise IOError(f"short read: got {len(data)} < {nbytes}")
        return data


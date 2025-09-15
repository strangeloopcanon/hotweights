"""UCX-based P2P server and client with striping and backpressure.

Environment variables:
  - UCX_P2P_PORT (int): listening port on each worker

Server: listens for GET <key> <nbytes> requests and returns raw bytes.
Client: fetches items concurrently with a bounded semaphore for backpressure.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:  # optional
    import ucp  # type: ignore
except Exception:  # pragma: no cover
    ucp = None  # type: ignore


@dataclass
class UCXP2PServer:
    host_agent: object
    port: int = 0

    async def _handler(self, ep):  # noqa: ANN001
        while True:
            # Read header length
            hdr = bytearray(64)
            mv = memoryview(hdr)
            try:
                await ep.recv(mv[:8])
            except Exception:
                break
            n = int(np.frombuffer(hdr[:8], dtype=np.int64)[0])
            if n <= 0 or n > 1024:
                break
            await ep.recv(mv[:n])
            parts = hdr[8:8+n].decode("utf-8").strip().split()
            cmd = parts[0]
            if cmd == "GET" and len(parts) == 3:
                _, key, nbytes_s = parts
                nbytes = int(nbytes_s)
                mvbuf = self.host_agent.read(key)  # type: ignore[attr-defined]
                data = bytes(mvbuf)[:nbytes]
                await ep.send(np.array([len(data)], dtype=np.int64))
                await ep.send(np.frombuffer(data, dtype=np.uint8))
            elif cmd == "GETR" and len(parts) == 4:
                _, key, off_s, nbytes_s = parts
                off = int(off_s); nbytes = int(nbytes_s)
                mvbuf = self.host_agent.read(key)  # type: ignore[attr-defined]
                data = bytes(mvbuf)[off:off+nbytes]
                await ep.send(np.array([len(data)], dtype=np.int64))
                await ep.send(np.frombuffer(data, dtype=np.uint8))
            else:
                break
        try:
            await ep.close()
        except Exception:
            pass

    def start(self) -> Tuple[str, int]:
        if ucp is None:
            raise RuntimeError("ucx-py not installed")
        from ..utils.env import env_int
        from ..telemetry.logging import get_logger
        port = env_int("UCX_P2P_PORT", 20000, minimum=1)
        self.port = port
        try:
            get_logger("UCXP2PServer").info(f"listen_port={port}")
        except Exception:
            pass
        # Launch listener in background event loop
        async def _run():
            await ucp.create_listener(self._handler, port=port)
            while True:
                await asyncio.sleep(3600)

        loop = asyncio.get_event_loop_policy().new_event_loop()
        asyncio.get_child_watcher() if hasattr(asyncio, 'get_child_watcher') else None
        t = loop.create_task(_run())
        import threading
        threading.Thread(target=loop.run_until_complete, args=(t,), daemon=True).start()
        return ("0.0.0.0", port)


async def _fetch_one(addr: str, port: int, key: str, nbytes: int, sem: asyncio.Semaphore) -> bytes:
    async with sem:
        ep = await ucp.create_endpoint(addr, port)
        # Send header length + header
        cmd = f"GET {key} {nbytes}".encode("utf-8")
        hdr = np.empty(8 + len(cmd), dtype=np.uint8)
        np.frombuffer(hdr, dtype=np.int64)[:1] = np.array([len(cmd)], dtype=np.int64)
        hdr[8:] = np.frombuffer(cmd, dtype=np.uint8)
        await ep.send(hdr)
        n = np.empty(1, dtype=np.int64)
        await ep.recv(n)
        size = int(n[0])
        out = np.empty(size, dtype=np.uint8)
        await ep.recv(out)
        try:
            await ep.close()
        except Exception:
            pass
        return bytes(out)


async def _fetch_range(addr: str, port: int, key: str, off: int, nbytes: int, sem: asyncio.Semaphore) -> bytes:
    async with sem:
        ep = await ucp.create_endpoint(addr, port)
        cmd = f"GETR {key} {off} {nbytes}".encode("utf-8")
        hdr = np.empty(8 + len(cmd), dtype=np.uint8)
        np.frombuffer(hdr, dtype=np.int64)[:1] = np.array([len(cmd)], dtype=np.int64)
        hdr[8:] = np.frombuffer(cmd, dtype=np.uint8)
        await ep.send(hdr)
        n = np.empty(1, dtype=np.int64)
        await ep.recv(n)
        size = int(n[0])
        out = np.empty(size, dtype=np.uint8)
        await ep.recv(out)
        try:
            await ep.close()
        except Exception:
            pass
        return bytes(out)


def fetch_items_concurrent(tasks: List[Tuple[str, int, str, int]], concurrency: int = 4) -> List[bytes]:
    """Fetch multiple items concurrently using UCX.

    tasks: list of (addr, port, key, nbytes)
    returns list of item bytes in order corresponding to tasks
    """
    if ucp is None:
        raise RuntimeError("ucx-py not installed")

    async def _run():
        sem = asyncio.Semaphore(concurrency)
        coros = [
            _fetch_one(addr, port, key, nbytes, sem) for (addr, port, key, nbytes) in tasks
        ]
        return await asyncio.gather(*coros)

    return asyncio.run(_run())


def fetch_ranges_concurrent(tasks: List[Tuple[str, int, str, int, int]], concurrency: int = 8) -> List[bytes]:
    """Fetch multiple ranges concurrently using UCX.

    tasks: list of (addr, port, key, off, nbytes)
    returns list of bytes in order corresponding to tasks
    """
    if ucp is None:
        raise RuntimeError("ucx-py not installed")

    async def _run():
        # env-tunable concurrency
        from ..utils.env import env_int
        conc = env_int("HOTWEIGHTS_P2P_CONCURRENCY", concurrency, minimum=1)
        sem = asyncio.Semaphore(max(1, conc))
        coros = [
            _fetch_range(addr, port, key, off, n, sem) for (addr, port, key, off, n) in tasks
        ]
        return await asyncio.gather(*coros)

    return asyncio.run(_run())

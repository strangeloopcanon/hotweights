"""Pinned-host staging agent (stub).

Future: allocate pinned host buffers, provide memoryviews, and seal to SHM paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

try:  # optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore


@dataclass
class HostAgent:
    """In-memory staging agent.

    If ``use_pinned`` is True and PyTorch is available, reserves pinned-host
    buffers for faster H2D copies; otherwise uses bytearrays.
    """

    buffers: Dict[str, Union[bytearray, "torch.Tensor"]]
    use_pinned: bool = False

    def __init__(self, use_pinned: bool = False) -> None:
        self.buffers = {}
        self.use_pinned = bool(use_pinned and (torch is not None))

    def reserve(self, key: str, nbytes: int) -> memoryview:
        buf = self.buffers.get(key)
        if isinstance(buf, bytearray):
            if len(buf) < nbytes:
                buf = bytearray(nbytes)
                self.buffers[key] = buf
            return memoryview(buf)[:nbytes]
        if torch is not None and self.use_pinned:
            if buf is None or (hasattr(buf, "numel") and int(buf.numel()) < nbytes):  # type: ignore[union-attr]
                t = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)  # type: ignore[attr-defined]
                self.buffers[key] = t
                buf = t
            # Expose a memoryview over the underlying storage via NumPy
            return memoryview(buf.numpy())[:nbytes]  # type: ignore[union-attr]
        # Fallback to bytearray if no buffer or torch unavailable
        ba = bytearray(nbytes)
        self.buffers[key] = ba
        return memoryview(ba)

    def write(self, key: str, off: int, buf: memoryview) -> None:
        mv = self.reserve(key, off + len(buf))
        mv[off : off + len(buf)] = buf  # noqa: E203

    def seal(self, key: str) -> str:
        # In a full impl, return a SHM path; here we just mark presence.
        if key not in self.buffers:
            raise KeyError(key)
        return f"shm://{key}"

    def read(self, key: str) -> memoryview:
        if key not in self.buffers:
            raise KeyError(key)
        buf = self.buffers[key]
        if isinstance(buf, bytearray):
            return memoryview(buf)
        # torch tensor
        return memoryview(buf.numpy())  # type: ignore[union-attr]

    # Torch-pinned access (optional)
    def torch_tensor(self, key: str):  # noqa: ANN001
        if torch is None:
            return None
        buf = self.buffers.get(key)
        if buf is None or isinstance(buf, bytearray):
            return None
        return buf

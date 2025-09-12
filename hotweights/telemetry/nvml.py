"""NVML helpers (optional).

Tries pynvml, falls back to parsing nvidia-smi if available.
"""
from __future__ import annotations

import shutil
import subprocess


def min_free_vram_mib() -> int | None:
    # Try pynvml
    try:  # pragma: no cover - optional path
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        free = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            free.append(mem.free // (1024 * 1024))
        pynvml.nvmlShutdown()
        return min(free) if free else None
    except Exception:
        pass

    # Fallback to nvidia-smi
    if shutil.which("nvidia-smi"):
        try:  # pragma: no cover - environment dependent
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
            return min(vals) if vals else None
        except Exception:
            return None
    return None


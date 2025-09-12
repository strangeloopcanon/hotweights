"""Telemetry subpackage (lightweight).

Exposes timers and Prometheus wrappers. Optional NVML/advanced metrics are
available via their modules when present.
"""

from .metrics import Timer
from .prom import Counter, Histogram, Gauge

__all__ = [
    "Timer",
    "Counter",
    "Histogram",
    "Gauge",
]

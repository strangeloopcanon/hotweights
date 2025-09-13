"""Prometheus integration (optional).

If prometheus_client is installed, exports real metrics; otherwise falls back
to the lightweight text server in metrics_server.
"""
from __future__ import annotations

from typing import Optional

try:  # optional
    from prometheus_client import Counter as _PCounter, Gauge as _PGauge, Histogram as _PHist, start_http_server as _p_start  # type: ignore
except Exception:  # pragma: no cover
    _PCounter = None  # type: ignore
    _PGauge = None  # type: ignore
    _PHist = None  # type: ignore
    _p_start = None  # type: ignore

from .metrics_server import MetricsServer, inc as _inc_text, set_gauge as _set_text

# Cache created metrics to avoid duplicate registration errors when clients
# construct metric wrappers multiple times (e.g., in tests or re-initialization).
_COUNTERS: dict[str, object] = {}
_GAUGES: dict[str, object] = {}
_HISTS: dict[str, object] = {}


class Counter:
    def __init__(self, name: str, desc: str = "") -> None:
        self._name = name
        if _PCounter is not None:
            if name in _COUNTERS:
                self._c = _COUNTERS[name]
            else:
                self._c = _PCounter(name, desc)
                _COUNTERS[name] = self._c
        else:
            self._c = None

    def inc(self, amt: float = 1.0) -> None:
        if self._c is not None:
            self._c.inc(amt)
        else:
            _inc_text(self._name, amt)


class Gauge:
    def __init__(self, name: str, desc: str = "", labelnames: list[str] | None = None) -> None:
        self._name = name
        self._labelnames = list(labelnames) if labelnames else None
        if _PGauge is not None:
            if name in _GAUGES:
                self._g = _GAUGES[name]
            else:
                if self._labelnames:
                    self._g = _PGauge(name, desc, self._labelnames)
                else:
                    self._g = _PGauge(name, desc)
                _GAUGES[name] = self._g
        else:
            self._g = None

    def set(self, val: float) -> None:
        if self._g is not None:
            self._g.set(val)
        else:
            _set_text(self._name, val)

    # Minimal labels() support; for text server, labels are ignored
    def labels(self, **kwargs):  # noqa: ANN003
        if self._g is not None and hasattr(self._g, "labels"):
            g = self._g.labels(**kwargs)
            class _Labeled:
                def __init__(self, inner):
                    self._inner = inner
                def set(self, val: float) -> None:
                    self._inner.set(val)
            return _Labeled(g)
        # Fallback no-op label wrapper
        class _LabeledText:
            def __init__(self, outer_name: str):
                self._name = outer_name
            def set(self, val: float) -> None:
                _set_text(self._name, val)
        return _LabeledText(self._name)


def start_http_server(port: int = 9099) -> Optional[object]:
    if _p_start is not None:
        _p_start(port)
        return None
    srv = MetricsServer(port=port)
    srv.start()
    return srv


class Histogram:
    def __init__(self, name: str, desc: str = "", buckets: Optional[list[float]] = None) -> None:
        self._name = name
        if _PHist is not None:
            if name in _HISTS:
                self._h = _HISTS[name]
            else:
                if buckets is not None:
                    self._h = _PHist(name, desc, buckets=buckets)
                else:
                    self._h = _PHist(name, desc)
                _HISTS[name] = self._h
        else:
            self._h = None
            self._g = Gauge(name + "_last", desc)

    def observe(self, val: float) -> None:
        if self._h is not None:
            self._h.observe(val)
        else:
            self._g.set(val)

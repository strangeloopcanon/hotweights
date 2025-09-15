from __future__ import annotations

from hotweights.utils.selection import choose_transport


def test_choose_transport_cpu_fallback():
    tr, caps = choose_transport("cpu", None)
    # Transport should be a shim with replicate(plan) attribute and caps show cpu_fallback
    assert hasattr(tr, "replicate")
    assert caps.get("transport") == "cpu_fallback"


from __future__ import annotations

import types

from hotweights.adapters.vllm_pause import pause_requests, resume_requests


def test_pause_resume_best_effort():
    eng = types.SimpleNamespace()
    calls = []
    eng.pause_requests = lambda: calls.append("pause")  # type: ignore[attr-defined]
    eng.resume_requests = lambda: calls.append("resume")  # type: ignore[attr-defined]
    # Simulate zero in-flight
    eng.num_active_requests = 0  # type: ignore[attr-defined]

    pause_requests(eng, timeout_s=0.01)
    resume_requests(eng)
    assert calls == ["pause", "resume"]


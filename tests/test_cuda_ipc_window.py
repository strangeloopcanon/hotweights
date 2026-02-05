from __future__ import annotations

import asyncio

from hotweights.transport.cuda_ipc import CudaIPCTransport


class _Metrics:
    def set_inflight(self, buckets: int, bytes_: int) -> None:
        _ = (buckets, bytes_)

    def set_congestion_risk(self, risk: float) -> None:
        _ = risk

    def set_recommended_window(self, window: int) -> None:
        _ = window

    def set_window(self, window: int) -> None:
        _ = window

    def set_path_utilization(self, path_key: str, value: float) -> None:
        _ = (path_key, value)


class _Log:
    def debug(self, msg: str) -> None:
        _ = msg


def test_run_replication_dynamic_window_does_not_raise_unboundlocal() -> None:
    class Dummy:
        pass

    d = Dummy()
    d._sched = None
    d.world_size = 2
    d._max_inflight_cap = 4
    d._max_inflight = 1
    d._max_inflight_bytes = 0
    d._adapt = False
    d.metrics = _Metrics()
    d._log = _Log()

    async def _replicate_one_bucket(bucket, plan):  # noqa: ANN001, ANN202
        _ = (bucket, plan)

    d._replicate_one_bucket = _replicate_one_bucket
    run = CudaIPCTransport._run_replication.__get__(d, CudaIPCTransport)
    asyncio.run(
        run(
            {
                "buckets": [
                    {"bucket_id": 0, "size": 1, "items": []},
                    {"bucket_id": 1, "size": 1, "items": []},
                ]
            }
        )
    )

"""GPU Broadcast Smoke Test (ROCm/Intel/NVIDIA without IPC).

Usage:
  python examples/gpu_broadcast_smoke.py --plan transfer.plan.json --device cuda
  python -m torch.distributed.run --nproc-per-node=2 examples/gpu_broadcast_smoke.py --plan transfer.plan.json --device cuda

Pre-reqs:
  - Plan JSON produced by `hotweights plan ...`.
  - PyTorch with ROCm (AMD), with Intel XPU + oneCCL, or CUDA.

This example exercises the GpuBroadcastTransport path and prints a short summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from hotweights.transport.gpu_broadcast import GpuBroadcastTransport
    from hotweights.staging.gpu_agent import GenericGpuAgent

    plan = json.loads(Path(args.plan).read_text())
    agent = GenericGpuAgent(device=args.device)
    if args.device.startswith("xpu"):
        backend = "ccl"
    else:
        backend = "nccl"
    tr = GpuBroadcastTransport(agent=agent, backend=backend)
    t0 = time.perf_counter()
    tr.replicate(plan)
    dt = time.perf_counter() - t0
    total_bytes = int(plan.get("total_bytes", 0))
    print(json.dumps({
        "mode": "gpu_broadcast",
        "device": args.device,
        "seconds": dt,
        "buckets": len(plan.get("buckets", [])),
        "bytes": total_bytes,
    }, indent=2))


if __name__ == "__main__":
    main()


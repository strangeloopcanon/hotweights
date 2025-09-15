Validation Checklist

Goals: sanity check selection, CPU fallback, GPU broadcast path, and metrics.

Prereqs
- Python 3.10+; install editable: `pip install -e .[dev]`
- A plan JSON file (create via `hotweights plan ...`).

CPU-only
1) Selection CPU fallback:
   - `python -c "from hotweights.utils.selection import choose_transport; print(choose_transport('cpu', None)[1])"`
   - Expect `{"transport": "cpu_fallback", ...}`.
2) Replicate (local):
   - `hotweights replicate --plan plan.json --device cpu --verify`
   - Confirm output and metrics at :9097.

NVIDIA (CUDA-IPC)
1) Replicate with IPC (single process):
   - `hotweights replicate --plan plan.json --device cuda`
   - Logs should show: "Using SOTA CUDA-IPC Transport Layer." and IPC copy/window params.
2) Metrics: verify `hotweights_ipc_*` histograms present.

AMD ROCm (RCCL) / Intel XPU (oneCCL)
1) GPU broadcast smoke:
   - ROCm: `python examples/gpu_broadcast_smoke.py --plan plan.json --device cuda`
   - Intel: `python examples/gpu_broadcast_smoke.py --plan plan.json --device xpu`
   - Expect JSON summary; logs show broadcast chunk and/or autotune.
2) Multi-process (single node):
   - `torchrun --nproc-per-node=2 examples/gpu_broadcast_smoke.py --plan plan.json --device <cuda|xpu>`
   - Confirms subgroup broadcast and per-rank scatter.
3) Metrics: verify `hotweights_bcast_*` metrics present.

Distributed (torchrun)
- Single node: `torchrun --nproc-per-node=8 -m hotweights.cli replicate --plan plan.json --device cuda`
- Multi node: see `docs/DEPLOYMENT.md` torchrun recipes.

CI (GitHub Actions)
- Runs `ruff`, `black --check`, and `pytest -m "not slow and not gpu and not mpi" -q` on Ubuntu for Python 3.10/3.11.


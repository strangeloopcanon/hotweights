Vendor Setup Guide (NVIDIA, AMD, Intel)

Overview

Hotweights selects the fastest available path per vendor automatically. This guide lists practical setup steps per platform and simple verification hints.

NVIDIA (CUDA)

- Install PyTorch with CUDA matching your driver version (see pytorch.org for the exact command):
  - Example (CUDA 12.1): `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- Optional: CuPy + KvikIO for GPUDirect Storage (GDS): `pip install cupy-cuda12x kvikio`
- Verify:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"`
  - For multi-process: use `torchrun` (sets WORLD_SIZE/RANK/LOCAL_RANK), or export envs manually.
- Run:
  - `hotweights replicate --plan plan.json --device cuda`
  - Logs should say: "Using SOTA CUDA-IPC Transport Layer."

AMD ROCm

- Install PyTorch ROCm build (see pytorch.org for ROCm wheels):
  - Example (ROCm 6.x): `pip install torch --index-url https://download.pytorch.org/whl/rocm6.0`
- Notes:
  - torch.distributed backend="nccl" uses RCCL on ROCm; no extra install needed.
  - CUDA IPC is not available; Hotweights will auto-select GPU broadcast path.
- Verify:
  - `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"`
  - Multi-process via `torchrun` with `--nproc-per-node` and standard rendezvous envs.
- Run:
  - `hotweights replicate --plan plan.json --device cuda`
  - Logs should say: "Using GPU broadcast transport (no IPC)."

Intel XPU

- Install PyTorch with Intel extensions and oneCCL bindings:
  - `pip install intel-extension-for-pytorch oneccl_bindings_for_pytorch`
- Configure oneCCL runtime environment:
  - Ensure `source /opt/intel/oneapi/setvars.sh` (path varies by install).
  - Alternatively, ensure `LD_LIBRARY_PATH` and `PATH` include oneCCL libraries/binaries.
- Verify:
  - `python -c "import torch; print(hasattr(torch, 'xpu') and torch.xpu.is_available())"`
  - `python -c "import torch, torch.distributed as dist; print('ccl' in dist.Backend.__members__)"`
- Run:
  - `hotweights replicate --plan plan.json --device xpu`
  - Logs should say: "Using GPU broadcast transport (Intel oneCCL)."

Common Issues

- Process group init hangs:
  - Verify `MASTER_ADDR/MASTER_PORT` reachability and that ranks see the same env configuration.
  - Use `NCCL_DEBUG=INFO` (NVIDIA/AMD) or enable oneCCL logging on Intel for diagnostics.
- Wrong NIC or fabric selection:
  - Set `NCCL_IB_HCA` (e.g., `mlx5_0`) for NCCL/RCCL.
  - For UCX fallback, set `HOTWEIGHTS_UCX_NET_DEVICES` or rely on `NCCL_IB_HCA` mirroring.
- Resource pressure:
  - Adjust bucket size (`--bucket-mb`), inflight window (`HOTWEIGHTS_IPC_INFLIGHT_BUCKETS`), and broadcast chunk size (`HOTWEIGHTS_BCAST_CHUNK_MB`).


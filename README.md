# hotweights

Fast, vendor‑aware weight updates for LLM serving and training.

This repository provides a control/plan/transport stack to hot‑swap large model
weights across a GPU cluster with minimal pause time. See `docs/ARCHITECTURE.md`
for a high‑level overview.

## Core Use Cases & Limitations

`hotweights` is designed for performance‑critical scenarios across NVIDIA, AMD, and Intel GPUs with automatic fast‑path selection. Here's where it excels and what its current limitations are.

### GPU Support Matrix

- NVIDIA: CUDA‑IPC zero‑copy within a node; optional GDS; hierarchical broadcast.
- AMD ROCm: GPU device broadcast via RCCL (backend="nccl") + device-side scatter.
- Intel XPU: GPU device broadcast via oneCCL (backend="ccl") + device-side scatter.
- CPU‑only: MPI/UCX/local fallback with pinned-host buffers.

> When you first land on CUDA hosts with CuPy + KvikIO installed, GDS (CuFile) staging is automatically enabled. Do a quick `pytest -k cuda_ipc` run or check the `hotweights_ipc_gds_enabled` metric after one replication to confirm the GPUDirect path is active; if the hardware is not ready yet, leave the default in place—no extra switches are required.

See `docs/VENDOR_SETUP.md` for ROCm and Intel oneCCL setup details.

### What It's Great For

*   **Rapidly Updating Inference Models**: Ideal for scenarios where you need to push new model versions (e.g., updated fine-tunes) to a live vLLM cluster with minimal downtime. It achieves this through a combination of delta-based transfers (Bodo-accelerated), high-speed transports (auto-selected MPI/UCX), and asynchronous device-to-host copies that happen in the background. The final commit is a near-instantaneous pointer swap.

*   **Synchronizing Weights in Distributed Training**: The `adapters.trainer_swap` module is designed to facilitate in-place weight swaps during training. This is valuable for advanced training schemes like reinforcement learning, federated learning, or any algorithm that requires periodically resetting or synchronizing model parameters across a cluster without restarting the training job.

### Current Limitations & Future Work

*   **KV-Cache Migration (Inference)**: Conservative, opt-in transforms are available (dtype/RoPE adjustments and head remapping for compatible GQA layouts), but full, production-grade KV migration for all architectures is ongoing work. For strict zero-downtime swaps on long sequences, plan to drain requests or validate transforms for your model family.

*   **Optimizer State (Training)**: Policy controls exist for Adam-like optimizers (preserve/reset/attenuate moments with an attenuation factor). Deeper transforms and end-to-end guides for FSDP/ZeRO training stacks will continue to evolve.

*   **Multi-node Perf Validation**: CUDA-IPC is implemented with adaptive windowing and optional GDS. We’ve added observability and fallbacks, but broader, published multi-node performance validation (8–32 GPUs) and SLO assertions are planned.

## Install (editable)

```
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -e .[dev]
```

## Quick start

```
# Create a simple manifest for a directory of files
hotweights publish --checkpoint ./example_ckpt --version 2025-09-11

# Show status (stub)
hotweights status

# Diff manifests and create a transfer plan (prev optional)
# If a coordinator is running, prev is fetched automatically as the current manifest
hotweights plan --next ./m_next.json --bucket-mb 64 --coord-endpoint tcp://127.0.0.1:5555 --output plan.json

# Built-in verification runs by default; to force failure on problems:
hotweights plan --next ./m_next.json --strict --output plan.json

# Optional: run verification as a separate step
hotweights verify-plan --plan plan.json --require-consumers || true
# If you use TP groups:
# hotweights verify-tp --tp-groups groups.json --world-size 2 || true

# Replicate and verify; fastest path auto‑selected per device
# NVIDIA (CUDA‑IPC):
hotweights replicate --plan plan.json --device cuda --verify
# AMD ROCm (broadcast + device scatter):
hotweights replicate --plan plan.json --device cuda --verify
# Intel XPU (oneCCL broadcast + device scatter):
hotweights replicate --plan plan.json --device xpu --verify
# Optional: commit into offline adapter
hotweights replicate --plan plan.json --verify --commit

# GPU broadcast smoke test (ROCm/Intel/NVIDIA w/o IPC)
python examples/gpu_broadcast_smoke.py --plan plan.json --device cuda

# Optional: run a coordinator and workers (ZeroMQ)
hotweights coord-serve --endpoint tcp://127.0.0.1:5555 &
HOTWEIGHTS_COORD=tcp://127.0.0.1:5555 mpirun -n 2 hotweights worker --pinned --no-verify

# CPU transport selection is automatic (MPI > UCX > local). Example without GPUs:
export WORLD_SIZE=2 MASTER_ADDR=127.0.0.1 MASTER_PORT=19999
# Terminal 1 (rank 0):
RANK=0 hotweights replicate --plan plan.json --verify --coord-endpoint tcp://127.0.0.1:5555 --manifest-next ./m_next.json
# Terminal 2 (rank 1):
RANK=1 hotweights replicate --plan plan.json --verify --coord-endpoint tcp://127.0.0.1:5555

# After commit, the next manifest is recorded as current (for prev-less planning next time).

### AMD ROCm and Intel XPU (GPU Broadcast)

- Broadcast device buffers using torch.distributed (RCCL/oneCCL) and scatter on device.
- Auto‑selected when CUDA‑IPC is unavailable or when `--device xpu` is used.
- Tunables for device scatter:
  - `HOTWEIGHTS_GPU_COPY_CHUNK_MB` (default 16) — device copy chunk size.
  - `HOTWEIGHTS_GPU_COPY_STREAMS` (default 2 on CUDA) — parallel streams for scatter.
  - `HOTWEIGHTS_GPU_COPY_MICRO=1` — run a quick local microbench to auto‑pick chunk/streams (see docs/TUNING.md).
- Tunables for broadcast:
  - `HOTWEIGHTS_BCAST_CHUNK_MB` (default 32) — broadcast chunk size.
  - `HOTWEIGHTS_BCAST_AUTOTUNE=1` — run a quick world broadcast microbench to auto‑pick chunk size (see docs/TUNING.md).
- Requirements:
  - AMD: PyTorch ROCm build; RCCL provided via backend="nccl".
  - Intel: PyTorch with IPEX + `oneccl_bindings_for_pytorch`; backend="ccl".
  - Ensure `torchrun` or env (`WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT`).
```

## Tests

```
pytest -q
```

## Notes
- Bodo and mpi4py are optional; code paths degrade gracefully when absent.
- Never commit model weights/checkpoints to the repository.
- For GPU staging, use `--pinned` so HostAgent reserves pinned-host buffers, then adapters can use async H2D copies.
- Prometheus metrics: workers export at :9099; coordinator exports at :9100 when running. A starter Grafana dashboard is in `dashboards/hotweights.json`.

### CUDA-IPC + GDS
- CUDA-IPC path now supports optional GPUDirect Storage (GDS) via KvikIO/CuFile. Enable with `HOTWEIGHTS_USE_GDS=1` and install `cupy` + `kvikio` on supported systems. Falls back to CPU assemble + H2D copy if unavailable.
- Concurrency/backpressure can be tuned via:
  - `HOTWEIGHTS_IPC_INFLIGHT_BUCKETS` (default 4)
  - `HOTWEIGHTS_IPC_INFLIGHT_BYTES` (bytes cap across inflight buckets, 0=off)
  - Adaptive windowing: `HOTWEIGHTS_IPC_ADAPT=1`, `HOTWEIGHTS_IPC_MIN_INFLIGHT`, `HOTWEIGHTS_IPC_MAX_INFLIGHT`, `HOTWEIGHTS_IPC_TARGET_BUCKET_MS`.
  - Metrics gauges: inflight buckets/bytes, window, and target bucket ms are exported.
- NVLink-aware copy tuning in the CUDA path:
  - `HOTWEIGHTS_IPC_COPY_CHUNK_MB` (default 16)
  - `HOTWEIGHTS_IPC_COPY_STREAMS` (default 2)
  - `HOTWEIGHTS_IPC_COPY_AUTOTUNE=1` uses a simple peer-access heuristic to default to 64MiB/4 streams on multi-GPU nodes.

### HA Control Plane (Redis)
- Coordinator can use Redis for durable state. Set `HOTWEIGHTS_COORD_BACKEND=redis` and `HOTWEIGHTS_REDIS_URL=redis://host:6379/0`.
- Handle TTL is configurable via `HOTWEIGHTS_HANDLE_TTL` (seconds). Acks shorten TTL for quicker cleanup.

### Optimizer/KV migration
- Optimizer moments (Adam-like) are preserved when shapes match; attenuation can be controlled via `HOTWEIGHTS_OPT_ATTENUATION` (default 1.0).
  - Policy control: `HOTWEIGHTS_OPT_POLICY=preserve|reset|attenuate` (default preserve). Attenuate multiplies moments by `HOTWEIGHTS_OPT_ATTENUATION`.
- KV-cache migration preserves dtype/shape and can adjust dtype/RoPE when `HOTWEIGHTS_KV_ALLOW_TRANSFORMS=1`. Optional head remapping via `HOTWEIGHTS_KV_HEAD_MAP='[ ... ]'` (JSON list) when layout permits.
  - Alternatively, use `HOTWEIGHTS_KV_HEAD_MAP_FILE=/path/to/map.json` to provide the head remapping as a JSON file.
  - A simple `derive_head_map(H, KvH)` helper exists for common GQA layouts; by default the identity map is used unless an explicit map is provided.
  - `HOTWEIGHTS_KV_HEAD_ORDER=grouped|interleaved` influences derived maps when both `H` and `KvH` are known (default grouped).

### CLI Tools
- kv-check: derive/validate KV head mapping
  - `hotweights kv-check --heads 32 --kv-heads 8 --order grouped`
- opt-check: preview optimizer policy effects
  - `hotweights opt-check --updated 100 --unchanged 100 --policy attenuate --attenuation 0.5`

### Tuning Guide
- See `docs/TUNING.md` for end-to-end tuning knobs and recommended defaults for different environments.

### Repo Structure (key directories)
- `hotweights/`: library sources (planning, transports, control plane, adapters, CLI)
- `docs/`: deployment, tuning, presets, walkthroughs, and architecture overview
- `dashboards/`: Prometheus/Grafana dashboards
- `scripts/`: helper scripts (presets and demos)
- `bench/`: small benchmark helpers
- `tests/`: unit tests (pytest)

## Metrics & Monitoring

Hotweights exposes Prometheus metrics for core components. If `prometheus_client` is not installed, a text metrics server is used.

- CUDA-IPC transport metrics:
  - `hotweights_ipc_handle_creation_seconds` (histogram)
  - `hotweights_ipc_gbytes_replicated_total` (gauge)
  - `hotweights_ipc_bandwidth_gbps` (gauge)
  - `hotweights_ipc_bucket_seconds` (histogram)

- GPU Broadcast transport metrics:
  - `hotweights_bcast_buckets_total` (counter)
  - `hotweights_bcast_bytes_total` (counter)
  - `hotweights_bcast_seconds` (histogram)
  - `hotweights_bcast_scatter_seconds` (histogram)

- HA control plane metrics:
  - `hotweights_handles_posted_total`, `hotweights_handles_fetched_total`, `hotweights_handles_acked_total`, `hotweights_handles_expired_total`
  - `hotweights_handles_active` (gauge)

Workers start a Prometheus endpoint on :9099; CLI replicate attempts to start on :9097. You can scrape these from Prometheus directly or via the provided text server endpoints.

## Security (Coordinator)

- Mutating coordinator RPCs (submit_plan, begin, precommit, commit, abort, set_current_manifest) can be guarded by a shared token.
- Start the coordinator with a token (via your own wrapper) and set `HOTWEIGHTS_COORD_TOKEN` in clients; the CLI forwards it automatically.
- Keep the coordinator on a trusted network segment or behind an auth proxy for multi-tenant environments.

## Developer Guide

- Code Structure
  - Core planning: `hotweights/planner_bodo.py` (Bodo JIT when available; pandas fallback), `hotweights/core/replicate.py` (plan helpers, verification, assemble/scatter).
  - Schemas & errors: `hotweights/core/schemas.py`, `hotweights/core/errors.py`.
- Transports: `hotweights/transport/*` with `transport/base.py` protocol and `TransportManager` auto-selection.
  - Optional NCCL transport for inter-node GPU broadcast (auto-selected when `torch.distributed` with CUDA is available and `WORLD_SIZE>1`).
  - Staging: `hotweights/staging/*` with `staging/base.py` protocol; `HostAgent` for CPU, `CudaIPCAgent` for GPU.
  - Coordinator: `hotweights/coordinator/*` (ZeroMQ REP/PUB server + client, optional HA control plane).
  - Adapters: `hotweights/adapters/*` (vLLM, trainers) with `adapters/base.py` protocol.
  - CLI: `hotweights/cli.py` thin front-end that calls core functions.

- Public APIs
  - Planning: `create_plan(prev, next, bucket_mb, consumer_map)` and `create_plan_from_current(next, provider, ...)` return a `Plan` with `plan_version`.
  - Verification: `verify_plan(plan, ...)` returns a structured report.
  - Transports implement `Transport` protocol; staging agents implement `StagingAgent`.

- Types & Contracts
  - Use the TypedDicts in `core/schemas.py` for Plan/Manifest items; keep plan JSON schema stable and versioned.
  - Prefer raising `HotweightsError` subclasses for transport/coordinator/validation failures.

- Style & Tooling
  - Python 3.10+, Black+isort+Ruff; mypy-friendly types across public modules.
  - Tests: pytest; add unit tests near changed logic; keep CLI thin and delegate logic to core functions for testability.

- Performance Notes
  - Call `planner_bodo.warmup()` in long-lived processes to amortize Bodo JIT; disabled by `HOTWEIGHTS_NO_WARMUP=1`.
  - Prefer pinned host buffers (`HostAgent(use_pinned=True)`) for efficient H2D copies.

## Planning Acceleration (Bodo)

Hotweights uses Bodo to accelerate planning on CPU; Bodo does not allocate GPU memory or move tensors.

- What Bodo speeds up
  - Delta detection: fast join/filter of next vs prev manifests on `(tensor, shard_rank)` and hash, to compute the minimal set of shards to transfer.
  - Bucket packing: JIT‑compiled first‑fit decreasing to group changed shards into size‑bounded buckets efficiently.
- What Bodo does not do
  - No GPU allocation, broadcast, or device copies — those are handled by CUDA‑IPC/NCCL/UCX transports.
- Behavior
  - Automatically uses JIT‑compiled paths when Bodo is installed; falls back to pandas if not, or when `HOTWEIGHTS_FORCE_PANDAS=1`.
  - Optional warm‑up (`planner_bodo.warmup()`) compiles once to avoid first‑use latency in long‑lived processes.
- Why it matters
  - Keeps plan computation sub‑second even for large manifests (100k–1M shards), so the control plane never dominates end‑to‑end time.
  - Enables true delta mode: moving 1–10% of bytes for fine‑tunes can beat end‑to‑end timings even when raw wire speed is similar.

## Validation & Ops

- Checklist
  - Plan: verify delta% and bucket size vs free VRAM (`hotweights plan --auto --alpha … --strict`).
  - Replicate: confirm multiple buckets in flight; stable bucket seconds; recommended window reasonable.
  - Hierarchical: set `HOTWEIGHTS_HIERARCHICAL=1`; ensure `torchrun` provides `WORLD_SIZE/RANK/LOCAL_RANK`.
  - Security: set `HOTWEIGHTS_COORD_TOKEN` in clients when using a shared coordinator.
  - See `docs/VALIDATION.md` for a step-by-step validation guide (CPU, CUDA‑IPC, ROCm/oneCCL broadcast, torchrun).

- Key knobs
  - CUDA‑IPC: `HOTWEIGHTS_IPC_INFLIGHT_BUCKETS`, `HOTWEIGHTS_IPC_ADAPT=1`, `HOTWEIGHTS_IPC_COPY_CHUNK_MB`, `HOTWEIGHTS_IPC_COPY_STREAMS`.
  - Hierarchical: `HOTWEIGHTS_HIERARCHICAL=1`, `HOTWEIGHTS_NODE_ID=<stable_node_name>` (defaults to hostname).
  - UCX: `HOTWEIGHTS_UCX_NET_DEVICES` (or set `NCCL_IB_HCA`, mirrored to UCX), `HOTWEIGHTS_UCX_CONCURRENCY` (autotunes by world size).
  - Planner: `HOTWEIGHTS_NO_WARMUP=1` (disable JIT warm-up), `HOTWEIGHTS_FORCE_PANDAS=1` (force pandas fallback).

- Gotchas & remedies
  - NCCL hangs: verify `MASTER_ADDR/PORT` reachability and consistent CUDA drivers; use `NCCL_DEBUG=INFO`.
  - Wrong NIC: set `NCCL_IB_HCA` (e.g., `mlx5_0`) or `HOTWEIGHTS_UCX_NET_DEVICES`.
  - Insufficient VRAM: reduce bucket size (`--bucket-mb` or `--auto --alpha`), or reduce inflight via `HOTWEIGHTS_IPC_INFLIGHT_BUCKETS`.
  - Mismatched ranks: prefer `torchrun` for correct `WORLD_SIZE/RANK/LOCAL_RANK` export.

### Synthetic Bench (Leaders Only)

Run a quick stage timing benchmark for hierarchical broadcast (leaders only):

```
HOTWEIGHTS_HIERARCHICAL=1 torchrun --nnodes=2 --nproc-per-node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  hotweights bench-hierarchical --size-mb 256 --repeat 3
```

This prints per-rank JSON with H2D (pinned→GPU), inter-node NCCL broadcast among leaders, and device copy proxies for intra-node and reload.

## Cluster Install (Quick Paths)

Easiest options to get multi‑GPU/multi‑node runs:

- Conda/dev install (bare metal)
  - Create the same Conda env on each node (or conda‑pack it):
    - `conda create -n hw python=3.10 -y && conda activate hw`
    - Install PyTorch with CUDA matching your driver: see pytorch.org for the right pip/conda command.
    - `pip install -e .[extras]`
  - Verify: `python -c "import torch; print(torch.cuda.is_available())"`
  - Launch with `torchrun` (sets WORLD_SIZE/RANK/LOCAL_RANK).

- Containers (recommended for consistency)
  - Base image: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
  - Install PyTorch (CUDA build) and `hotweights[extras]`.
  - Run with `--gpus all` and NVIDIA Container Toolkit; orchestrate with your scheduler (K8s/Slurm).
  - Minimal Dockerfile:
    ```Dockerfile
    FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
    RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
    RUN pip3 install --upgrade pip
    # Install PyTorch CUDA build (choose matching version from pytorch.org)
    RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121
    # Copy and install hotweights (or pip install from Git)
    COPY . /opt/hotweights
    RUN pip3 install -e /opt/hotweights[extras]
    ENTRYPOINT ["python3", "-m", "hotweights.cli"]
    ```

- AWS Quickstart
  - Instance types: p4d/p5 (NVIDIA A100/H100, high bandwidth). Use AWS Deep Learning AMI (DLAMI) or Deep Learning Containers (DLC) with PyTorch.
  - DLAMI path (Ubuntu):
    - Launch p4d/p5 with DLAMI (PyTorch). Drivers/CUDA/NCCL are preinstalled.
    - `conda activate pytorch` (or your env) and `pip install -e .[extras]`.
  - DLC path (containers): use AWS DLC PyTorch containers with EFA support; mount your code or bake a custom image as above.
  - Multi‑node: use `torchrun --nnodes` with a reachable `MASTER_ADDR:MASTER_PORT`. For EFA networks, ensure the EFA drivers/Libfabric are present (DLAMI/DLC EFA variants handle this) and set `FI_PROVIDER=efa` if needed.

Minimal launch (2 nodes × 8 GPUs):

```
export MASTER_ADDR=<rank0-host>
export MASTER_PORT=29500
hotweights coord-serve --endpoint tcp://<coord-host>:5555 &
# Build plan once and share plan.json to all nodes
hotweights plan --next m_next.json --coord-endpoint tcp://<coord-host>:5555 --strict --output plan.json
HOTWEIGHTS_HIERARCHICAL=1 torchrun --nnodes=2 --nproc-per-node=8 \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m hotweights.cli replicate --plan plan.json --coord-endpoint tcp://<coord-host>:5555
```



## Transport Tuning (UCX/MPI)

UCX environment variables:

- `HOTWEIGHTS_UCX_CHUNK_MB`: chunk size per broadcast block (default 8 MiB)
- `HOTWEIGHTS_UCX_CONCURRENCY`: max number of peers to send to per chunk (0=fanout)

MPI options:

- `--window`: MPI in-flight buckets window
- `--mpi-chunk-mb`: chunk size for MPI broadcast (0=off)

## Tensor-Parallel Group Mapping

Provide per-group rank maps via `HOTWEIGHTS_TP_GROUPS` (JSON string or path). See `docs/TP_GROUPS.md` for schema and examples.

## vLLM / Torch integration (adapter)
- The adapter can ingest staged items and either:
  - `bind_module(module, name_map)` then `commit(version)` to flip/copy params, or
  - `apply_from_host_to_module(items, host, module, name_map, device="cuda")` for in-place async H2D copies.
- `name_map` maps staged keys like `"layers.0.w_q:0"` to dotted module param paths (e.g., `"model.layers.0.q_proj.weight"`).
- For vLLM, wire these calls inside a WorkerExtension lifecycle to keep pause within budget.

### vLLM plugin helper
Call from a vLLM worker after model load:

```
from hotweights.adapters.vllm_plugin import update_weights_from_coordinator

def name_map_fn(plan):
    # Example: map manifest names to module param paths
    mapping = {}
    for b in plan["buckets"]:
        for it in b["items"]:
            mapping[it["key"]] = it["tensor"].replace("/", ".")
    return mapping

update_weights_from_coordinator(model_module, name_map_fn, endpoint="tcp://127.0.0.1:5555", use_mpi=True, pinned=True, verify=False, device="cuda")
```

Or bind a background updater in a vLLM worker after model construction:

```
from hotweights.adapters.vllm_bind import bind_to_vllm

binding = bind_to_vllm(vllm_worker_or_engine, name_map_fn, endpoint="tcp://127.0.0.1:5555", use_mpi=True, pinned=True, device="cuda")
# binding.stop() to stop the background thread
```

### Tightening With vLLM Hooks
- Preferred integration is a WorkerExtension or explicit engine hooks where available:
  - begin_update(manifest/version): allocate shadow/storage
  - request_buffer/finalize_shard: stream staged bytes
  - precommit(): barrier with running requests drained
  - commit(version): atomic pointer flip or in-place copy
- Our binder uses best-effort pause/resume (tries common vLLM engine methods). For strict SLOs, wire these calls into your worker lifecycle and use `apply_from_host_to_module(...)` inside the pause/commit window.
- Auto-bind (optional): `from hotweights.adapters.vllm_auto import install_autobind; install_autobind(name_map_fn, endpoint=...)` patches vLLM engine constructors to start background updates.

### WorkerExtension Implementation
- A `HotweightsWorkerExtension` is provided in `hotweights.adapters.vllm_extension` with `begin_update`, `request_buffer`, `finalize_shard`, `precommit`, `commit` stubs and a convenience `bind_module(module)` + `apply_update()` to trigger the end‑to‑end plan‑replicate‑apply flow.
- This lets you plug into versions of vLLM that support custom worker extensions while keeping the heavy lifting in Hotweights.

### Name Mapping
- Provide a `name_map_fn(plan)->dict` mapping staged keys (e.g., `layers.0.attn.q_proj.weight:0`) to module parameter paths.
- If omitted, the library infers mappings by longest suffix match on parameter names and boosts matches by dtype/shape.

### Coordinator Hooks
- Coordinator RPCs: `submit_plan`, `begin`, `precommit`, `commit`, `status`. `commit` returns ack status; binders should gate flips on precommit quorum.

### Metrics & Dashboards
- Workers export `hotweights_buckets_total` and `hotweights_last_bucket_size_bytes`.
- Coordinator exports `hotweights_coord_workers`, `hotweights_coord_precommit_acks`, and `hotweights_coord_have_buckets`.
- Import `dashboards/hotweights.json` into Grafana for a basic view; extend panels per your needs.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

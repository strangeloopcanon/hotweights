# Architecture Overview

Hotweights delivers fast, versioned weight updates for LLM serving and training. It is organized into three planes:

- Control plane: versioned update lifecycle, worker orchestration, late-join; implemented with ZeroMQ (simple) and a Redis‑backed HA control plane.
- Data plane: delta compute + bucket planning (Bodo‑accelerated) and transports (CUDA‑IPC intra‑node; GPU broadcast via NCCL/RCCL/oneCCL when IPC is unavailable; UCX/MPI inter‑ or intra‑node fallbacks).
- Server plane: adapters for inference/training (vLLM, PyTorch) to ingest staged data and commit with minimal pause.

## Flow

1. Publisher posts a new checkpoint and builds the next manifest. The current (prev) manifest is tracked by the coordinator.
2. Planner computes delta (prev vs next) and packs items into buckets (size‑aware). Plan includes `plan_version` and a verification report.
3. Transport replicates: root assembles buckets and broadcasts/IPC shares to consumers.
4. Workers/adapters load staged shards (CPU→GPU or IPC→device) into shadow storage and commit.
5. Late joiners fetch missing buckets and join at the next commit.

## Components and Paths

- Manifests and Planning
  - `hotweights/manifest.py`: schema + hashing + DataFrame conversion.
  - `hotweights/planner_bodo.py`: delta and bucketization (Bodo when available; pandas fallback).
  - `hotweights/core/replicate.py`: plan helpers, assemble/scatter, consumer set derivation (pattern + TP groups + AUTO).

- Transports
  - CUDA‑IPC (`hotweights/transport/cuda_ipc.py` + `hotweights/staging/cuda_ipc_agent.py`):
    - Adaptive windowing, topology‑informed recommendations, optional GPUDirect Storage (KvikIO/CuPy), multi‑stream device scatter, handle signing + HA‑backed handle lifecycle.
  - UCX (`hotweights/transport/ucx_stream.py`): chunking, concurrency, retries/backoff + metrics, optional autotune; P2P utilities in `ucx_p2p.py`.
  - MPI (`hotweights/transport/mpi_stream.py`): streaming overlap, chunking, subgroup communicators; metrics.
  - GPU Broadcast (`hotweights/transport/gpu_broadcast.py` + `hotweights/staging/gpu_agent.py`): vendor‑neutral device broadcast using torch.distributed backends (NCCL/RCCL/oneCCL), device‑side multi‑stream scatter with overlap and microbench autotune.

- Control Plane
  - ZeroMQ server/client (`hotweights/coordinator/zmq_server.py`, `zmq_client.py`): submit plan, begin/precommit/commit, heartbeats.
  - HA control plane (`hotweights/coordinator/ha_control_plane.py`): Redis‑backed KV store, version‑scoped handle post/get/ack with TTL cleanup; metrics; leader election.

- Adapters (Server Plane)
  - vLLM / Torch integration (`hotweights/adapters/vllm_*.py`): CPU→GPU apply or IPC→device copy, name mapping helpers, binder hooks.
  - Training (`hotweights/adapters/trainer_swap.py`): in‑place param copy at safe barriers.
  - KV cache migration (`hotweights/adapters/kv_cache_migration.py`): conservative dtype/RoPE/head transforms (guarded).
  - Optimizer sync (`hotweights/adapters/optimizer_sync.py`): policy (preserve/reset/attenuate) for Adam‑like optimizers.

- CLI and Docs
  - CLI entrypoint (`hotweights/cli.py`): publish/plan/replicate/worker/coord; metrics; kv/opt tools; plan/TP validators; bench.
  - Walkthroughs, presets, tuning, and dashboards under `docs/` and `dashboards/`.

## Key Concepts

- Buckets: size‑bounded groups of changed items streamed as contiguous buffers.
- Consumer sets: per‑bucket `consumer_ranks` derived from patterns or TP groups to scope replication.
- Handle lifecycle: CUDA‑IPC handles posted/fetched/acked via HA control plane; TTL cleanup ensures no leaks.
- Observability: Prometheus metrics for timing, windows/inflight, congestion, per‑path utilization; structured logs.

## Optional Dependencies

- CUDA/Torch for IPC path; Bodo for planning acceleration; UCX‑Py and mpi4py for fallbacks; pyzmq for control plane; Redis for HA; CuPy + KvikIO for GDS.

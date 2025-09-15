Tuning Guide

Overview

This guide summarizes practical knobs to tune Hotweights behavior across different environments and SLOs.

CUDA-IPC Replication

- Window and Backpressure
  - HOTWEIGHTS_IPC_INFLIGHT_BUCKETS: base window (default 4)
  - HOTWEIGHTS_IPC_INFLIGHT_BYTES: cap of in-flight bytes (0=off)
  - HOTWEIGHTS_IPC_ADAPT=1: adaptive window targetting HOTWEIGHTS_IPC_TARGET_BUCKET_MS (default 250)
  - HOTWEIGHTS_IPC_MIN_INFLIGHT, HOTWEIGHTS_IPC_MAX_INFLIGHT: window bounds
  - Metrics: hotweights_ipc_window, hotweights_ipc_recommended_window, hotweights_ipc_inflight_buckets/bytes.

- Copy Tuning (NVLink-friendly)
  - HOTWEIGHTS_IPC_COPY_AUTOTUNE=1 enables heuristics based on peer access
  - HOTWEIGHTS_IPC_COPY_CHUNK_MB (default 16) and HOTWEIGHTS_IPC_COPY_STREAMS (default 2)
  - Larger chunks and more streams often help on NVLink nodes; smaller on PCIe.

- GPUDirect Storage (optional)
  - HOTWEIGHTS_USE_GDS=1 with CuPy + KvikIO installed
  - Falls back to CPU assemble + H2D copy if unavailable

Topology Metrics

- Congestion risk (0–1): hotweights_ipc_congestion_risk
- Per-path utilization (share of chunk assignments): hotweights_ipc_path_utilization{path="src->dst"}
- Use these to verify window settings and validate path balance.

KV-Cache Migration

- Controls
  - HOTWEIGHTS_KV_ALLOW_TRANSFORMS=1: enable conservative transforms (dtype/RoPE)
  - HOTWEIGHTS_KV_HEAD_MAP (inline JSON) or HOTWEIGHTS_KV_HEAD_MAP_FILE (JSON file)
  - HOTWEIGHTS_KV_HEAD_ORDER=grouped|interleaved influences derived maps
  - CLI: hotweights kv-check --heads H --kv-heads KvH --order grouped

Optimizer State

- Controls
  - HOTWEIGHTS_OPT_POLICY=preserve|reset|attenuate
  - HOTWEIGHTS_OPT_ATTENUATION (float)
  - CLI: hotweights opt-check --updated N --unchanged M --policy attenuate --attenuation 0.5

HA Control Plane

- HOTWEIGHTS_COORD_BACKEND=redis to use Redis; HOTWEIGHTS_REDIS_URL
- HOTWEIGHTS_HANDLE_TTL seconds; handle acks shorten TTL.

CPU Transport Fallbacks (auto-selected)

- UCX: HOTWEIGHTS_UCX_CHUNK_MB, HOTWEIGHTS_UCX_CONCURRENCY, HOTWEIGHTS_UCX_INFLIGHT_LIMIT_MB, HOTWEIGHTS_UCX_RETRIES, HOTWEIGHTS_UCX_RETRY_DELAY_MS
- MPI: --window, --mpi-chunk-mb CLI flags (optional overrides)

GPU Broadcast Replication (ROCm/Intel/NVIDIA without IPC)

- Broadcast Chunking
  - HOTWEIGHTS_BCAST_CHUNK_MB (default 32): size per broadcast block.
  - Tune larger for high-bandwidth fabrics to reduce overhead; smaller to reduce latency jitter.

- Device-side Scatter Tuning
  - HOTWEIGHTS_GPU_COPY_AUTOTUNE=1: enable heuristic tuning based on peer access (CUDA).
  - HOTWEIGHTS_GPU_COPY_CHUNK_MB (default 16): per-copy chunk size on device tensors.
  - HOTWEIGHTS_GPU_COPY_STREAMS (default 2 on CUDA): number of parallel CUDA streams for scatter.

- Prefetch/Overlap
  - HOTWEIGHTS_BCAST_PREFETCH=1: on root, assemble and copy the next bucket to device while broadcasting the current one.
  - Helps overlap CPU/disk and H2D with network broadcast; disable if it contends on your platform.

- Backends
  - AMD ROCm: torch.distributed backend="nccl" uses RCCL under the hood.
  - Intel XPU: backend="ccl" via oneccl_bindings_for_pytorch; falls back to CPU paths if unavailable.

- Microbenchmark Autotune (optional)
  - Broadcast: HOTWEIGHTS_BCAST_AUTOTUNE=1 runs a quick world broadcast microbench and sets HOTWEIGHTS_BCAST_CHUNK_MB automatically (default dataset 128 MiB; override via HOTWEIGHTS_BCAST_AUTOTUNE_MB; candidates via HOTWEIGHTS_BCAST_AUTOTUNE_CHUNKS_MB="8,16,32,64").
  - Device copy: HOTWEIGHTS_GPU_COPY_MICRO=1 runs a local device copy microbench and sets HOTWEIGHTS_GPU_COPY_CHUNK_MB and HOTWEIGHTS_GPU_COPY_STREAMS (size via HOTWEIGHTS_GPU_COPY_AUTOTUNE_MB; candidates via HOTWEIGHTS_GPU_COPY_MICRO_CHUNKS_MB and HOTWEIGHTS_GPU_COPY_MICRO_STREAMS).

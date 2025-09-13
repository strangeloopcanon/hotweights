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

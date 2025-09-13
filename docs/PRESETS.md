Environment Presets

This doc provides ready-to-use environment presets for common deployments.

NVLink Nodes (intra-node, multi-GPU)

Use larger device copy chunks and more CUDA streams; allow the adaptive window to target smaller per-bucket times.

```
export HOTWEIGHTS_IPC_COPY_AUTOTUNE=1          # enable peer-access heuristic
export HOTWEIGHTS_IPC_COPY_CHUNK_MB=64         # explicit override (optional)
export HOTWEIGHTS_IPC_COPY_STREAMS=4           # explicit override (optional)
export HOTWEIGHTS_IPC_INFLIGHT_BUCKETS=6       # base window
export HOTWEIGHTS_IPC_ADAPT=1                  # adaptive windowing on
export HOTWEIGHTS_IPC_MIN_INFLIGHT=2
export HOTWEIGHTS_IPC_MAX_INFLIGHT=12
export HOTWEIGHTS_IPC_TARGET_BUCKET_MS=200
# (Optional) GPUDirect Storage
export HOTWEIGHTS_USE_GDS=1
```

PCIe-only Nodes

Prefer smaller chunk sizes and fewer streams; slightly larger target per-bucket time.

```
export HOTWEIGHTS_IPC_COPY_AUTOTUNE=0          # disable heuristics
export HOTWEIGHTS_IPC_COPY_CHUNK_MB=8
export HOTWEIGHTS_IPC_COPY_STREAMS=2
export HOTWEIGHTS_IPC_INFLIGHT_BUCKETS=3
export HOTWEIGHTS_IPC_ADAPT=1
export HOTWEIGHTS_IPC_MIN_INFLIGHT=1
export HOTWEIGHTS_IPC_MAX_INFLIGHT=6
export HOTWEIGHTS_IPC_TARGET_BUCKET_MS=350
```

Redis-backed HA Coordinator

Enable Redis for handle lifecycle and HA state. Acks shorten TTL automatically.

```
export HOTWEIGHTS_COORD_BACKEND=redis
export HOTWEIGHTS_REDIS_URL=redis://127.0.0.1:6379/0
export HOTWEIGHTS_HANDLE_TTL=45
hotweights coord-serve --endpoint tcp://127.0.0.1:5555
```

GQA KV-Cache Migration (safe mode)

Enable conservative transforms with a derived head map; choose grouping order.

```
export HOTWEIGHTS_KV_ALLOW_TRANSFORMS=1
export HOTWEIGHTS_KV_HEAD_ORDER=grouped   # or interleaved
```

Optimizer State Policy (training)

```
export HOTWEIGHTS_OPT_POLICY=attenuate
export HOTWEIGHTS_OPT_ATTENUATION=0.5
```

Notes

- The adaptive window publishes current vs recommended window via metrics; use the provided Grafana dashboard to monitor inflight buckets/bytes, congestion risk, and path utilization.
- For multi-node clusters, pair CUDA-IPC intra-node with the auto-selected CPU transport (MPI > UCX) for inter-node traffic.

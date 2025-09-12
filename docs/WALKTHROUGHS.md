Example Walkthroughs

This guide shows end-to-end flows using the presets and CLI tools.

Prereqs

- Python 3.10+
- Install extras: `pip install -e .[extras]`
- Optional: Redis for HA (`brew install redis` or `apt-get install redis-server`), CuPy+KvikIO for GDS.

1) Single-Node NVLink + GDS (CUDA-IPC)

Targets a multi-GPU node (NVLink/NVSwitch). Assembles buckets directly into device memory (GDS) and uses CUDA-IPC for zero-copy intra-node distribution.

```
source scripts/preset_env_nvlink.sh
export HOTWEIGHTS_USE_GDS=1            # optional

# Build toy checkpoints
rm -rf demo_ckpt_a demo_ckpt_b plan.json
mkdir -p demo_ckpt_a demo_ckpt_b
dd if=/dev/zero of=demo_ckpt_a/a.bin bs=1m count=16 >/dev/null 2>&1
cp demo_ckpt_a/a.bin demo_ckpt_b/a.bin
dd if=/dev/zero of=demo_ckpt_b/b.bin bs=1m count=8 >/dev/null 2>&1

# Publish, plan, and verify
hotweights publish --checkpoint demo_ckpt_a --version v0 --output m_prev.json
hotweights publish --checkpoint demo_ckpt_b --version v1 --output m_next.json
hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 64 --output plan.json
hotweights verify-plan --plan plan.json --require-consumers || true

# Replicate (CUDA-IPC preferred); metrics on :9097
hotweights metrics --start --port 9097
RANK=0 WORLD_SIZE=1 hotweights replicate --plan plan.json --device cuda

# Inspect metrics and use dashboards/hotweights.json in Grafana.
```

2) Two Ranks over UCX (fallback broadcast)

Runs a small UCX broadcast with two ranks (no CUDA requirement). Each rank receives its own scatter into host buffers and verifies hashes.

```
./scripts/run_ucx_broadcast_demo.sh plan.json
# Rank 0 listens and sends; rank 1 connects and receives
```

3) Redis HA Coordinator + Worker Agents (CUDA-IPC)

Starts a Redis-backed HA coordinator and worker agent(s) that replicate via CUDA-IPC and commit on quorum.

```
source scripts/preset_env_redis.sh
hotweights coord-serve --endpoint tcp://127.0.0.1:5555 &

# Build toy checkpoints and plan
rm -rf demo_ckpt_a demo_ckpt_b plan.json
mkdir -p demo_ckpt_a demo_ckpt_b
echo "hello" > demo_ckpt_a/a.bin
cp demo_ckpt_a/a.bin demo_ckpt_b/a.bin
echo "world" > demo_ckpt_b/b.bin

hotweights publish --checkpoint demo_ckpt_a --version v0 --output m_prev.json
hotweights publish --checkpoint demo_ckpt_b --version v1 --output m_next.json
hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 8 --output plan.json
hotweights verify-plan --plan plan.json --world-size 1 || true

# Submit plan and begin
hotweights coord-submit-plan --endpoint tcp://127.0.0.1:5555 --plan plan.json
hotweights begin --endpoint tcp://127.0.0.1:5555 --version v1

# Start a worker agent (CUDA-IPC path)
HOTWEIGHTS_COORD=tcp://127.0.0.1:5555 RANK=0 WORLD_SIZE=1 hotweights worker --device cuda

# Coordinator metrics on :9100; worker metrics on :9099.
```

CLI Utilities

- KV mapping: `hotweights kv-check --heads 32 --kv-heads 8 --order grouped`
- Optimizer policy: `hotweights opt-check --updated 100 --unchanged 100 --policy attenuate --attenuation 0.5`
- Plan validation: `hotweights verify-plan --plan plan.json --world-size 8 --require-consumers`

Notes

- For multi-node deployments, pair CUDA-IPC intra-node with UCX or MPI inter-node as needed; use TP group mappings (docs/TP_GROUPS.md) to scope consumer_ranks.
- The HA coordinator can be run without Redis; Redis adds durability and TTL cleanup but is optional.


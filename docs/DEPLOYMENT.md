# Deployment & Validation Guide

This guide walks through bringing up Hotweights locally, integrating with vLLM, and preparing for multi‑GPU validation later.

## Prerequisites
- Python 3.10+
- Install extras: `pip install -e .[extras]`
  - Optional: `pyzmq` (coordinator), `mpi4py` (MPI), `ucx-py` (UCX/RDMA), `prometheus-client` (metrics)
- UCX notes: requires libucx on your system; on CPU‑only hosts, UCX paths are optional.

## Coordinator & Metrics
```
hotweights coord-serve --endpoint tcp://127.0.0.1:5555 &
# Metrics: http://localhost:9100/metrics
```

To enable Redis-backed HA state:

```
pip install redis
HOTWEIGHTS_COORD_BACKEND=redis HOTWEIGHTS_REDIS_URL=redis://127.0.0.1:6379/0 \
  hotweights coord-serve --endpoint tcp://127.0.0.1:5555 &
```

## Create Manifests & Plan
```
hotweights publish --checkpoint ./ckpt_a --version v0 --output m_prev.json
hotweights publish --checkpoint ./ckpt_b --version v1 --output m_next.json
hotweights plan --prev m_prev.json --next m_next.json --bucket-mb 64 --output plan.json
hotweights coord-submit-plan --endpoint tcp://127.0.0.1:5555 --plan plan.json
hotweights begin --endpoint tcp://127.0.0.1:5555 --version v1
```

## Replicate Locally
```
hotweights replicate --plan plan.json --verify --commit --device cpu
# Metrics: http://localhost:9097/metrics
```

Enable CUDA-IPC with GPUDirect Storage (optional):

```
pip install cupy-cuda11x kvikio
HOTWEIGHTS_USE_GDS=1 hotweights replicate --plan plan.json --device cuda
```

## UCX Broadcast Demo (two ranks)
```
./scripts/run_ucx_broadcast_demo.sh plan.json
# WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT control rank orchestration
```

## Worker Agent (UCX late-join)
```
./scripts/run_worker_ucx_demo.sh
# Workers use UCX P2P with topology-aware donor selection and striping
# Worker metrics: http://localhost:9099/metrics
```

## vLLM Integration
- Background binder (no refactor):
  ```python
  from hotweights.adapters.vllm_auto import install_autobind
  def name_map_fn(plan):
      return {it['key']: it['tensor'].replace('/', '.') for b in plan['buckets'] for it in b['items']}
  install_autobind(name_map_fn, endpoint="tcp://127.0.0.1:5555", use_mpi=True, pinned=True, device="cuda")
  ```
- WorkerExtension (preferred for tight pause SLOs):
  ```python
  from hotweights.adapters.vllm_extension import HotweightsWorkerExtension
  ext = HotweightsWorkerExtension(endpoint="tcp://127.0.0.1:5555", use_mpi=True, pinned=True, device="cuda")
  ext.bind_module(model_module)
  # Wire ext.begin_update/finalize_shard/precommit/commit into your vLLM worker lifecycle.
  # See examples/vllm_extension_register.py
  ```

## Tuning & Validation
- Broadcast: choose MPI (`--mpi`) or UCX (`--ucx`).
- Buckets: plan with `--bucket-mb` or `--auto --alpha`.
- UCX broadcast: adjust `MASTER_ADDR/PORT`, chunk size in UCXReplicator if needed.
- Late-join P2P: UCX striping slices shards across donors, adjust concurrency in fetch_ranges_concurrent.
- Staging: use `--pinned` for pinned host buffers; CUDA stream copies are used when available.
- Dashboards: import `dashboards/hotweights.json` into Grafana.

## Benchmark
```
python bench/bench_replicate.py plan.json --repeat 3 --pinned
```

## Notes
- All GPU‑specific paths fall back to CPU; safe on Apple Silicon.
- For production SLOs, wire the WorkerExtension hooks into your vLLM version’s lifecycle so shards stream and commit within a tight pause window.

## Example Walkthroughs

See `docs/WALKTHROUGHS.md` for:

- Single-node NVLink + GDS (CUDA-IPC)
- Two-rank UCX broadcast
- Redis-backed HA coordinator + worker agents

**Repo name:** `hotweights`
**Tagline:** Fast, versioned weight updates for LLM serving and training, with Bodo‑accelerated planning and transport.

---

## 1) Objective

Build a production‑grade system to **hot‑swap large model weights** across a GPU cluster for:

* **Inference**: live vLLM servers reload weights with sub‑second pause.
* **Training / RL**: distributed trainers replace parameters at safe barriers without reinit.

The system must minimize bytes moved and maximize wire speed. It must tolerate worker churn and partial failures.


Inspiration: https://github.com/MoonshotAI/checkpoint-engine

**Non‑immediate goals:** model training, optimizer math, tokenizer changes, KV‑cache preservation across swaps.

---

## 2) Outcomes and targets

* **8–32 GPUs, 2–5 GiB update:** end‑to‑end ≤ 7–15 s; serving pause at commit ≤ 300 ms.
* **Delta mode:** only changed shards move; zero false commits (hash‑verified).
* **Late join:** new workers catch up without global restart.
* **Training swap:** in‑place param copy across DDP/FSDP/ZeRO at step boundaries.

---

## 3) System architecture

**Planes**

* **Control plane (Python + ZeroMQ):** versioned update lifecycle, worker registration, late‑join orchestration.
* **Data plane (Bodo + mpi4py):** Bodo‑accelerated delta compute and bucket planning; mpi4py for broadcast; optional P2P for late join.
* **Server plane (Python + CUDA):** vLLM WorkerExtension for hot‑reload; Trainer hook for in‑place param copy.

**Data flow**

1. Publisher posts new checkpoint + manifest.
2. Planner computes **delta** and **bucket schedule** (Bodo DataFrame + JIT).
3. Transport streams buckets: root reads → **broadcast** → stage on each node.
4. Worker extension **loads to shadow** params → **commit** flips pointers atomically.
5. Late joiners P2P fetch missing buckets then join at next commit.

---

## 4) Repository layout

```
hotweights/
  hotweights/
    __init__.py
    config.py
    manifest.py            # formats + hashing
    planner_bodo.py        # delta + bucketization + donor assignment
    transport/
      __init__.py
      mpi_stream.py        # mpi4py broadcast pipeline
      p2p.py               # donor/receiver streams (TCP/UCX)
    coordinator/
      server.py            # ZMQ PUB/REP
      client.py
    staging/
      host_agent.py        # /dev/shm pinned-host staging; CUDA IPC later
    adapters/
      vllm_ext.py          # WorkerExtension for inference hot-reload
      trainer_swap.py      # DDP/FSDP/ZeRO in-place copy hook
    telemetry/
      nvml.py              # free VRAM, NIC info
      metrics.py           # stage timings, pause budget
    cli.py                 # hotweights publish/commit/status
  examples/
    vllm_demo.sh
    trainer_demo.py
  tests/
    test_manifest.py
    test_planner.py
    test_stream_small.py
    test_trainer_swap.py
  README.md
  LICENSE
  pyproject.toml
```

---

## 5) Interfaces

### 5.1 Manifest (versioned, deterministic)

```json
// checkpoint.manifest.json
{
  "model_id": "Qwen3-72B",
  "version": "2025-09-11T12:00:00Z",
  "tensors": [{
    "name": "layers.0.attn.q_proj.weight",
    "dtype": "bf16",
    "shape": [12288, 12288],
    "partitioning": {"tp": 8, "pp": 1, "scheme": "row"},
    "quant": null,
    "shards": [{
      "rank": 0, "bytes": 18874368, "hash": "xxh64:…", "uri": "file:///…/tensors/…/0.bin"
    }]
  }]
}
```

### 5.2 Control RPCs (ZeroMQ REP/PUB)

* `register(worker_id, caps)` → `{ok, current_version}`
* `begin(version, manifest_digest)` → PUB `{event:"begin", version}`
* `commit(version)` → PUB `{event:"commit", version}`
* `abort(reason)` → PUB `{event:"abort", reason}`
* `who_has(bucket_id)` → donor set

### 5.3 Planner API (Bodo)

```python
# DataFrames: cols = [tensor, shard_rank, nbytes, hash, path]
Delta = compute_delta(prev_df, next_df)                 # Bodo DF
Plan  = pack_buckets(Delta, max_bucket_bytes)           # Bodo JIT greedy bin-pack
Donors = assign_donors(HaveTable, NeedTable, k=2)       # Bodo DF join
```

### 5.4 Transport API

```python
class BucketStreamer:
    def __iter__(self): yield (bucket_id, np.ndarray[uint8])  # root provides
class Replicator:
    def replicate(self, stream: BucketStreamer): pass          # broadcasts to all ranks

# mpi_stream.py
replicator = MPIReplicator(comm, bucket_bytes=512<<20, window=2)
replicator.replicate(stream)
```

### 5.5 Staging API (per-node host agent)

```python
class HostAgent:
    def reserve(self, key: str, nbytes: int) -> memoryview      # pinned host buffer
    def write(self, key: str, off: int, buf: memoryview) -> None
    def seal(self, key: str) -> str                              # returns shm path
```

### 5.6 vLLM Inference Adapter

```python
class HotReloadExtension:
    def begin_update(self, version: str, manifest: dict) -> None: ...
    def request_buffer(self, tensor: str, shard_rank: int, nbytes: int) -> "Target":
        """Return pinned-host pointer or CUDA IPC handle description."""
    def finalize_shard(self, tensor: str, shard_rank: int, hash: str) -> None: ...
    def precommit(self) -> None: ...
    def commit(self, version: str) -> None: ...
```

### 5.7 Trainer Swap Adapter

```python
class SwapHook:
    def maybe_swap(self, manifest: dict) -> None:
        torch.cuda.synchronize()
        with torch.no_grad():
            for param, new_slice in shard_iter():
                assert param.shape == new_slice.shape
                param.data.copy_(new_slice, non_blocking=True)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
```

### 5.8 CLI

```
hotweights publish --checkpoint /mnt/ckpt/Qwen3-72B --version 2025-09-11 \
  --bucket-mb 512 --delta-from 2025-09-08

hotweights commit --version 2025-09-11
hotweights status
```

---

## 6) Algorithms and where Bodo adds value

### 6.1 Delta compute (Bodo DataFrame)

* Join `next_manifest` with `prev_manifest` on `(tensor, shard_rank)`.
* Filter `hash != prev_hash` or `missing`.
* Sort by `nbytes` desc.

```python
# planner_bodo.py
import pandas as pd, numpy as np, bodo

@bodo.jit(cache=True)
def compute_delta(prev_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    j = next_df.merge(prev_df, on=["tensor","shard_rank"], how="left", suffixes=("", "_prev"))
    mask = j["hash_prev"].isna() | (j["hash_prev"] != j["hash"])
    return j[mask][["tensor","shard_rank","nbytes","hash","path"]]
```

### 6.2 Bucketization (Bodo JIT)

* Greedy bin pack to `bucket_bytes` target; stable ordering by size to reduce tail.

```python
@bodo.jit(cache=True)
def pack_buckets(delta_df: pd.DataFrame, bucket_bytes: int) -> pd.DataFrame:
    df = delta_df.sort_values("nbytes", ascending=False).reset_index(drop=True)
    bucket_id = np.empty(len(df), dtype=np.int64)
    cur, acc = 0, 0
    for i in range(len(df)):
        sz = int(df["nbytes"].iat[i])
        if acc + sz > bucket_bytes and acc > 0:
            cur += 1; acc = 0
        bucket_id[i] = cur; acc += sz
    df["bucket_id"] = bucket_id
    return df
```

### 6.3 Broadcast pipeline (mpi4py + Bodo‑optimized buffers)

* Root mmaps shard files into preallocated **numpy** buffers.
* Double‑buffered pipeline: while GPU copies bucket N, broadcast bucket N+1.
* Use `MPI.Ibcast` on the contiguous bucket buffer; receivers scatter into per‑shard staging via Bodo‑compiled copy loops (no Python per‑shard overhead).

### 6.4 Late‑join P2P (Bodo DF planner + Python streamers)

* Maintain table `Have(bucket_id, rank)`.
* For `Need`, assign k donors per bucket (round‑robin or topology‑aware).
* P2P transport: TCP sockets or UCX‑Py for RDMA when available.
* Striping across donors for throughput.

### 6.5 Autotuning

* Collect free VRAM per rank (NVML). Choose `bucket_bytes = min_rank_free_vram * α` with α≈0.25–0.33.
* Adapt window size (`inflight buckets`) to NIC saturation metrics.

---

## 7) vLLM integration details

* Start vLLM with the extension class and a “dummy” load format so parameters are injectible.
* Extension responsibilities:

  * Allocate **shadow** parameter storage per layer.
  * Copy staged shards host→device with `torch.cuda.memcpy_async` streams.
  * `precommit()` barrier in worker.
  * `commit()` flips pointers atomically; free old storage lazily.
* Backpressure: if VRAM low, request smaller buckets; extension can return pinned‑host buffers and perform layer‑by‑layer device copies.

---

## 8) Training integration details

* Swap occurs at step boundaries only.
* **DDP**: in‑place `param.data.copy_(new)` preserves wrappers and grads.
* **FSDP/ZeRO**: operate on local shards only; never all‑gather full params for swap.
* **Mixed precision**: swap the exact dtype in use (bf16/fp16/fp8). Shapes must match.
* Optimizer state:

  * Default: keep moments (fastest) when deltas are small.
  * Option: attenuate moments after swap, or reset per‑layer when hash changed.

---

## 9) Security, integrity, and idempotency

* Per‑shard `xxhash64` (fast) plus optional `blake3` for auditing.
* Two‑phase update: `stage → verify → precommit → commit`.
* Idempotent retries per bucket; partial failures roll back to prior version.

---

## 10) Telemetry and SLOs

* Timers: file read, broadcast, stage write, H2D copy, pointer‑flip pause.
* Counters: bytes moved, buckets, retries, P2P hits.
* Health: min/max free VRAM, NIC throughput, CPU stall time.
* Export: Prometheus endpoint from each node.

---

## 11) Testing strategy

* **Unit**: manifest parsing, hashing, bucketization determinism.
* **Integration (single node)**: 8×GPU, 2–3 GiB swap; assert pause ≤ 300 ms; correctness (logits before/after match new weights).
* **Integration (multi node)**: 2×8 GPUs over 100 GbE; total time and bandwidth targets.
* **Chaos**: kill a worker mid‑update; ensure others commit or abort cleanly.
* **Training**: DDP swap every N steps on a toy model; loss continuity checks.

---

## 12) Milestones

**MVP (2–3 weeks)**

* Manifest, hashing, planner (Bodo DF + JIT).
* MPI broadcast pipeline (mpi4py), pinned‑host staging.
* vLLM extension: shadow alloc + commit, CPU→GPU copy.
* CLI: `publish`, `commit`, `status`.
* Metrics: stage timings, pause budget.

**M2 (3–5 weeks)**

* Delta mode on by default.
* Late‑join P2P with donor striping.
* Autotune bucket size via VRAM telemetry.
* Trainer swap hook for DDP/FSDP.

**M3 (4–6 weeks)**

* CUDA IPC zero‑copy path in staging and extension.
* UCX/OFI backend for RDMA where present.
* Topology‑aware donor selection and NIC pinning.

---

## 13) Implementation notes and constraints

* **Bodo use**: confine to compute‑heavy parts (delta, packing, copy loops). Network calls remain in Python via mpi4py for portability. This gets most of Bodo’s speed without fighting MPI bindings inside JIT.
* **I/O**: use `mmap` + readahead for root file reads; preallocate numpy buffers.
* **Memory**: bucket sizing must account for shadow storage + in‑flight copies; enforce safety factor α.
* **Compatibility**: only equal shapes/dtypes across swaps; enforce at `begin_update`.
* **Failure domains**: coordinator remains stateless; workers retry shards; version monotonicity guarantees idempotence.

---

## 14) Example code stubs

**Planner**

```python
# hotweights/planner_bodo.py
import pandas as pd, numpy as np, bodo

@bodo.jit(cache=True)
def compute_delta(prev_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    j = next_df.merge(prev_df, on=["tensor","shard_rank"], how="left", suffixes=("", "_prev"))
    mask = j["hash_prev"].isna() | (j["hash_prev"] != j["hash"])
    return j.loc[mask, ["tensor","shard_rank","nbytes","hash","path"]]

@bodo.jit(cache=True)
def pack_buckets(delta_df: pd.DataFrame, bucket_bytes: int) -> pd.DataFrame:
    df = delta_df.sort_values("nbytes", ascending=False).reset_index(drop=True)
    n = len(df); bucket_id = np.empty(n, dtype=np.int64)
    cur, acc = 0, 0
    for i in range(n):
        sz = int(df["nbytes"].iat[i])
        if acc + sz > bucket_bytes and acc > 0: cur += 1; acc = 0
        bucket_id[i] = cur; acc += sz
    df["bucket_id"] = bucket_id
    return df
```

**Transport**

```python
# hotweights/transport/mpi_stream.py
import numpy as np
from mpi4py import MPI

class MPIReplicator:
    def __init__(self, comm=MPI.COMM_WORLD, bucket_bytes=512<<20, window=2):
        self.comm, self.bucket_bytes, self.window = comm, bucket_bytes, window
        self.rank = comm.Get_rank()

    def replicate(self, bucket_iter):
        inflight = []
        for bucket_id, buf in bucket_iter:
            assert isinstance(buf, np.ndarray) and buf.flags['C_CONTIGUOUS']
            req = self.comm.Ibcast(buf, root=0)
            inflight.append((bucket_id, buf, req))
            if len(inflight) >= self.window:
                self._drain(inflight.pop(0))
        for x in inflight: self._drain(x)

    def _drain(self, item):
        bucket_id, buf, req = item
        req.Wait()
```

**vLLM adapter skeleton**

```python
# hotweights/adapters/vllm_ext.py
import torch

class HotReloadExtension:
    def __init__(self):
        self.shadow = {}  # tensor_name -> device tensor

    def begin_update(self, version, manifest):
        self.version = version
        # allocate or reuse shadow storage per tensor/layer

    def request_buffer(self, tensor, shard_rank, nbytes):
        # return a pinned-host buffer pointer via host_agent
        return {"key": f"{tensor}:{shard_rank}", "nbytes": nbytes}

    def finalize_shard(self, tensor, shard_rank, hash_):
        # copy from pinned host to shadow on a stream
        pass

    def precommit(self):
        torch.cuda.synchronize()

    def commit(self, version):
        # atomically flip module parameter pointers to shadow
        pass
```

**Trainer swap**

```python
# hotweights/adapters/trainer_swap.py
import torch, contextlib

@contextlib.contextmanager
def swap_barrier():
    torch.distributed.barrier()
    try: yield
    finally: torch.distributed.barrier()

def in_place_swap(shard_iter, optimizer):
    torch.cuda.synchronize()
    with torch.no_grad():
        for param, new_slice in shard_iter():
            param.data.copy_(new_slice, non_blocking=True)
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
```

---

## 15) Deployment

* Run coordinator as a small container (`hotweights-coord`).
* Launch vLLM with the extension class and host‑agent sidecar per pod.
* Publisher runs `hotweights publish …` on rank‑0; transport uses `mpirun` or spawn‑mode.
* Expose Prometheus metrics from host‑agent and adapters.

---

## 16) Open questions

* vLLM extension API surfaces to bind: exact method names and lifecycle hooks in your target version.
* CUDA IPC handle passing: finalize interface for zero‑copy path.
* UCX availability in your clusters for RDMA P2P.
* Optimizer state policy knobs for training swaps.

---

## 17) Work breakdown (assignable units)

1. **Manifest + hashing** (2–3 d)
2. **Planner (Bodo) delta + buckets** (3–4 d)
3. **MPI transport pipeline** (3–4 d)
4. **Host agent (pinned host)** (2 d)
5. **Coordinator (ZMQ)** (2 d)
6. **vLLM extension MVP** (4–6 d)
7. **CLI + metrics** (2 d)
8. **Tests + small‑cluster benchmarks** (5–7 d)
9. **Trainer swap adapter** (3–5 d)
10. **Late‑join P2P** (4–6 d)

---

## 18) Decision rationale

* Use **Bodo** where it wins: vectorized delta selection, compiled packing, and eliminating Python overhead in copy/plan loops.
* Keep network IO in **mpi4py** for portability.
* Keep GPU hot‑swap in the **server/trainer extensions** where CUDA context and pointer ownership live.

Hand this spec to a senior engineer. They can start with the planner + transport + vLLM adapter and hit MVP quickly.

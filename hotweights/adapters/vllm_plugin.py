"""vLLM integration helpers.

Designed to be called from inside a vLLM worker process after model load.
Provides a function to fetch a plan from the coordinator, replicate buckets
locally (or via MPI), and apply weights into a torch.nn.Module with async H2D.

Usage in vLLM worker (pseudo-code):

  from hotweights.adapters.vllm_plugin import update_weights_from_coordinator

  def name_map_fn(items):
      # Map staged keys to module param dotted paths
      return { it['key']: it['tensor'].replace('/', '.') for b in plan['buckets'] for it in b['items'] }

  update_weights_from_coordinator(module, name_map_fn, endpoint="tcp://coord:5555", use_mpi=True)

This file avoids importing vLLM directly to keep dependencies light.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from ..coordinator.zmq_client import Client
from ..staging.host_agent import HostAgent
from ..transport.mpi_stream import MPIReplicator
from ..adapters.vllm_ext import HotReloadExtension
from ..core.replicate import assemble_bucket as _assemble_bucket
from ..core.replicate import scatter_bucket as _scatter_bucket
from ..core.replicate import verify_items as _verify_items
from ..telemetry.metrics import Timer
from ..utils.torch_utils import torch_dtype_from_numpy_str


def _normalize_key(s: str) -> str:
    s = s.replace("/", ".")
    for ext in (".bin", ".npy", ".pt"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    return s.lower()


def infer_name_map(module, plan: dict) -> Dict[str, str]:  # noqa: ANN001
    """Infer a mapping from staged keys to module parameter names.

    Heuristics:
    - Normalize manifest names (replace '/' with '.', strip extensions).
    - Prefer parameters whose names end with the normalized key suffix.
    - Boost score when dtype/shape match.
    """
    if torch is None:
        # Best-effort: identity mapping
        return {it["key"]: _normalize_key(it["tensor"]) for b in plan.get("buckets", []) for it in b["items"]}

    # Collect parameter info
    params = list(module.named_parameters())
    out: Dict[str, str] = {}
    for b in plan.get("buckets", []):
        for it in b["items"]:
            key = it["key"]
            norm = _normalize_key(it["tensor"])
            shape = tuple(int(x) for x in (it.get("shape") or []))
            dtype_str = (it.get("dtype") or "").replace("|", "")
            t_dtype = torch_dtype_from_numpy_str(dtype_str)
            best = (0.0, None)  # score, name
            for pname, p in params:
                score = 0.0
                if pname.lower().endswith(norm):
                    score += 2.0
                elif norm.endswith(pname.lower()):
                    score += 1.0
                if shape and tuple(p.shape) == shape:
                    score += 1.0
                if t_dtype is not None and hasattr(p, "dtype") and p.dtype == t_dtype:
                    score += 1.0
                if score > best[0]:
                    best = (score, pname)
            out[key] = best[1] or norm
    return out


def update_weights_from_coordinator(
    module,  # torch.nn.Module
    name_map: Dict[str, str] | Callable[[dict], Dict[str, str]] | None,
    endpoint: str = "tcp://127.0.0.1:5555",
    use_mpi: bool = False,
    pinned: bool = True,
    verify: bool = False,
    device: str = "cuda",
) -> None:
    c = Client(endpoint)
    resp = c.call("get_plan")
    plan = resp.get("plan")
    if not plan:
        raise RuntimeError("No plan available in coordinator")

    host = HostAgent(use_pinned=pinned)
    replicator = None
    if use_mpi:
        try:
            replicator = MPIReplicator()
        except Exception:  # pragma: no cover
            replicator = None

    all_items: list[dict] = []
    if replicator is None:
        for b in plan.get("buckets", []):
            items = b["items"]
            buf = _assemble_bucket(items)
            _scatter_bucket(host, items, buf)
            if verify:
                _verify_items(host, items)
            all_items.extend(items)
    else:
        bucket_bufs: list[tuple[dict, "numpy.ndarray"]] = []
        import numpy as np

        def gen():
            for b in plan.get("buckets", []):
                items = b["items"]
                size = int(b["size"])  # precomputed during planning
                if replicator.rank == 0:
                    buf = _assemble_bucket(items)
                else:
                    buf = np.empty(size, dtype=np.uint8)
                bucket_bufs.append((b, buf))
                yield (int(b["bucket_id"]), buf)

        with Timer("broadcast"):
            replicator.replicate(gen())
        for b, buf in bucket_bufs:
            items = b["items"]
            _scatter_bucket(host, items, buf)
            if verify and replicator.rank == 0:
                _verify_items(host, items)
            all_items.extend(items)

    # derive name mapping
    if callable(name_map):
        mapping = name_map(plan)
    elif isinstance(name_map, dict):
        mapping = name_map
    else:
        mapping = infer_name_map(module, plan)

    ext = HotReloadExtension()
    ext.apply_from_host_to_module(all_items, host, module, mapping, device=device)
    ext.precommit(); ext.commit(plan.get("version", "unknown"))

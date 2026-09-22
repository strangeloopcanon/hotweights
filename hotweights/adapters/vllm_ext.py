"""vLLM WorkerExtension skeleton for hot reload (stub).

Torch is optional; falls back to NumPy arrays when unavailable.
"""
from __future__ import annotations

try:  # optional
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
import numpy as np
from ..core.errors import HotweightsError
from ..utils.torch_utils import torch_dtype_from_numpy_str
from ..telemetry.metrics import Timer
from ..telemetry.prom import Histogram


class SwapError(HotweightsError):
    """Raised when a weight update cannot be staged or verified safely.

    Callers should treat this as "do not touch the live module": the update
    must be discarded, not partially applied.
    """


class HotReloadExtension:
    def __init__(self) -> None:
        self.shadow: dict[str, object] = {}
        self.params: dict[str, object] = {}
        self.version: str | None = None
        self._module = None
        self._map: dict[str, str] = {}
        self._host_buffers: dict[str, memoryview] = {}
        self._pinned: dict[str, object] = {}
        self._h2d_hist = Histogram(
            "hotweights_adapter_h2d_seconds", "H2D copy time per shard"
        )

    def begin_update(self, version: str, manifest: dict) -> None:  # noqa: ANN001
        self.version = version

    def request_buffer(self, tensor: str, shard_rank: int, nbytes: int):  # noqa: ANN001, ANN201
        key = f"{tensor}:{shard_rank}"
        # In a vLLM hook, this would allocate a pinned host buffer and return a pointer
        # Here we return a descriptor and keep a memoryview for finalize_shard
        if torch is not None:
            buf = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=True)
            mv = memoryview(buf.numpy())
            self._host_buffers[key] = mv
            self._pinned[key] = buf
            return {"key": key, "nbytes": nbytes, "buffer": mv}
        else:
            ba = bytearray(int(nbytes))
            mv = memoryview(ba)
            self._host_buffers[key] = mv
            return {"key": key, "nbytes": nbytes, "buffer": mv}

    def finalize_shard(self, tensor: str, shard_rank: int, hash_: str) -> None:  # noqa: ANN001
        # Copy from pinned host to shadow/device on a stream
        key = f"{tensor}:{shard_rank}"
        mv = self._host_buffers.get(key)
        if mv is None:
            return
        nbytes = len(mv)
        # Use dtype/shape hints if available via map (requires name mapping to target parameter)
        target_name = self._map.get(key)
        if target_name and self._module is not None and torch is not None:
            mod = self._module
            parts = target_name.split(".")
            for p in parts[:-1]:
                mod = getattr(mod, p)
            pname = parts[-1]
            param = getattr(mod, pname)
            # Copy bytes into typed CPU tensor then async H2D
            with Timer("h2d") as t:
                cpu_t = torch.empty_like(param.data, device="cpu", pin_memory=True)
                view_dst = cpu_t.view(torch.uint8)
                src = self._pinned.get(key)
                if isinstance(src, torch.Tensor) and src.dtype == torch.uint8:
                    view_dst[:nbytes].copy_(src[:nbytes])
                else:
                    src_buf = torch.frombuffer(  # type: ignore[attr-defined]
                        bytearray(mv.tobytes()), dtype=torch.uint8
                    )
                    view_dst[:nbytes].copy_(src_buf[:nbytes])
                if torch.cuda.is_available():
                    param.data.copy_(cpu_t.to(param.data.device, non_blocking=True))
                else:
                    param.data.copy_(cpu_t)
            self._h2d_hist.observe(t.elapsed)
        else:
            # fallback: store to shadow as bytes
            data = bytes(mv)
            if torch is not None:
                self.shadow[key] = torch.frombuffer(  # type: ignore[attr-defined]
                    bytearray(data), dtype=torch.uint8
                ).clone()
            else:
                self.shadow[key] = np.frombuffer(
                    bytearray(data), dtype=np.uint8
                ).copy()

    def precommit(self) -> None:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def commit(self, version: str) -> None:  # noqa: ANN001
        # Atomically flip module parameter pointers to shadow where possible.
        self.version = version
        self.params = dict(self.shadow)
        if self._module is not None and torch is not None:
            with torch.no_grad():
                for key, target_name in self._map.items():
                    if key not in self.shadow:
                        continue
                    t = self.shadow[key]
                    mod = self._module
                    parts = target_name.split(".")
                    for p in parts[:-1]:
                        mod = getattr(mod, p)
                    pname = parts[-1]
                    param = getattr(mod, pname)
                    if isinstance(t, torch.Tensor):
                        if t.shape == param.data.shape and t.dtype == param.data.dtype:
                            param.data = t.detach().clone()
                        else:
                            param.data.copy_(t.to(dtype=param.data.dtype).reshape_as(param.data))

    # --- helper for tests/local integration ---
    def ingest_from_host(self, items: list[dict], host) -> None:  # noqa: ANN001
        """Load staged host buffers into shadow CPU tensors.

        Stores tensors keyed by "tensor:rank". If dtype/shape present on items,
        reinterpret the buffer accordingly and store a typed tensor (CPU). If
        CUDA is available and pinned buffers are used, callers can move to GPU
        via ``to_device``.
        """
        for it in items:
            key = it["key"]
            n = int(it["nbytes"])  # expected length
            dtype = (it.get("dtype") or "uint8").replace("|", "")
            shape = it.get("shape")
            # Prefer pinned tensor from host when available
            pinned_t = getattr(host, "torch_tensor", lambda *_: None)(key)
            if torch is not None and pinned_t is not None:
                t = pinned_t[:n]
                if shape and isinstance(shape, list):
                    try:
                        t = t.view(dtype=getattr(torch, str(np.dtype(dtype))), shape=tuple(int(x) for x in shape))
                    except Exception:
                        pass
                self.shadow[key] = t.clone()  # own storage in shadow
                continue
            # Fallback: bytes → numpy → tensor/array
            data = bytes(host.read(key))[:n]
            try:
                np_dtype = np.dtype(dtype)
            except Exception:
                np_dtype = np.dtype("uint8")
            arr = np.frombuffer(bytearray(data), dtype=np_dtype)
            if shape and isinstance(shape, list):
                try:
                    arr = arr.reshape(tuple(int(x) for x in shape))
                except Exception:
                    pass
            if torch is not None:
                t = torch.from_numpy(arr) if arr.dtype != np.uint8 else torch.frombuffer(bytearray(data), dtype=torch.uint8)
                self.shadow[key] = t.clone()
            else:
                self.shadow[key] = arr.copy()

    def to_device(self, device: str = "cuda") -> None:
        """Copy shadow tensors to device asynchronously if CUDA available."""
        if torch is None:
            return
        if device.startswith("cuda") and (not torch.cuda.is_available()):
            return
        dev = torch.device(device)
        for k, t in list(self.shadow.items()):
            if isinstance(t, torch.Tensor):
                self.shadow[k] = t.to(dev, non_blocking=True)

    def bind_module(self, module, name_map: dict[str, str]) -> None:  # noqa: ANN001
        """Bind a torch.nn.Module so commit() can flip/copy parameters.

        name_map: key -> module parameter dotted path
        """
        self._module = module
        self._map = dict(name_map)

    # --- GPU path: copy from HostAgent pinned buffers into module params ---
    def apply_from_host_to_module(self, items: list[dict], host, module, name_map: dict[str, str], device: str = "cuda"):  # noqa: ANN001, ANN201
        if torch is None:
            raise RuntimeError("Torch is required for GPU in-place apply")
        self.bind_module(module, name_map)
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        dev = torch.device(device) if use_cuda else torch.device("cpu")
        stream = torch.cuda.Stream() if use_cuda else None
        with torch.cuda.stream(stream) if stream else NullContext():
            for it in items:
                key = it["key"]
                target = name_map.get(key)
                if target is None:
                    continue
                # locate param
                mod = module
                parts = target.split(".")
                for p in parts[:-1]:
                    mod = getattr(mod, p)
                pname = parts[-1]
                param = getattr(mod, pname)
                dtype_str = (it.get("dtype") or "uint8").replace("|", "")
                t_dtype = torch_dtype_from_numpy_str(dtype_str) or param.data.dtype
                shape = tuple(
                    int(x) for x in (it.get("shape") or list(param.data.shape))
                )
                # source pinned bytes
                src_bytes = getattr(host, "torch_tensor", lambda *_: None)(key)
                if src_bytes is None:
                    # Fallback: construct a typed CPU tensor via NumPy, then copy.
                    data = bytes(host.read(key))[: int(it["nbytes"]) ]
                    try:
                        np_dtype = __import__("numpy").dtype(dtype_str)  # lazy import
                    except Exception:
                        np_dtype = __import__("numpy").dtype("uint8")
                    np_arr = __import__("numpy").frombuffer(  # type: ignore[attr-defined]
                        bytearray(data), dtype=np_dtype
                    ).reshape(shape)
                    cpu_typed = torch.from_numpy(np_arr).clone()
                else:
                    # materialize typed host tensor; only pin when using CUDA
                    cpu_typed = torch.empty(
                        shape, dtype=t_dtype, pin_memory=use_cuda
                    )
                    # copy raw bytes into dst view; best-effort path
                    view_dst = cpu_typed.view(torch.uint8)
                    view_src = (
                        src_bytes
                        if src_bytes.dtype == torch.uint8
                        else src_bytes.view(torch.uint8)
                    )
                    n = int(it["nbytes"])  # expected bytes
                    view_dst[:n].copy_(view_src[:n])
                # async H2D copy into param
                if use_cuda:
                    param.data.copy_(cpu_typed.to(dev, non_blocking=True))
                else:
                    param.data.copy_(cpu_typed)
        if stream is not None:
            stream.synchronize()


class NullContext:  # pragma: no cover - simple helper if no torch.cuda.stream
    def __enter__(self) -> "NullContext":  # noqa: ANN001
        return self

    def __exit__(self, *exc: object) -> bool:  # noqa: ANN001
        return False


def _resolve_param(module, target: str):  # noqa: ANN001, ANN202
    """Resolve a dotted parameter path on a module; fail loudly if missing.

    Silent skips here are how half-applied updates happen, so a missing
    target is a hard error.
    """
    mod = module
    parts = target.split(".")
    try:
        for p in parts[:-1]:
            mod = getattr(mod, p)
        return getattr(mod, parts[-1])
    except AttributeError as exc:
        raise SwapError(f"Cannot resolve parameter {target!r} on module") from exc


def _materialize_staged_tensor(item: dict, src, param):  # noqa: ANN001, ANN202
    """Convert a staged uint8 byte tensor into one matching ``param``.

    Returns a tensor with the same shape, dtype and device as ``param.data``.
    The result may alias the agent's staging buffer; it is safe to keep alive
    via ``param.data`` (the buffer is reference-counted), but callers that
    mutate staging afterwards should clone first.
    """
    tmp = src.to(device=param.data.device, dtype=param.data.dtype, non_blocking=True)
    try:
        return tmp.reshape(tuple(param.data.shape))
    except Exception as exc:
        raise SwapError(
            f"Staged tensor for item {item.get('key')!r} has "
            f"{tmp.numel()} elements but target has {param.data.numel()}"
        ) from exc


def stage_from_ipc_agent(
    items: list[dict],
    agent,
    module,
    name_map: dict[str, str],
    device: str = "cuda",
) -> dict:  # noqa: ANN001, ANN201
    """Stage new weights from a CUDA-IPC agent WITHOUT touching the live module.

    Returns an ordered dict mapping module parameter dotted path -> new tensor
    (matching the live parameter's shape/dtype/device). Raises SwapError if
    any item has no staged data or its target parameter cannot be resolved, so
    callers can abort before mutating anything.
    """
    if torch is None:
        raise RuntimeError("Torch is required for staged weight apply")
    _ = device  # device placement follows the live parameters
    staged: dict[str, object] = {}
    for it in items:
        key = it["key"]
        target = name_map.get(key)
        if target is None:
            continue
        src = getattr(agent, "get_staged_tensor", lambda *_: None)(key)
        if src is None:
            raise SwapError(
                f"No staged tensor for plan item {key!r}; refusing partial update"
            )
        param = _resolve_param(module, target)
        staged[target] = _materialize_staged_tensor(it, src, param)
    if items and not staged:
        raise SwapError("Staging produced no tensors; refusing empty update")
    return staged


def atomic_swap_params(module, staged: dict) -> None:  # noqa: ANN001
    """Atomically flip module parameters to staged tensors.

    Each ``param.data`` pointer is replaced (not copied into), so the old
    weights stay live and untouched until the flip. Call only after the
    coordinator has accepted the commit. Raises SwapError on shape/dtype
    mismatch instead of installing a corrupt parameter.
    """
    if torch is None:
        raise RuntimeError("Torch is required for atomic weight swap")
    with torch.no_grad():
        for target, new_t in staged.items():
            param = _resolve_param(module, target)
            try:
                ok = (
                    tuple(new_t.shape) == tuple(param.data.shape)
                    and new_t.dtype == param.data.dtype
                )
            except Exception:
                ok = False
            if not ok:
                raise SwapError(
                    f"Staged tensor for {target!r} does not match live parameter "
                    f"shape/dtype; refusing to swap"
                )
            param.data = new_t
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def verify_staged_hashes(items: list[dict], read_bytes) -> None:  # noqa: ANN001
    """Verify staged tensors against plan hashes BEFORE precommit.

    ``read_bytes`` maps an item key -> raw bytes of the staged tensor.
    Raises SwapError on the first mismatch.
    """
    import hashlib

    for it in items:
        algo, expect = it["hash"].split(":", 1)
        h = hashlib.new(algo)
        h.update(read_bytes(it["key"]))
        got = h.hexdigest()
        if got != expect:
            raise SwapError(
                f"hash mismatch for {it['key']}: got {got}, expect {expect}"
            )


def read_ipc_staged_bytes(agent, key: str, nbytes: int) -> bytes:  # noqa: ANN001, ANN202
    """Read up to ``nbytes`` raw bytes from an agent's staged uint8 tensor."""
    t = getattr(agent, "get_staged_tensor", lambda *_: None)(key)
    if t is None:
        raise SwapError(f"No staged tensor for key {key!r}")
    if hasattr(t, "tobytes"):
        try:
            return bytes(t.tobytes()[:nbytes])
        except Exception:
            pass
    try:
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)
        return bytes(arr.tobytes()[:nbytes])
    except Exception as exc:
        raise SwapError(f"Cannot read staged bytes for key {key!r}") from exc


def apply_from_ipc_agent_to_module(
    items: list[dict],
    agent,
    module,
    name_map: dict[str, str],
    device: str = "cuda",
) -> None:  # noqa: ANN001, ANN201
    """Apply staged tensors from a CUDA-IPC agent into a torch.nn.Module.

    - items: list of plan items with key/shape/dtype fields
    - agent: object exposing get_staged_tensor(key) -> torch.Tensor
    - module: target torch.nn.Module
    - name_map: mapping from item key to module parameter dotted path
    - device: e.g., 'cuda' (used for stream selection)
    """
    if torch is None:
        raise RuntimeError("Torch is required for GPU in-place apply")
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    stream = torch.cuda.Stream() if use_cuda else None
    with torch.cuda.stream(stream) if stream else NullContext():
        for it in items:
            key = it["key"]
            target = name_map.get(key)
            if target is None:
                continue
            src = getattr(agent, "get_staged_tensor", lambda *_: None)(key)
            if src is None:
                continue
            # Lenient legacy path: skip unresolvable params (strict callers
            # should use stage_from_ipc_agent + atomic_swap_params instead).
            try:
                param = _resolve_param(module, target)
                new_t = _materialize_staged_tensor(it, src, param)
            except SwapError:
                continue
            param.data.copy_(new_t, non_blocking=True)
    if stream is not None:
        stream.synchronize()

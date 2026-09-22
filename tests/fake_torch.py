"""Minimal numpy-backed fake of the torch API surface used by the swap paths.

Only implements what ``hotweights.adapters.vllm_ext`` (commit/finalize) and
``hotweights.adapters.trainer_swap`` touch: tensor creation from buffers,
dtype/shape introspection, views, copies, and the no_grad/cuda/distributed
stubs. Anything else raises loudly so the fake cannot silently diverge.
"""
from __future__ import annotations

import contextlib
import types

import numpy as np


class FakeDType:
    def __init__(self, name: str, np_dtype) -> None:
        self.name = name
        self.np = np.dtype(np_dtype)
        self.itemsize = self.np.itemsize

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeDType) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


uint8 = FakeDType("uint8", np.uint8)
float32 = FakeDType("float32", np.float32)


class FakeTensor:
    requires_grad = True

    def __init__(self, arr, dtype: FakeDType | None = None) -> None:
        self._arr = np.array(arr, copy=False)
        if dtype is None:
            dtype = {np.dtype(np.uint8): uint8, np.dtype(np.float32): float32}[
                self._arr.dtype
            ]
        assert self._arr.dtype == dtype.np, (self._arr.dtype, dtype.np)
        self.dtype = dtype

    @property
    def shape(self):  # noqa: ANN201
        return tuple(self._arr.shape)

    @property
    def device(self):  # noqa: ANN201
        return "cpu"

    @property
    def data(self):  # noqa: ANN201
        return self

    def numel(self) -> int:
        return int(self._arr.size)

    def element_size(self) -> int:
        return self.dtype.itemsize

    def view(self, dtype: FakeDType):  # noqa: ANN201
        return FakeTensor(self._arr.view(dtype.np), dtype)

    def reshape(self, shape):  # noqa: ANN201
        return FakeTensor(self._arr.reshape(shape), self.dtype)

    def to(self, *args, **kwargs):  # noqa: ANN201, ANN003
        # to(device) / to(dtype) / to(device=..., dtype=...)
        dtype = kwargs.get("dtype")
        for a in args:
            if isinstance(a, FakeDType):
                dtype = a
        if dtype is not None and dtype != self.dtype:
            return FakeTensor(self._arr.astype(dtype.np), dtype)
        return self

    def detach(self):  # noqa: ANN201
        return self

    def clone(self):  # noqa: ANN201
        return FakeTensor(self._arr.copy(), self.dtype)

    def copy_(self, src: FakeTensor, **kwargs):  # noqa: ANN201, ANN003
        np.copyto(self._arr, src._arr)
        return self

    def tobytes(self) -> bytes:
        return self._arr.tobytes()

    def __getitem__(self, idx):  # noqa: ANN201
        return FakeTensor(self._arr[idx], self.dtype)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return (
            isinstance(other, FakeTensor)
            and self.dtype == other.dtype
            and self.shape == other.shape
            and bool(np.array_equal(self._arr, other._arr))
        )


def frombuffer(buf, dtype: FakeDType):  # noqa: ANN201
    return FakeTensor(np.frombuffer(bytes(buf), dtype=dtype.np).copy(), dtype)


def empty_like(t: FakeTensor, device=None, pin_memory=False):  # noqa: ANN201
    return FakeTensor(np.empty_like(t._arr), t.dtype)


def no_grad():  # noqa: ANN201
    return contextlib.nullcontext()


cuda = types.SimpleNamespace(
    is_available=lambda: False,
    synchronize=lambda: None,
)
distributed = types.SimpleNamespace(is_initialized=lambda: False)
nn = types.SimpleNamespace(Module=type("Module", (), {}))


def make_module() -> types.ModuleType:
    mod = types.ModuleType("torch")
    mod.Tensor = FakeTensor
    mod.uint8 = uint8
    mod.float32 = float32
    mod.frombuffer = frombuffer
    mod.empty_like = empty_like
    mod.no_grad = no_grad
    mod.cuda = cuda
    mod.distributed = distributed
    mod.nn = nn
    optim_mod = types.ModuleType("torch.optim")
    optim_mod.Optimizer = type("Optimizer", (), {})
    mod.optim = optim_mod
    mod.__path__ = []  # mark as package so "torch.optim" imports resolve
    return mod

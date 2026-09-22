"""Real tensor-semantics tests for the swap paths, via a numpy-backed fake torch.

``torch`` is not installed in this environment, so these tests inject a
minimal fake into ``sys.modules`` and reload the two adapter modules under
test. The fake implements exactly the torch surface the swap code touches
(tensor creation from buffers, views, copies, no_grad, cuda/distributed
stubs), so shape/dtype/device semantics are exercised for real.
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

from tests import fake_torch as ft

_MISSING = object()
_RELOAD = ["hotweights.adapters.vllm_ext", "hotweights.adapters.trainer_swap"]


@pytest.fixture()
def torch_mods():
    """Import vllm_ext/trainer_swap with the fake torch bound."""
    saved = {name: sys.modules.get(name, _MISSING) for name in ["torch", "torch.optim", *_RELOAD]}
    fake = ft.make_module()
    sys.modules["torch"] = fake
    sys.modules["torch.optim"] = fake.optim
    for name in _RELOAD:
        sys.modules.pop(name, None)
    vllm_ext = importlib.import_module("hotweights.adapters.vllm_ext")
    trainer_swap = importlib.import_module("hotweights.adapters.trainer_swap")
    yield fake, vllm_ext, trainer_swap
    for name, mod in saved.items():
        if mod is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _t(arr, dtype=None):
    arr = np.asarray(arr, dtype=np.float32 if dtype is None else dtype.np)
    return ft.FakeTensor(arr, dtype or ft.float32)


def _param(values):
    return types.SimpleNamespace(data=_t(values))


def _module():
    return types.SimpleNamespace(layer=types.SimpleNamespace(weight=_param([1.0, 2.0])))


# --- vllm_ext.HotReloadExtension.commit ------------------------------------


def test_commit_flips_all_params(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = _module()
    ext.bind_module = lambda m: setattr(ext, "_module", m)  # noqa: E731
    ext._module = mod  # noqa: SLF001
    ext._map = {"k": "layer.weight"}  # noqa: SLF001
    old = mod.layer.weight.data
    ext.shadow["k"] = _t([10.0, 20.0])  # noqa: SLF001
    ext.commit("v2")
    assert mod.layer.weight.data is not old
    assert mod.layer.weight.data == _t([10.0, 20.0])


def test_commit_bad_name_map_raises_and_leaves_module_untouched(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = _module()
    ext._module = mod  # noqa: SLF001
    ext._map = {"k": "layer.nonexistent"}  # noqa: SLF001
    old = mod.layer.weight.data
    ext.shadow["k"] = _t([10.0, 20.0])  # noqa: SLF001
    with pytest.raises(vllm_ext.SwapError):
        ext.commit("v2")
    assert mod.layer.weight.data is old


def test_commit_shape_mismatch_raises_and_leaves_module_untouched(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = _module()
    ext._module = mod  # noqa: SLF001
    ext._map = {"k": "layer.weight"}  # noqa: SLF001
    old = mod.layer.weight.data
    ext.shadow["k"] = _t([10.0, 20.0, 30.0])  # noqa: SLF001
    with pytest.raises(vllm_ext.SwapError):
        ext.commit("v2")
    assert mod.layer.weight.data is old


def test_commit_partial_failure_is_all_or_nothing(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = types.SimpleNamespace(
        a=types.SimpleNamespace(w=_param([1.0])),
        b=types.SimpleNamespace(w=_param([2.0])),
    )
    ext._module = mod  # noqa: SLF001
    ext._map = {"ka": "a.w", "kb": "b.w"}  # noqa: SLF001
    old_a, old_b = mod.a.w.data, mod.b.w.data
    ext.shadow["ka"] = _t([100.0])  # noqa: SLF001
    ext.shadow["kb"] = _t([200.0, 300.0])  # noqa: SLF001  # bad: shape mismatch
    with pytest.raises(vllm_ext.SwapError):
        ext.commit("v2")
    # The first target must NOT have flipped either.
    assert mod.a.w.data is old_a
    assert mod.b.w.data is old_b


def test_commit_reinterprets_raw_staged_bytes(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = _module()
    ext._module = mod  # noqa: SLF001
    ext._map = {"k": "layer.weight"}  # noqa: SLF001
    raw = np.array([10.0, 20.0], dtype=np.float32).tobytes()
    ext.shadow["k"] = ft.FakeTensor(  # noqa: SLF001
        np.frombuffer(raw, dtype=np.uint8).copy(), ft.uint8
    )
    ext.commit("v2")
    assert mod.layer.weight.data.dtype == ft.float32
    assert mod.layer.weight.data == _t([10.0, 20.0])


def test_finalize_shard_never_touches_live_params(torch_mods):
    _, vllm_ext, _ = torch_mods
    ext = vllm_ext.HotReloadExtension()
    mod = _module()
    ext._module = mod  # noqa: SLF001
    ext._map = {"w:0": "layer.weight"}  # noqa: SLF001
    old = mod.layer.weight.data
    payload = np.array([7.0, 8.0], dtype=np.float32).tobytes()
    ext._host_buffers["w:0"] = memoryview(bytearray(payload))  # noqa: SLF001
    ext._pinned["w:0"] = ft.FakeTensor(  # noqa: SLF001
        np.frombuffer(payload, dtype=np.uint8).copy(), ft.uint8
    )
    ext.finalize_shard("w", 0, "deadbeef")
    assert mod.layer.weight.data is old
    staged = ext.shadow["w:0"]  # noqa: SLF001
    assert staged.dtype == ft.float32
    assert staged == _t([7.0, 8.0])


# --- trainer_swap -----------------------------------------------------------


def _fake_model():
    table = {"a": _param([1.0, 2.0]), "b": _param([3.0, 4.0])}

    class Model:
        def get_parameter(self, name):
            return table[name].data

        def named_parameters(self):
            return [(k, v.data) for k, v in table.items()]

    return Model(), table


def _fake_optimizer():
    return types.SimpleNamespace(param_groups=[], state={})


def test_trainer_swap_success(torch_mods):
    _, _, trainer_swap = torch_mods
    model, table = _fake_model()
    staged = {"ka": _t([10.0, 20.0]), "kb": _t([30.0, 40.0])}
    trainer_swap.sota_in_place_swap(
        model, _fake_optimizer(), staged, {}, {"ka": "a", "kb": "b"}
    )
    assert table["a"].data == _t([10.0, 20.0])
    assert table["b"].data == _t([30.0, 40.0])


def test_trainer_swap_shape_mismatch_raises_before_mutating(torch_mods):
    _, _, trainer_swap = torch_mods
    model, table = _fake_model()
    old_a = table["a"].data
    staged = {"ka": _t([10.0, 20.0]), "kb": _t([30.0])}  # kb bad shape
    with pytest.raises(RuntimeError, match="aborted"):
        trainer_swap.sota_in_place_swap(
            model, _fake_optimizer(), staged, {}, {"ka": "a", "kb": "b"}
        )
    # Nothing was mutated, so a later optimizer sync cannot mix versions.
    assert table["a"].data is old_a


def test_trainer_swap_unresolvable_param_raises(torch_mods):
    _, _, trainer_swap = torch_mods
    model, table = _fake_model()
    with pytest.raises(RuntimeError, match="aborted"):
        trainer_swap.sota_in_place_swap(
            model, _fake_optimizer(), {"kz": _t([1.0])}, {}, {"kz": "missing"}
        )

"""Toy trainer swap demo (CPU-only).

Shows how to call in_place_swap at a barrier; this is a skeleton and not a
real training loop.
"""
from __future__ import annotations

from hotweights.adapters.trainer_swap import swap_barrier, in_place_swap


def fake_shard_iter():  # noqa: ANN201 - demo stub
    import numpy as np

    class Param:
        def __init__(self, arr):
            self.data = arr
            self.shape = arr.shape

    p = Param(np.zeros((2,), dtype=np.float32))
    new = np.ones((2,), dtype=np.float32)

    def gen():
        yield p, new

    return gen


class Optim:
    def zero_grad(self, set_to_none=True):  # noqa: ANN001
        pass


if __name__ == "__main__":
    with swap_barrier():
        in_place_swap(fake_shard_iter(), Optim())
    print("swap done")


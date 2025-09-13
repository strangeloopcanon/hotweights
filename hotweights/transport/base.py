"""Transport base interfaces."""
from __future__ import annotations

from typing import Protocol, Iterable, Callable, Tuple, List
import numpy as np


Bucket = Tuple[int, np.ndarray]


class Transport(Protocol):
    """Protocol for broadcast transports."""

    def replicate(self, bucket_iter: Iterable[Bucket]) -> None:
        ...

    def replicate_stream(
        self,
        bucket_iter: Iterable[Tuple[int, np.ndarray] | Tuple[int, np.ndarray, List[int]]],
        on_complete: Callable[[int, np.ndarray], None],
    ) -> None:
        ...


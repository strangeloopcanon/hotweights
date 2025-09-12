"""Configuration helpers for hotweights.

Currently minimal; expand as features land.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HashAlgo:
    name: str = "sha256"
    block_size: int = 1024 * 1024


DEFAULT_HASH = HashAlgo()


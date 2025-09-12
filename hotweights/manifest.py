"""Manifest schema and utilities.

A manifest describes the tensors/shards for a checkpoint version.
For the MVP, we treat each file under a checkpoint directory as a shard.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TypedDict
import hashlib
import json
import os

from .config import DEFAULT_HASH
import numpy as np


class Shard(TypedDict):
    rank: int
    bytes: int
    hash: str
    uri: str


class TensorEntry(TypedDict):
    name: str
    dtype: Optional[str]
    shape: Optional[List[int]]
    partitioning: Optional[Dict[str, int | str]]
    quant: Optional[Dict[str, object]]
    shards: List[Shard]


class Manifest(TypedDict):
    model_id: str
    version: str
    tensors: List[TensorEntry]


def _file_hash(path: Path, algo: str = DEFAULT_HASH.name, block_size: int = DEFAULT_HASH.block_size) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def _iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def build_simple_manifest(model_id: str, version: str, checkpoint_dir: os.PathLike[str] | str) -> Manifest:
    """Build a simple manifest using files as shards.

    - name: relative path to file
    - shard_rank: 0 (single shard per file in MVP)
    """
    root = Path(checkpoint_dir).resolve()
    tensors: List[TensorEntry] = []
    for f in sorted(_iter_files(root)):
        rel = f.relative_to(root).as_posix()
        dtype: Optional[str] = None
        shape: Optional[List[int]] = None
        if f.suffix == ".npy":  # infer dtype/shape for numpy arrays
            try:
                arr = np.load(f, mmap_mode="r")
                dtype = str(arr.dtype)
                shape = [int(x) for x in arr.shape]
            except Exception:
                dtype, shape = None, None

        tensors.append(
            TensorEntry(
                name=rel,
                dtype=dtype,
                shape=shape,
                partitioning=None,
                quant=None,
                shards=[
                    Shard(
                        rank=0,
                        bytes=f.stat().st_size,
                        hash=_file_hash(f),
                        uri=f.as_uri(),
                    )
                ],
            )
        )
    return Manifest(model_id=model_id, version=version, tensors=tensors)


def to_dataframe(manifest: Manifest):
    """Convert manifest tensors to a pandas DataFrame for planning.

    Columns: [tensor, shard_rank, nbytes, hash, path]
    """
    import pandas as pd

    rows = []
    for t in manifest["tensors"]:
        for s in t["shards"]:
            rows.append(
                {
                    "tensor": t["name"],
                    "shard_rank": s["rank"],
                    "nbytes": s["bytes"],
                    "hash": s["hash"],
                    "path": t["name"],
                    "dtype": t.get("dtype"),
                    "shape": t.get("shape"),
                }
            )
    return pd.DataFrame(rows)


def dump_manifest(manifest: Manifest, path: os.PathLike[str] | str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: os.PathLike[str] | str) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # type: ignore[return-value]

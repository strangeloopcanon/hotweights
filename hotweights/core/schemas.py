"""Shared schema types for manifests, plans, and items.

Define stable TypedDicts for interchange between components.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Literal

try:  # pragma: no cover - import shim for pydantic compatibility
    from typing_extensions import TypedDict
except ImportError:  # pragma: no cover - fallback for Python >=3.12
    from typing import TypedDict  # type: ignore[misc, assignment]


# Versioning for plan schema produced by create_plan
PLAN_SCHEMA_VERSION: str = "1"


class Shard(TypedDict):
    rank: int
    bytes: int
    hash: str
    uri: str


class TensorEntry(TypedDict, total=False):
    name: str
    dtype: Optional[str]
    shape: Optional[List[int]]
    partitioning: Dict[str, int | str]
    quant: Dict[str, object]
    shards: List[Shard]


class Manifest(TypedDict):
    model_id: str
    version: str
    tensors: List[TensorEntry]


class PlanItem(TypedDict, total=False):
    tensor: str
    shard_rank: int
    nbytes: int
    hash: str
    uri: str
    dtype: Optional[str]
    shape: Optional[List[int]]
    key: str
    offset: int


class PlanBucket(TypedDict, total=False):
    bucket_id: int
    size: int
    items: List[PlanItem]
    consumer_ranks: List[int]


class Plan(TypedDict, total=False):
    plan_version: str
    version: str  # next version
    bucket_bytes: int
    total_bytes: int
    buckets: List[PlanBucket]
    verification: Dict[str, object]

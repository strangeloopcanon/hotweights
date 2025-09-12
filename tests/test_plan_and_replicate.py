from __future__ import annotations

import json
from pathlib import Path

from hotweights.manifest import build_simple_manifest, dump_manifest
from hotweights.core.replicate import create_plan as _create_plan
from hotweights.core.replicate import assemble_bucket as _assemble_bucket
from hotweights.core.replicate import scatter_bucket as _scatter_bucket
from hotweights.core.replicate import verify_items as _verify_items
from hotweights.staging.host_agent import HostAgent


def _write_ckpt(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_plan_and_local_replicate(tmp_path: Path) -> None:
    # prev checkpoint
    ckpt_prev = tmp_path / "ckpt_prev"
    _write_ckpt(ckpt_prev, {"a.bin": b"a" * 4, "dir/b.bin": b"b" * 3})
    m_prev = build_simple_manifest("toy", "v0", ckpt_prev)
    dump_manifest(m_prev, tmp_path / "prev.json")

    # next checkpoint (one changed, one new)
    ckpt_next = tmp_path / "ckpt_next"
    _write_ckpt(ckpt_next, {"a.bin": b"a" * 4, "dir/b.bin": b"B" * 5, "c.bin": b"c" * 2})
    m_next = build_simple_manifest("toy", "v1", ckpt_next)
    dump_manifest(m_next, tmp_path / "next.json")

    plan = _create_plan(m_prev, m_next, bucket_mb=1)
    # Expect b.bin (changed) and c.bin (added); a.bin unchanged
    items = [i for b in plan["buckets"] for i in b["items"]]
    names = sorted(set(i["tensor"] for i in items))
    assert names == ["c.bin", "dir/b.bin"]

    # replicate locally
    host = HostAgent()
    for b in plan["buckets"]:
        buf = _assemble_bucket(b["items"])
        _scatter_bucket(host, b["items"], buf)
        _verify_items(host, b["items"])  # raises on mismatch
        # dtype/shape fields should exist even if None except for .npy detection path
        for it in b["items"]:
            assert "dtype" in it and "shape" in it

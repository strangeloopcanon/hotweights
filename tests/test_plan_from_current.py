from __future__ import annotations

from pathlib import Path

from hotweights.manifest import build_simple_manifest
from hotweights.core.replicate import create_plan_from_current


def _write_ckpt(root: Path, files: dict[str, bytes]) -> None:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def test_create_plan_from_current(tmp_path: Path) -> None:
    ckpt_prev = tmp_path / "prev"
    _write_ckpt(ckpt_prev, {"a.bin": b"a" * 4, "b.bin": b"b" * 2})
    prev = build_simple_manifest("toy", "v0", ckpt_prev)

    ckpt_next = tmp_path / "next"
    _write_ckpt(ckpt_next, {"a.bin": b"a" * 4, "b.bin": b"B" * 3, "c.bin": b"c" * 2})
    nxt = build_simple_manifest("toy", "v1", ckpt_next)

    plan = create_plan_from_current(nxt, current_provider=lambda: prev, bucket_mb=1)

    items = [i for b in plan["buckets"] for i in b["items"]]
    names = sorted(set(i["tensor"] for i in items))
    # a.bin unchanged; expect b.bin changed and c.bin new
    assert names == ["b.bin", "c.bin"]


from __future__ import annotations

import json
from pathlib import Path

from hotweights.manifest import build_simple_manifest, to_dataframe


def test_build_manifest_and_df(tmp_path: Path) -> None:
    # create dummy checkpoint files
    (tmp_path / "layer0").mkdir()
    f1 = tmp_path / "layer0" / "w.bin"
    f1.write_bytes(b"a" * 10)
    f2 = tmp_path / "emb.bin"
    f2.write_bytes(b"b" * 5)

    m = build_simple_manifest("toy", "v1", tmp_path)
    assert m["model_id"] == "toy"
    assert m["version"] == "v1"
    names = sorted(t["name"] for t in m["tensors"])
    assert names == ["emb.bin", "layer0/w.bin"]

    df = to_dataframe(m)
    assert {"tensor", "shard_rank", "nbytes", "hash", "path"}.issubset(set(df.columns))
    assert int(df["nbytes"].sum()) == 15
    assert df.shape[0] == 2

from __future__ import annotations

from hotweights.adapters.vllm_ext import HotReloadExtension
from hotweights.staging.host_agent import HostAgent


def test_vllm_adapter_ingest_and_commit():
    host = HostAgent()
    # stage two items
    items = [
        {"key": "a:0", "nbytes": 4},
        {"key": "b:0", "nbytes": 3},
    ]
    host.write("a:0", 0, memoryview(b"abcd"))
    host.write("b:0", 0, memoryview(b"xyz"))
    host.seal("a:0"); host.seal("b:0")

    ext = HotReloadExtension()
    ext.begin_update("v1", {})
    ext.ingest_from_host(items, host)
    assert set(ext.shadow.keys()) == {"a:0", "b:0"}
    ext.precommit(); ext.commit("v1")
    assert ext.params["a:0"].tolist() == list(b"abcd")
    assert ext.version == "v1"


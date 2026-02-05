from __future__ import annotations

from hotweights.cli import _derive_pub_endpoint as cli_pub_endpoint
from hotweights.coordinator.zmq_server import _derive_pub_endpoint as server_pub_endpoint
from hotweights.worker.agent import _derive_pub_endpoint as worker_pub_endpoint


def test_derive_pub_endpoint_from_tcp_rep_endpoint() -> None:
    rep = "tcp://10.1.2.3:7000"
    expected = "tcp://10.1.2.3:7001"
    assert cli_pub_endpoint(rep) == expected
    assert worker_pub_endpoint(rep) == expected
    assert server_pub_endpoint(rep) == expected

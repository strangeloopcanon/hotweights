from __future__ import annotations

from hotweights.transport.donor_select import order_donors


def test_order_donors_prefers_same_host_then_prefix():
    workers = {
        "w0": {"rank": 0, "host": "h1", "addr": "10.0.1.2", "port": 9000},
        "w1": {"rank": 1, "host": "h2", "addr": "10.0.1.3", "port": 9001},
        "w2": {"rank": 2, "host": "h3", "addr": "10.0.2.4", "port": 9002},
    }
    ranks = [0, 1, 2]
    self_meta = {"host": "h1", "addr": "10.0.1.10"}
    ordered = order_donors(workers, ranks, self_meta)
    ids = [wid for wid, _ in ordered]
    assert ids[0] == "w0"  # same host
    assert set(ids[1:]) == {"w1", "w2"}


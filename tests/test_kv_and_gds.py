from __future__ import annotations

import pytest

from hotweights.transport.gds_storage import GDSDirectTransfer


def test_gds_direct_transfer_init_does_not_raise_unboundlocal() -> None:
    tr = GDSDirectTransfer()
    assert isinstance(tr.is_available, bool)


def test_kv_migration_handles_models_without_device_attr() -> None:
    torch = pytest.importorskip("torch")
    from hotweights.adapters.kv_cache_migration import migrate_kv_cache

    model = torch.nn.Linear(2, 2)
    kv_old = [(torch.zeros(2, 1, 1), torch.zeros(2, 1, 1))]
    kv_new, report = migrate_kv_cache(kv_old, model, {"buckets": []})
    assert len(kv_new) == 1
    assert report["migrated_layers"] == 1

from __future__ import annotations

from hotweights.transport.transport_manager import TransportManager
from hotweights.transport.ucx_stream import UCXReplicator
from hotweights.transport.mpi_stream import MPIReplicator


def test_transport_manager_preference_ucx(monkeypatch):
    # Force discovery to believe both UCX (rdma) and MPI are available
    monkeypatch.setattr(TransportManager, "_check_rdma_availability", lambda self: True)
    monkeypatch.setattr(TransportManager, "_check_mpi_availability", lambda self: True)
    # Build manager with a preference for UCX
    tm = TransportManager(world_size=2, rank=0, auto_select=True, preferred_transport="ucx")
    rep = tm.get_replicator()
    assert isinstance(rep, UCXReplicator)


def test_transport_manager_preference_mpi(monkeypatch):
    monkeypatch.setattr(TransportManager, "_check_rdma_availability", lambda self: True)
    monkeypatch.setattr(TransportManager, "_check_mpi_availability", lambda self: True)
    tm = TransportManager(world_size=2, rank=0, auto_select=True, preferred_transport="mpi")
    rep = tm.get_replicator()
    assert isinstance(rep, MPIReplicator)


"""Transport subpackage.

Provides multiple transport backends:
- MPI: Standard MPI broadcast using mpi4py
- UCX: High-performance UCX transport for RDMA
- CUDA-IPC: Zero-copy GPU memory sharing with traffic shaping
"""

from .mpi_stream import MPIReplicator
from .ucx_stream import UCXReplicator
from .cuda_ipc import CudaIPCTransport
from .topology_scheduler import TopologyDiscovery, TopologyAwareScheduler

__all__ = [
    "MPIReplicator",
    "UCXReplicator", 
    "CudaIPCTransport",
    "TopologyDiscovery",
    "TopologyAwareScheduler",
]

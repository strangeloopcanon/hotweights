"""Unified transport manager with automatic backend selection.

Provides intelligent transport selection based on:
1. Hardware capabilities (CUDA, RDMA, NVLink)
2. Scale (node-local vs cluster-wide)
3. Performance requirements
4. Fallback mechanisms
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Union
import time

try:  # optional
    import torch
except Exception:
    torch = None

try:
    from mpi4py import MPI
except Exception:
    MPI = None

try:
    import ucp
except Exception:
    ucp = None

try:
    from .nccl_stream import NCCLReplicator
except Exception:
    NCCLReplicator = None  # type: ignore
from .mpi_stream import MPIReplicator
from .ucx_stream import UCXReplicator
from .cuda_ipc import CudaIPCTransport
from .topology_scheduler import TopologyDiscovery, TopologyAwareScheduler


logger = logging.getLogger(__name__)


@dataclass
class TransportCapabilities:
    """Capabilities of a transport backend."""
    name: str
    supports_cuda: bool
    supports_rdma: bool
    supports_nvlink: bool
    max_bandwidth_gbps: float
    latency_us: float
    scalability: str  # 'node', 'rack', 'cluster', 'wan'
    reliability: float  # 0-1
    setup_overhead: float  # seconds


@dataclass
class TransportSelection:
    """Transport selection with rationale."""
    transport_class: type
    capabilities: TransportCapabilities
    rationale: str
    fallback_order: list[type]


class TransportManager:
    """Manages transport backend selection and lifecycle."""
    
    # Define transport capabilities
    TRANSPORT_CAPS = {
        'nccl': TransportCapabilities(
            name='NCCL',
            supports_cuda=True,
            supports_rdma=True,
            supports_nvlink=True,
            max_bandwidth_gbps=250.0,
            latency_us=5.0,
            scalability='cluster',
            reliability=0.92,
            setup_overhead=0.5
        ),
        'cuda_ipc': TransportCapabilities(
            name='CUDA-IPC',
            supports_cuda=True,
            supports_rdma=False,
            supports_nvlink=True,
            max_bandwidth_gbps=300.0,
            latency_us=1.0,
            scalability='node',
            reliability=0.95,
            setup_overhead=0.1
        ),
        'ucx': TransportCapabilities(
            name='UCX',
            supports_cuda=True,
            supports_rdma=True,
            supports_nvlink=False,
            max_bandwidth_gbps=200.0,
            latency_us=5.0,
            scalability='cluster',
            reliability=0.90,
            setup_overhead=1.0
        ),
        'mpi': TransportCapabilities(
            name='MPI',
            supports_cuda=False,
            supports_rdma=False,
            supports_nvlink=False,
            max_bandwidth_gbps=25.0,
            latency_us=50.0,
            scalability='cluster',
            reliability=0.99,
            setup_overhead=0.01
        )
    }
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        auto_select: bool = True,
        preferred_transport: Optional[str] = None,
    ) -> None:
        self.world_size = world_size
        self.rank = rank
        self.auto_select = auto_select
        self.preferred_transport = preferred_transport
        
        # Topology discovery
        self.topology = TopologyDiscovery()
        self.scheduler = TopologyAwareScheduler(self.topology)
        
        # Transport instances
        self._transports: dict[str, Union[MPIReplicator, UCXReplicator, CudaIPCTransport]] = {}
        self._active_transport: Optional[str] = None
        
        # Performance monitoring
        self._transport_performance: dict[str, list[float]] = {}
        self._selection_history: list[TransportSelection] = []
        
        # Initialize if auto-selection is enabled
        if auto_select:
            self._discover_and_initialize_transports()
    
    def _discover_and_initialize_transports(self) -> None:
        """Discover available transports and initialize them."""
        logger.info(f"Discovering transports for rank {self.rank}/{self.world_size}")
        
        # Discover CUDA capabilities
        cuda_available = self._check_cuda_availability()
        nvlink_available = self._check_nvlink_availability() if cuda_available else False
        
        # Discover RDMA capabilities  
        rdma_available = self._check_rdma_availability()
        
        # Discover MPI availability
        mpi_available = self._check_mpi_availability()
        
        logger.info(f"Transport discovery: CUDA={cuda_available}, NVLink={nvlink_available}, RDMA={rdma_available}, MPI={mpi_available}")
        
        # Initialize available transports
        # NCCL (inter-node GPU broadcast)
        if self._check_nccl_availability():
            try:
                if NCCLReplicator is not None:
                    self._transports['nccl'] = NCCLReplicator()
                    logger.info("NCCL transport initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize NCCL: {e}")
        # Note: CUDA-IPC requires agent/metrics; construct explicitly where available (CLI/worker).
        # Avoid registering a placeholder class here to prevent selection of a non-instantiated transport.
        
        if rdma_available:
            try:
                self._transports['ucx'] = UCXReplicator()
                logger.info("UCX transport initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize UCX: {e}")
        
        if mpi_available:
            try:
                self._transports['mpi'] = MPIReplicator()
                logger.info("MPI transport initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MPI: {e}")
        
        # Select best transport
        if self._transports:
            self._select_optimal_transport()
    
    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        if not torch:
            return False
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    
    def _check_nvlink_availability(self) -> bool:
        """Check if NVLink is available between GPUs."""
        if not torch or torch.cuda.device_count() < 2:
            return False
        
        try:
            # Test P2P access between GPUs
            can_access = torch.cuda.can_device_access_peer(0, 1)
            return can_access
        except:
            return False
    
    def _check_rdma_availability(self) -> bool:
        """Check if RDMA/UCX is available."""
        if not ucp:
            return False
        
        try:
            # Simple UCX availability check
            return hasattr(ucp, 'init')
        except:
            return False
    
    def _check_mpi_availability(self) -> bool:
        """Check if MPI is available."""
        if not MPI:
            return False
        
        try:
            comm = MPI.COMM_WORLD
            return comm.Get_size() > 1
        except:
            return False

    def _check_nccl_availability(self) -> bool:
        """Check if NCCL is likely available via torch.distributed."""
        try:
            import torch
            import torch.distributed as dist  # noqa: F401
        except Exception:
            return False
        if not torch.cuda.is_available():
            return False
        # If world size is 1, skip NCCL
        try:
            from ..utils.env import env_int
            ws = env_int('WORLD_SIZE', 1, minimum=1)
        except Exception:
            ws = 1
        return ws > 1
    
    def _select_optimal_transport(self) -> TransportSelection:
        """Select the optimal transport based on capabilities and requirements."""
        
        # If user specified a preference, try to honor it
        if self.preferred_transport and self.preferred_transport in self._transports:
            selection = TransportSelection(
                transport_class=type(self._transports[self.preferred_transport]),
                capabilities=self.TRANSPORT_CAPS[self.preferred_transport],
                rationale=f"User preferred: {self.preferred_transport}",
                fallback_order=self._get_fallback_order(self.preferred_transport)
            )
            self._active_transport = self.preferred_transport
            return selection
        
        # Score each available transport
        best_score = -1
        best_transport = None
        best_rationale = ""
        
        for transport_name, transport_instance in self._transports.items():
            score, rationale = self._score_transport(transport_name)
            
            if score > best_score:
                best_score = score
                best_transport = transport_name
                best_rationale = rationale
        
        if best_transport:
            selection = TransportSelection(
                transport_class=type(self._transports[best_transport]),
                capabilities=self.TRANSPORT_CAPS[best_transport],
                rationale=best_rationale,
                fallback_order=self._get_fallback_order(best_transport)
            )
            self._active_transport = best_transport
            self._selection_history.append(selection)
            
            logger.info(f"Selected {best_transport} transport: {best_rationale}")
            return selection
        
        raise RuntimeError("No suitable transport available")
    
    def _score_transport(self, transport_name: str) -> tuple[float, str]:
        """Score a transport based on current conditions."""
        caps = self.TRANSPORT_CAPS[transport_name]
        score = 0.0
        rationale_parts = []
        
        # Scale appropriateness
        if self.world_size <= 8 and caps.scalability in ['node']:
            score += 30
            rationale_parts.append("node-local optimal")
        elif self.world_size <= 64 and caps.scalability in ['node', 'rack']:
            score += 20
            rationale_parts.append("rack-scale suitable")
        elif caps.scalability in ['cluster', 'wan']:
            score += 10
            rationale_parts.append("cluster suitable")
        
        # Performance characteristics
        score += min(caps.max_bandwidth_gbps / 10, 20)  # Up to 20 points for bandwidth
        score += min(10.0 / caps.latency_us, 10)  # Up to 10 points for low latency
        score += caps.reliability * 10  # Up to 10 points for reliability
        
        rationale_parts.extend(
            [
                f"{caps.max_bandwidth_gbps}Gbps",
                f"{caps.latency_us}us latency",
                f"{caps.reliability:.2f} reliability",
            ]
        )
        
        # Hardware compatibility
        if caps.supports_cuda and torch and torch.cuda.is_available():
            score += 10
            rationale_parts.append("CUDA optimized")
        
        if caps.supports_nvlink and self._check_nvlink_availability():
            score += 15
            rationale_parts.append("NVLink capable")
        
        if caps.supports_rdma and self._check_rdma_availability():
            score += 10
            rationale_parts.append("RDMA capable")
        
        return score, ", ".join(rationale_parts)
    
    def _get_fallback_order(self, primary_transport: str) -> list[type]:
        """Get fallback order for transport selection."""
        fallback_map = {
            'nccl': [UCXReplicator, MPIReplicator],
            'cuda_ipc': [UCXReplicator, MPIReplicator],
            'ucx': [CudaIPCTransport, MPIReplicator],
            'mpi': [UCXReplicator, CudaIPCTransport]
        }
        
        # Filter to only available transports
        available_fallbacks: list[type] = []
        for transport_class in fallback_map.get(primary_transport, []):
            transport_name = self._get_transport_name(transport_class)
            if transport_name in self._transports:
                available_fallbacks.append(transport_class)
        
        return available_fallbacks
    
    def _get_transport_name(self, transport_class: type) -> str:
        """Get transport name from class."""
        name_map = {
            NCCLReplicator: 'nccl' if NCCLReplicator is not None else 'unknown',
            CudaIPCTransport: 'cuda_ipc',
            UCXReplicator: 'ucx',
            MPIReplicator: 'mpi'
        }
        return name_map.get(transport_class, 'unknown')
    
    def get_replicator(self) -> Union[MPIReplicator, UCXReplicator, CudaIPCTransport]:
        """Get the currently selected replicator."""
        if not self._active_transport:
            self._select_optimal_transport()

        if self._active_transport not in self._transports:
            raise RuntimeError(f"Active transport {self._active_transport} not available")

        return self._transports[self._active_transport]
    
    def replicate(self, bucket_iter: object) -> None:
        """Replicate using the selected transport."""
        replicator = self.get_replicator()
        
        # Track performance
        start_time = time.time()
        
        try:
            replicator.replicate(bucket_iter)
            
            # Record performance
            duration = time.time() - start_time
            transport_name = self._active_transport
            
            if transport_name not in self._transport_performance:
                self._transport_performance[transport_name] = []
            self._transport_performance[transport_name].append(duration)
            
            # Keep only recent history
            if len(self._transport_performance[transport_name]) > 100:
                self._transport_performance[transport_name].pop(0)
                
        except Exception as e:
            logger.error(f"Transport {self._active_transport} failed: {e}")
            
            # Try fallback transports
            selection = self._selection_history[-1] if self._selection_history else None
            if selection and selection.fallback_order:
                for fallback_class in selection.fallback_order:
                    try:
                        fallback_name = self._get_transport_name(fallback_class)
                        if fallback_name in self._transports:
                            logger.info(f"Trying fallback transport: {fallback_name}")
                            self._active_transport = fallback_name
                            self._transports[fallback_name].replicate(bucket_iter)
                            return
                    except Exception as fallback_e:
                        logger.error(
                            f"Fallback transport {fallback_name} also failed: {fallback_e}"
                        )
                        continue
            
            # All transports failed
            raise RuntimeError(f"All transports failed for replication: {e}")
    
    def get_performance_report(self) -> dict[str, Any]:
        """Get performance report for all transports."""
        report: dict[str, Any] = {
            'active_transport': self._active_transport,
            'available_transports': list(self._transports.keys()),
            'transport_performance': {},
            'selection_history': [],
        }
        
        # Transport performance
        for transport_name, durations in self._transport_performance.items():
            if durations:
                report['transport_performance'][transport_name] = {
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'count': len(durations),
                }
        
        # Selection history
        for selection in self._selection_history:
            report['selection_history'].append(
                {
                    'transport': selection.capabilities.name,
                    'rationale': selection.rationale,
                    'timestamp': time.time(),
                }
            )
        
        return report
    
    def __enter__(self) -> "TransportManager":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        # Cleanup transports
        for transport in self._transports.values():
            if hasattr(transport, 'cleanup'):
                try:
                    transport.cleanup()  # type: ignore[misc]
                except Exception:
                    pass

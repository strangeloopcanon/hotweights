"""Topology-aware scheduling for CUDA-IPC traffic shaping.

Implements:
1. GPU topology discovery (PCIe switches, NVLink domains, NUMA affinity)
2. Bandwidth-aware lane assignment
3. Congestion-aware routing
4. Dynamic load balancing across multiple paths
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Callable
import weakref

import numpy as np

try:  # optional
    import torch
    import pynvml  # NVIDIA Management Library
except Exception:
    torch = None
    pynvml = None

from ..telemetry.metrics import Timer
from ..telemetry.prom import Gauge, Counter


@dataclass
class GpuTopology:
    """Complete GPU topology information."""
    gpu_id: int
    rank: int
    pci_bus_id: str
    numa_node: int
    pcie_switch: str
    nvlink_domain: str
    memory_bandwidth_gbps: float
    pci_bandwidth_gbps: float
    nvlink_bandwidth_gbps: float
    latency_us: float
    
    # Connection topology
    direct_nvlink_gpus: Set[int] = field(default_factory=set)
    same_pcie_switch_gpus: Set[int] = field(default_factory=set)
    same_numa_node_gpus: Set[int] = field(default_factory=set)
    
    # Performance characteristics
    max_p2p_bandwidth_gbps: float = 0.0
    optimal_chunk_size: int = 64 * 1024 * 1024  # 64MB default


@dataclass
class NetworkPath:
    """A network path between GPUs with performance characteristics."""
    src_rank: int
    dst_rank: int
    bandwidth_gbps: float
    latency_us: float
    hop_count: int
    path_type: str  # 'nvlink', 'pcie', 'infiniband', 'ethernet'
    congestion_score: float = 0.0
    reliability_score: float = 1.0
    
    def effective_bandwidth(self) -> float:
        """Calculate effective bandwidth considering congestion."""
        return self.bandwidth_gbps * (1.0 - min(self.congestion_score, 0.9))


@dataclass
class TransferSchedule:
    """Optimized transfer schedule for a bucket."""
    bucket_id: int
    chunk_assignments: Dict[int, List[NetworkPath]]  # chunk_id -> paths
    expected_duration: float
    total_bandwidth: float
    congestion_risk: float


class TopologyDiscovery:
    """Discovers GPU and network topology."""
    
    def __init__(self):
        self.gpus: Dict[int, GpuTopology] = {}
        self.paths: Dict[Tuple[int, int], List[NetworkPath]] = {}
        self.node_topology = self._discover_node_topology()
        
    def _discover_node_topology(self) -> Dict[str, any]:
        """Discover node-level topology (NUMA, PCIe)."""
        topology = {}
        
        # Try to read from nvidia-ml-py first
        if pynvml:
            try:
                pynvml.nvmlInit()
                topology['nvml_available'] = True
                topology['gpu_count'] = pynvml.nvmlDeviceGetCount()
            except:
                topology['nvml_available'] = False
        else:
            topology['nvml_available'] = False
            
        # Fallback to lspci and /sys discovery
        try:
            # Discover PCIe topology
            result = subprocess.run(['lspci', '-tv'], capture_output=True, text=True)
            topology['pcie_topology'] = result.stdout
        except:
            topology['pcie_topology'] = None
            
        # Discover NUMA topology
        try:
            with open('/sys/devices/system/node/online', 'r') as f:
                topology['numa_nodes'] = f.read().strip()
        except:
            topology['numa_nodes'] = '0-0'  # Default single node
            
        return topology

    def discover_gpu_topology(self, rank: int, gpu_id: int) -> GpuTopology:
        """Discover topology for a specific GPU."""
        if pynvml:
            return self._discover_gpu_topology_nvml(rank, gpu_id)
        else:
            return self._discover_gpu_topology_fallback(rank, gpu_id)
    
    def _discover_gpu_topology_nvml(self, rank: int, gpu_id: int) -> GpuTopology:
        """Discover GPU topology using NVML."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        
        # Get PCI info
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        pci_bus_id = pci_info.busId.decode('utf-8')
        
        # Get memory bandwidth
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # Estimate bandwidth based on GPU generation (simplified)
        memory_bandwidth = self._estimate_memory_bandwidth(gpu_id)
        
        # Discover NUMA affinity
        numa_node = self._get_numa_node_for_pcie_bus(pci_bus_id)
        
        # Discover PCIe switch
        pcie_switch = self._get_pcie_switch_for_bus(pci_bus_id)
        
        # Discover NVLink connections
        nvlink_gpus = self._discover_nvlink_connections(gpu_id)
        
        # Calculate bandwidths
        pci_bandwidth = 32.0  # PCIe Gen4 x16
        nvlink_bandwidth = 300.0 if nvlink_gpus else 0.0  # NVLink 3.0
        
        gpu_topo = GpuTopology(
            gpu_id=gpu_id,
            rank=rank,
            pci_bus_id=pci_bus_id,
            numa_node=numa_node,
            pcie_switch=pcie_switch,
            nvlink_domain=f"domain_{gpu_id // 4}",  # Simplified
            memory_bandwidth_gbps=memory_bandwidth,
            pci_bandwidth_gbps=pci_bandwidth,
            nvlink_bandwidth_gbps=nvlink_bandwidth,
            latency_us=1.0 if nvlink_gpus else 5.0,
            direct_nvlink_gpus=nvlink_gpus,
            same_pcie_switch_gpus=self._get_same_pcie_gpus(pci_bus_id),
            same_numa_node_gpus=self._get_same_numa_gpus(numa_node),
            max_p2p_bandwidth_gbps=max(pci_bandwidth, nvlink_bandwidth),
            optimal_chunk_size=self._calculate_optimal_chunk_size(gpu_id, nvlink_gpus)
        )
        
        self.gpus[rank] = gpu_topo
        return gpu_topo
    
    def _discover_gpu_topology_fallback(self, rank: int, gpu_id: int) -> GpuTopology:
        """Fallback topology discovery without NVML."""
        # Assume simple topology for testing
        return GpuTopology(
            gpu_id=gpu_id,
            rank=rank,
            pci_bus_id=f"0000:{gpu_id:02x}:00.0",
            numa_node=gpu_id // 4,
            pcie_switch=f"switch_{gpu_id // 4}",
            nvlink_domain=f"domain_{gpu_id // 8}",
            memory_bandwidth_gbps=900.0,  # A100-like
            pci_bandwidth_gbps=32.0,
            nvlink_bandwidth_gbps=300.0,
            latency_us=1.0,
            direct_nvlink_gpus={i for i in range(8) if i != rank and i // 8 == rank // 8},
            same_pcie_switch_gpus={i for i in range(4) if i != rank},
            same_numa_node_gpus={i for i in range(8) if i != rank and i // 4 == rank // 4},
            max_p2p_bandwidth_gbps=300.0,
            optimal_chunk_size=64 * 1024 * 1024
        )
    
    def _estimate_memory_bandwidth(self, gpu_id: int) -> float:
        """Estimate memory bandwidth based on GPU generation."""
        # Simplified estimation - real implementation would query device properties
        if torch and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(gpu_id)
            # Rough estimation based on compute capability
            if props.major >= 8:  # Ampere or newer
                return 900.0  # GB/s
            elif props.major >= 7:  # Turing
                return 650.0
            else:
                return 400.0
        return 900.0  # Default assumption
    
    def _get_numa_node_for_pcie_bus(self, pci_bus_id: str) -> int:
        """Get NUMA node for PCIe bus."""
        try:
            # Try to read NUMA affinity from sysfs
            bus_id = pci_bus_id.replace(':', '').replace('.', '')
            numa_path = f"/sys/bus/pci/devices/0000:{pci_bus_id}/numa_node"
            with open(numa_path, 'r') as f:
                return int(f.read().strip())
        except:
            return 0  # Default to NUMA node 0
    
    def _get_pcie_switch_for_bus(self, pci_bus_id: str) -> str:
        """Get PCIe switch identifier."""
        # Simplified - real implementation would parse lspci output
        device_id = int(pci_bus_id.split(':')[1], 16)
        return f"switch_{device_id // 4}"
    
    def _discover_nvlink_connections(self, gpu_id: int) -> Set[int]:
        """Discover NVLink connections."""
        if not pynvml:
            return set()
            
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            nvlink_gpus = set()
            
            # Query NVLink connections (simplified)
            for other_gpu in range(pynvml.nvmlDeviceGetCount()):
                if other_gpu == gpu_id:
                    continue
                    
                # Check if NVLink connection exists
                try:
                    # This is a simplified check - real implementation would
                    # query NVML NVLink capabilities
                    other_handle = pynvml.nvmlDeviceGetHandleByIndex(other_gpu)
                    if self._are_gpus_nvlink_connected(handle, other_handle):
                        nvlink_gpus.add(other_gpu)
                except:
                    continue
                    
            return nvlink_gpus
        except:
            return set()
    
    def _are_gpus_nvlink_connected(self, handle1, handle2) -> bool:
        """Check if two GPUs are NVLink connected."""
        # Simplified check - real implementation would query NVML
        # For now, assume GPUs in same node might be connected
        return True  # Placeholder
    
    def _get_same_pcie_gpus(self, pci_bus_id: str) -> Set[int]:
        """Get GPUs on same PCIe switch."""
        switch = self._get_pcie_switch_for_bus(pci_bus_id)
        return {rank for rank, gpu in self.gpus.items() 
                if gpu.pcie_switch == switch}
    
    def _get_same_numa_gpus(self, numa_node: int) -> Set[int]:
        """Get GPUs on same NUMA node."""
        return {rank for rank, gpu in self.gpus.items() 
                if gpu.numa_node == numa_node}
    
    def _calculate_optimal_chunk_size(self, gpu_id: int, nvlink_gpus: Set[int]) -> int:
        """Calculate optimal chunk size based on topology."""
        if nvlink_gpus:
            # NVLink can handle larger chunks efficiently
            return 128 * 1024 * 1024  # 128MB
        else:
            # PCIe benefits from smaller chunks for pipelining
            return 32 * 1024 * 1024   # 32MB


class TopologyAwareScheduler:
    """Schedules transfers based on GPU topology."""
    
    def __init__(self, topology_discovery: TopologyDiscovery):
        self.topology = topology_discovery
        self.scheduled_transfers: Dict[int, TransferSchedule] = {}
        self.congestion_tracker: Dict[Tuple[int, int], float] = {}
        self.performance_history: Dict[Tuple[int, int], List[float]] = {}
        
        # Performance monitoring
        self.schedule_duration = Gauge("topology_schedule_seconds", "Time to compute schedule")
        self.congestion_score = Gauge("topology_congestion_score", "Current congestion score", ["path"])
        self.path_utilization = Gauge("topology_path_utilization", "Path utilization %", ["path"])
        
    def compute_transfer_schedule(self, bucket_id: int, src_rank: int, 
                                 dst_ranks: List[int], size_bytes: int) -> TransferSchedule:
        """Compute optimal transfer schedule for a bucket."""
        with Timer() as timer:
            # Get source GPU topology
            src_gpu = self.topology.gpus.get(src_rank)
            if not src_gpu:
                # Fallback to simple schedule
                return self._create_fallback_schedule(bucket_id, src_rank, dst_ranks, size_bytes)
            
            # Discover all possible paths
            all_paths = {}
            for dst_rank in dst_ranks:
                paths = self._discover_paths(src_rank, dst_rank)
                all_paths[dst_rank] = paths
            
            # Optimize chunk assignments to paths
            chunk_size = src_gpu.optimal_chunk_size
            num_chunks = (size_bytes + chunk_size - 1) // chunk_size
            
            assignments = self._optimize_chunk_assignments(
                all_paths, num_chunks, chunk_size
            )
            
            # Calculate expected performance
            duration, total_bw, congestion = self._calculate_performance(
                assignments, all_paths
            )
            
            schedule = TransferSchedule(
                bucket_id=bucket_id,
                chunk_assignments=assignments,
                expected_duration=duration,
                total_bandwidth=total_bw,
                congestion_risk=congestion
            )
            
            self.scheduled_transfers[bucket_id] = schedule
            
        self.schedule_duration.set(timer.elapsed)
        return schedule
    
    def _discover_paths(self, src_rank: int, dst_rank: int) -> List[NetworkPath]:
        """Discover all possible network paths between GPUs."""
        if src_rank == dst_rank:
            return []
            
        src_gpu = self.topology.gpus.get(src_rank)
        dst_gpu = self.topology.gpus.get(dst_rank)
        
        if not src_gpu or not dst_gpu:
            # Fallback path
            return [NetworkPath(
                src_rank=src_rank,
                dst_rank=dst_rank,
                bandwidth_gbps=25.0,  # Conservative estimate
                latency_us=10.0,
                hop_count=1,
                path_type='ethernet'
            )]
        
        paths = []
        
        # Direct NVLink path (fastest)
        if dst_rank in src_gpu.direct_nvlink_gpus:
            paths.append(NetworkPath(
                src_rank=src_rank,
                dst_rank=dst_rank,
                bandwidth_gbps=src_gpu.nvlink_bandwidth_gbps,
                latency_us=1.0,
                hop_count=1,
                path_type='nvlink'
            ))
        
        # Same PCIe switch path
        if dst_rank in src_gpu.same_pcie_switch_gpus:
            paths.append(NetworkPath(
                src_rank=src_rank,
                dst_rank=dst_rank,
                bandwidth_gbps=src_gpu.pci_bandwidth_gbps,
                latency_us=5.0,
                hop_count=2,
                path_type='pcie'
            ))
        
        # Same NUMA node path
        if dst_rank in src_gpu.same_numa_node_gpus:
            paths.append(NetworkPath(
                src_rank=src_rank,
                dst_rank=dst_rank,
                bandwidth_gbps=src_gpu.pci_bandwidth_gbps * 0.8,  # Some degradation
                latency_us=10.0,
                hop_count=3,
                path_type='numa'
            ))
        
        # Fallback network path
        paths.append(NetworkPath(
            src_rank=src_rank,
            dst_rank=dst_rank,
            bandwidth_gbps=10.0,  # Conservative network estimate
            latency_us=50.0,
            hop_count=4,
            path_type='network'
        ))
        
        return paths
    
    def _optimize_chunk_assignments(self, all_paths: Dict[int, List[NetworkPath]], 
                                   num_chunks: int, chunk_size: int) -> Dict[int, List[NetworkPath]]:
        """Optimize chunk-to-path assignments."""
        assignments = {}
        
        for chunk_id in range(num_chunks):
            # For each chunk, select the best available path
            best_path = None
            best_score = -1
            
            for dst_rank, paths in all_paths.items():
                for path in paths:
                    # Calculate path score considering congestion and performance
                    score = self._calculate_path_score(path)
                    
                    if score > best_score:
                        best_score = score
                        best_path = path
            
            if best_path:
                if chunk_id not in assignments:
                    assignments[chunk_id] = []
                assignments[chunk_id].append(best_path)
                
                # Update congestion tracking
                path_key = (best_path.src_rank, best_path.dst_rank)
                self.congestion_tracker[path_key] = self.congestion_tracker.get(path_key, 0.0) + 0.1
                
        return assignments
    
    def _calculate_path_score(self, path: NetworkPath) -> float:
        """Calculate path selection score."""
        # Consider effective bandwidth, latency, and congestion
        bandwidth_score = path.effective_bandwidth() / 1000.0  # Normalize to 0-1
        latency_score = max(0, 1.0 - path.latency_us / 100.0)  # Prefer low latency
        congestion_score = max(0, 1.0 - path.congestion_score)
        
        # Weighted combination
        return (0.5 * bandwidth_score + 0.3 * latency_score + 0.2 * congestion_score)
    
    def _calculate_performance(self, assignments: Dict[int, List[NetworkPath]], 
                              all_paths: Dict[int, List[NetworkPath]]) -> Tuple[float, float, float]:
        """Calculate expected performance metrics."""
        total_chunks = len(assignments)
        if total_chunks == 0:
            return 0.0, 0.0, 0.0
        
        # Calculate total bandwidth
        total_bandwidth = sum(
            path.effective_bandwidth() 
            for chunk_paths in assignments.values() 
            for path in chunk_paths
        )
        
        # Estimate duration (simplified)
        avg_bandwidth = total_bandwidth / total_chunks
        chunk_size = 64 * 1024 * 1024  # Assume 64MB chunks
        estimated_duration = chunk_size / (avg_bandwidth * 1e9 / 8)  # seconds
        
        # Calculate congestion risk
        max_congestion = max(self.congestion_tracker.values()) if self.congestion_tracker else 0.0
        congestion_risk = min(max_congestion / 10.0, 1.0)  # Normalize to 0-1
        
        return estimated_duration, total_bandwidth, congestion_risk
    
    def _create_fallback_schedule(self, bucket_id: int, src_rank: int, 
                                 dst_ranks: List[int], size_bytes: int) -> TransferSchedule:
        """Create a simple fallback schedule when topology is unknown."""
        num_chunks = (size_bytes + (32 * 1024 * 1024) - 1) // (32 * 1024 * 1024)
        
        assignments = {}
        for chunk_id in range(num_chunks):
            # Simple round-robin assignment
            dst_rank = dst_ranks[chunk_id % len(dst_ranks)]
            path = NetworkPath(
                src_rank=src_rank,
                dst_rank=dst_rank,
                bandwidth_gbps=25.0,
                latency_us=10.0,
                hop_count=1,
                path_type='fallback'
            )
            assignments[chunk_id] = [path]
        
        return TransferSchedule(
            bucket_id=bucket_id,
            chunk_assignments=assignments,
            expected_duration=10.0,  # Conservative estimate
            total_bandwidth=25.0 * len(dst_ranks),
            congestion_risk=0.5
        )
    
    def update_performance_metrics(self, bucket_id: int, actual_duration: float):
        """Update performance history with actual results."""
        schedule = self.scheduled_transfers.get(bucket_id)
        if not schedule:
            return
            
        # Update path performance history
        for chunk_paths in schedule.chunk_assignments.values():
            for path in chunk_paths:
                path_key = (path.src_rank, path.dst_rank)
                if path_key not in self.performance_history:
                    self.performance_history[path_key] = []
                
                # Calculate performance ratio
                expected_chunk_time = schedule.expected_duration / len(schedule.chunk_assignments)
                actual_chunk_time = actual_duration / len(schedule.chunk_assignments)
                performance_ratio = expected_chunk_time / actual_chunk_time
                
                self.performance_history[path_key].append(performance_ratio)
                
                # Keep only recent history
                if len(self.performance_history[path_key]) > 100:
                    self.performance_history[path_key].pop(0)
                
                # Update path reliability score
                if self.performance_history[path_key]:
                    avg_performance = sum(self.performance_history[path_key]) / len(self.performance_history[path_key])
                    path.reliability_score = max(0.1, min(1.0, avg_performance))
    
    def get_congestion_report(self) -> Dict[str, float]:
        """Get current congestion report."""
        report = {}
        for (src, dst), congestion in self.congestion_tracker.items():
            path_key = f"{src}->{dst}"
            report[path_key] = congestion
            self.congestion_score.labels(path=path_key).set(congestion)
        return report


class AdaptiveLaneManager:
    """Manages adaptive lane scaling based on performance."""
    
    def __init__(self, scheduler: TopologyAwareScheduler):
        self.scheduler = scheduler
        self.lane_scaling_history: Dict[str, List[float]] = {}
        self.optimal_lanes: Dict[str, int] = {}
        
    def adapt_lane_count(self, src_rank: int, dst_rank: int, 
                        current_lanes: int, performance_metrics: Dict[str, float]) -> int:
        """Adapt lane count based on performance."""
        path_key = f"{src_rank}->{dst_rank}"
        
        # Track performance history
        if path_key not in self.lane_scaling_history:
            self.lane_scaling_history[path_key] = []
            
        utilization = performance_metrics.get('bandwidth_utilization', 0.0)
        self.lane_scaling_history[path_key].append(utilization)
        
        # Keep recent history
        if len(self.lane_scaling_history[path_key]) > 20:
            self.lane_scaling_history[path_key].pop(0)
        
        # Calculate average utilization
        if not self.lane_scaling_history[path_key]:
            return current_lanes
            
        avg_utilization = sum(self.lane_scaling_history[path_key]) / len(self.lane_scaling_history[path_key])
        
        # Adaptive scaling logic
        if avg_utilization > 0.8 and current_lanes < 16:
            # High utilization - add lanes
            new_lanes = min(current_lanes + 2, 16)
        elif avg_utilization < 0.3 and current_lanes > 1:
            # Low utilization - reduce lanes
            new_lanes = max(current_lanes - 1, 1)
        else:
            new_lanes = current_lanes
            
        self.optimal_lanes[path_key] = new_lanes
        return new_lanes
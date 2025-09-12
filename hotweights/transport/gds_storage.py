"""
GPUDirect Storage integration for true zero-copy transfers.

Provides direct file-to-GPU memory transfers without CPU involvement.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

try:
    import torch
    import nvidia.gds as gds  # GPUDirect Storage Python bindings
    GDS_AVAILABLE = True
except ImportError:
    GDS_AVAILABLE = False
    gds = None
    torch = None

logger = logging.getLogger(__name__)


class GDSDirectTransfer:
    """True zero-copy file-to-GPU transfer using GPUDirect Storage."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._initialized = False
        self._gds_context = None
        
        if GDS_AVAILABLE:
            try:
                self._initialize_gds()
            except Exception as e:
                logger.warning(f"Failed to initialize GDS: {e}")
                GDS_AVAILABLE = False
    
    def _initialize_gds(self):
        """Initialize GPUDirect Storage context."""
        if not GDS_AVAILABLE:
            return
            
        # Set CUDA device
        torch.cuda.set_device(self.device_id)
        
        # Initialize GDS context
        self._gds_context = gds.Context()
        self._initialized = True
        logger.info(f"GPUDirect Storage initialized on device {self.device_id}")
    
    def assemble_bucket_direct(
        self, 
        items: List[Dict[str, Any]], 
        gpu_buffer: torch.Tensor,
        offset: int = 0
    ) -> None:
        """
        Assemble bucket directly into GPU memory using GDS.
        
        Args:
            items: List of tensor items with file paths
            gpu_buffer: Pre-allocated GPU buffer
            offset: Starting offset in GPU buffer
        """
        if not self._initialized:
            raise RuntimeError("GDS not initialized")
        
        current_offset = offset
        
        for item in items:
            uri = item["uri"]
            if not uri.startswith("file://"):
                raise ValueError(f"Unsupported URI: {uri}")
            
            file_path = Path(uri[len("file://"):])
            nbytes = int(item["nbytes"])
            
            # Create file handle
            with gds.FileHandle(str(file_path), gds.FileMode.READ) as fh:
                # Create GPU memory view
                gpu_slice = gpu_buffer[current_offset:current_offset + nbytes]
                
                # Perform direct file-to-GPU transfer
                fh.read_into_cuda_buffer(
                    gpu_slice.data_ptr(), 
                    nbytes,
                    offset=int(item.get("file_offset", 0))
                )
                
                # Verify transfer
                expected_hash = item.get("hash")
                if expected_hash:
                    self._verify_transfer(gpu_slice, expected_hash)
            
            current_offset += nbytes
            logger.debug(f"Transferred {nbytes} bytes from {file_path}")
    
    def _verify_transfer(self, gpu_data: torch.Tensor, expected_hash: str):
        """Verify transferred data against expected hash."""
        # Copy small portion to CPU for hash verification
        cpu_copy = gpu_data.cpu().numpy()
        
        import hashlib
        algo, expected_digest = expected_hash.split(":", 1)
        h = hashlib.new(algo)
        h.update(cpu_copy.tobytes())
        actual_digest = h.hexdigest()
        
        if actual_digest != expected_digest:
            raise ValueError(f"Hash mismatch: expected {expected_digest}, got {actual_digest}")
    
    def get_optimal_transfer_size(self, file_size: int) -> int:
        """Get optimal transfer chunk size based on file size and hardware."""
        # GDS works best with larger transfers
        if file_size < 4 * 1024 * 1024:  # < 4MB
            return file_size
        elif file_size < 64 * 1024 * 1024:  # < 64MB
            return 4 * 1024 * 1024  # 4MB chunks
        else:
            return 16 * 1024 * 1024  # 16MB chunks
    
    @property
    def is_available(self) -> bool:
        """Check if GPUDirect Storage is available and initialized."""
        return self._initialized and GDS_AVAILABLE


class GDSFallbackTransfer:
    """Fallback to traditional CPU-based transfer when GDS is unavailable."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        
    def assemble_bucket_direct(
        self, 
        items: List[Dict[str, Any]], 
        gpu_buffer: torch.Tensor,
        offset: int = 0
    ) -> None:
        """Fallback implementation using CPU memory map + async H2D copy."""
        import mmap
        
        current_offset = offset
        stream = torch.cuda.Stream(device=self.device_id)
        
        with torch.cuda.stream(stream):
            for item in items:
                uri = item["uri"]
                file_path = Path(uri[len("file://"):])
                nbytes = int(item["nbytes"])
                
                # Memory map file
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), nbytes, access=mmap.ACCESS_READ) as mm:
                        # Create CPU pinned buffer
                        cpu_buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                        cpu_buffer.copy_(torch.frombuffer(mm, dtype=torch.uint8))
                        
                        # Async copy to GPU
                        gpu_slice = gpu_buffer[current_offset:current_offset + nbytes]
                        gpu_slice.copy_(cpu_buffer, non_blocking=True)
                
                current_offset += nbytes
        
        # Wait for async operations to complete
        stream.synchronize()


def create_gds_transfer(device_id: int = 0) -> GDSDirectTransfer:
    """Factory function to create appropriate transfer mechanism."""
    if GDS_AVAILABLE:
        try:
            return GDSDirectTransfer(device_id)
        except Exception as e:
            logger.warning(f"GDS initialization failed: {e}, falling back to pinned memory")
    
    return GDSFallbackTransfer(device_id)
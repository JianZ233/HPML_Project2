import torch
from typing import List, Dict, Optional, Tuple
from gpu_verification import verify_gpu_availability

class GPUMemoryTracker:
    """
    GPU memory usage and allocation tracker.
    
    This class provides utilities for tracking and managing GPU memory usage
    across multiple devices. It maintains statistics about memory allocation,
    tracks peak memory usage, and helps optimize device selection for tensor
    operations.
    """
    
    def __init__(self, device_ids: List[int]):
        """
        Initialize the GPU memory tracker.
        
        Args:
            device_ids: List of GPU device IDs to monitor
        """
        # Verify GPU availability first
        verified_devices, warning = verify_gpu_availability(device_ids)
        if warning:
            print(warning)
        
        self.device_ids = verified_devices
        self.peak_memory = {device: 0 for device in verified_devices}
        
        # Rest of the initialization remains the same
        for device in self.device_ids:
            torch.cuda.reset_peak_memory_stats(device)
        
        self.memory_stats = {
            device: self._get_memory_stats(device)
            for device in self.device_ids
        }
    
    def _get_memory_stats(self, device: int) -> Dict[str, int]:
        """
        Get detailed memory statistics for a GPU.
        
        Args:
            device: GPU device ID
            
        Returns:
            Dict containing memory statistics:
                - total: Total memory available
                - allocated: Currently allocated memory
                - reserved: Reserved memory
                - free: Available memory
                - cached: Cached memory
                - max_allocated: Maximum allocated memory
        """
        props = torch.cuda.get_device_properties(device)
        current_allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        stats = {
            'total': props.total_memory,
            'allocated': current_allocated,
            'reserved': reserved,
            'free': props.total_memory - current_allocated,
            'cached': reserved - current_allocated,
            'max_allocated': torch.cuda.max_memory_allocated(device)
        }
        
        # Update peak memory
        self.peak_memory[device] = max(
            self.peak_memory[device],
            stats['allocated']
        )
        
        return stats
    
    def get_optimal_device(self, tensor_size: int) -> int:
        """
        Get the GPU with the most available memory.
        
        Args:
            tensor_size: Size of tensor to allocate in bytes
            
        Returns:
            Device ID of GPU with most available memory
        """
        self.update_stats()
        available_memory = {
            device: stats['free']
            for device, stats in self.memory_stats.items()
        }
        return max(available_memory.items(), key=lambda x: x[1])[0]
    
    def update_stats(self) -> None:
        """Update memory statistics for all monitored GPUs."""
        for device in self.device_ids:
            self.memory_stats[device] = self._get_memory_stats(device)
    
    def get_memory_info(self, device: int) -> Dict[str, float]:
        """
        Get formatted memory information for a device.
        
        Args:
            device: GPU device ID
            
        Returns:
            Dict containing memory information in MB:
                - total: Total memory
                - allocated: Allocated memory
                - reserved: Reserved memory
                - free: Free memory
                - cached: Cached memory
                - peak: Peak memory usage
        """
        stats = self._get_memory_stats(device)
        return {
            'total': stats['total'] / (1024**2),
            'allocated': stats['allocated'] / (1024**2),
            'reserved': stats['reserved'] / (1024**2),
            'free': stats['free'] / (1024**2),
            'cached': stats['cached'] / (1024**2),
            'peak': self.peak_memory[device] / (1024**2)
        }
    
    def get_peak_memory(self, device: int) -> float:
        """
        Get peak memory usage for a specific device in bytes.
        
        Args:
            device: GPU device ID
            
        Returns:
            Peak memory usage in bytes
        """
        return self.peak_memory.get(device, 0)
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics for all devices."""
        for device in self.device_ids:
            torch.cuda.reset_peak_memory_stats(device)
            self.peak_memory[device] = 0
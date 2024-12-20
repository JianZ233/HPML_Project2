import torch
from typing import List, Dict, Optional
import time

class GPUMemoryTracker:
    """Enhanced GPU memory usage and allocation tracker."""
    def __init__(self, device_ids: List[int]):
        """Initialize the GPU memory tracker.
        
        Args:
            device_ids: List of GPU device IDs to monitor
        """
        self.device_ids = device_ids
        self.start_time = time.time()
        
        # Initialize peak memory tracking first
        self.peak_memory = {device: 0 for device in device_ids}
        
        # Reset peak memory stats for each device
        for device in device_ids:
            torch.cuda.reset_peak_memory_stats(device)
        
        # Initialize memory stats after peak memory is set up
        self.memory_stats = {device: self._get_memory_stats(device)
                           for device in device_ids}
    
    def _get_memory_stats(self, device: int) -> Dict[str, int]:
        """Get detailed memory statistics for a GPU."""
        props = torch.cuda.get_device_properties(device)
        current_allocated = torch.cuda.memory_allocated(device)
        
        stats = {
            'total': props.total_memory,
            'allocated': current_allocated,
            'reserved': torch.cuda.memory_reserved(device),
            'free': props.total_memory - current_allocated,
            'cached': torch.cuda.memory_reserved(device) - current_allocated,
            'max_allocated': torch.cuda.max_memory_allocated(device)
        }
        
        # Update peak memory
        self.peak_memory[device] = max(
            self.peak_memory[device],
            stats['allocated']
        )
        
        return stats
    
    def get_optimal_device(self, tensor_size: int) -> int:
        """Get the GPU with the most available memory."""
        self.update_stats()
        available_memory = {
            device: stats['free']
            for device, stats in self.memory_stats.items()
        }
        return max(available_memory.items(), key=lambda x: x[1])[0]
    
    def update_stats(self):
        """Update memory statistics for all GPUs."""
        for device in self.device_ids:
            self.memory_stats[device] = self._get_memory_stats(device)
    
    def log_memory_usage(self, prefix: str = ""):
        """Log detailed memory usage of all GPUs."""
        self.update_stats()
        if prefix:
            print(f"\n{prefix}")
            
        for device, stats in self.memory_stats.items():
            print(f"\nDevice {device}:")
            print(f"  Total Memory: {stats['total'] / (1024**2):.2f} MB")
            print(f"  Allocated Memory: {stats['allocated'] / (1024**2):.2f} MB")
            print(f"  Reserved Memory: {stats['reserved'] / (1024**2):.2f} MB")
            print(f"  Free Memory: {stats['free'] / (1024**2):.2f} MB")
            print(f"  Cached Memory: {stats['cached'] / (1024**2):.2f} MB")
            print(f"  Peak Memory Used: {self.peak_memory[device] / (1024**2):.2f} MB")

    def get_peak_memory(self, device: int) -> float:
        """Get peak memory usage for a specific device."""
        return self.peak_memory.get(device, 0)
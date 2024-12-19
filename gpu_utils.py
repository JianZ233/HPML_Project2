import torch
from typing import List, Dict

class GPUMemoryTracker:
    """Tracks GPU memory usage and allocation."""
    def __init__(self, device_ids: List[int]):
        self.device_ids = device_ids
        self.memory_stats = {device: self._get_memory_stats(device)
                             for device in device_ids}
    
    def _get_memory_stats(self, device: int) -> Dict[str, int]:
        """Get memory statistics for a GPU."""
        return {
            'total': torch.cuda.get_device_properties(device).total_memory,
            'allocated': torch.cuda.memory_allocated(device),
            'reserved': torch.cuda.memory_reserved(device)
        }
    
    def get_optimal_device(self, tensor_size: int) -> int:
        """Get the GPU with the most available memory."""
        self.update_stats()
        available_memory = {
            device: stats['total'] - stats['allocated']
            for device, stats in self.memory_stats.items()
        }
        return max(available_memory.items(), key=lambda x: x[1])[0]
    
    def update_stats(self):
        """Update memory statistics for all GPUs."""
        for device in self.device_ids:
            self.memory_stats[device] = self._get_memory_stats(device)
    
    def log_memory_usage(self):
        """Log the current memory usage of all GPUs."""
        self.update_stats()
        for device, stats in self.memory_stats.items():
            print(f"Device {device}:")
            print(f"  Total Memory: {stats['total'] / (1024 ** 2):.2f} MB")
            print(f"  Allocated Memory: {stats['allocated'] / (1024 ** 2):.2f} MB")
            print(f"  Reserved Memory: {stats['reserved'] / (1024 ** 2):.2f} MB")
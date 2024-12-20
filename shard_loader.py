import queue
import threading
import torch
from typing import Dict, Optional, Tuple, Any

class AsyncShardLoader:
    """
    Asynchronous loader for model weight shards.
    
    This class implements an asynchronous loading mechanism for large model weights,
    using a producer-consumer pattern with a prefetch queue. It processes shards
    in a background thread while allowing the main thread to consume processed
    shards efficiently.
    """
    
    def __init__(self, num_prefetch: int = 2):
        """
        Initialize the asynchronous shard loader.
        
        Args:
            num_prefetch: Maximum number of shards to prefetch (default: 2)
        """
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.current_shard = None
        self.thread = None
        self.error = None
        self._processed_count = 0

    def start_prefetch(
        self,
        shard_mapping: Dict[str, Any],
        fp8_handler: Any,
        device: torch.device
    ) -> None:
        """
        Start the prefetch thread for processing shards.
        
        Args:
            shard_mapping: Mapping of shard IDs to their data
            fp8_handler: Handler for FP8 quantization
            device: Target device for processed tensors
        """
        def _prefetch_worker():
            try:
                for shard_id, shard_data in shard_mapping.items():
                    if self.stop_event.is_set():
                        break
                    
                    processed = fp8_handler.process_weights(shard_data)
                    self.queue.put((shard_id, processed))
                    self._processed_count += 1
                    
                self.queue.put(None)  # Signal completion
            except Exception as e:
                self.error = e
                self.queue.put(None)

        self.thread = threading.Thread(target=_prefetch_worker)
        self.thread.start()

    def get_next_shard(self) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """
        Get the next processed shard from the queue.
        
        Returns:
            Optional tuple containing shard ID and processed tensors,
            or None if no more shards are available.
            
        Raises:
            RuntimeError: If an error occurred in the prefetch worker
        """
        if self.error:
            raise RuntimeError(f"Error in prefetch worker: {self.error}")
        
        if self.current_shard is None:
            self.current_shard = self.queue.get()
        return self.current_shard

    def advance(self) -> None:
        """
        Advance to the next shard by clearing the current shard reference.
        This allows the next call to get_next_shard to fetch a new shard.
        """
        self.current_shard = None

    def stop(self) -> None:
        """
        Stop the prefetch thread and clean up resources.
        Blocks until the thread has completed.
        """
        self.stop_event.set()
        if self.thread:
            self.thread.join()
    
    @property
    def processed_count(self) -> int:
        """
        Get the number of shards that have been processed.
        
        Returns:
            int: Number of processed shards
        """
        return self._processed_count
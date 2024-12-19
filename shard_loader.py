import queue
import threading
import torch
from typing import Dict, Optional, Tuple

class AsyncShardLoader:
    """Handles asynchronous loading of weight shards."""
    
    def __init__(self, num_prefetch: int = 2):
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.current_shard = None
        self.thread = None
        self.error = None

    def start_prefetch(self, shard_mapping: Dict, fp8_handler, device: torch.device):
        """Start prefetch thread for shards."""
        def _prefetch_worker():
            try:
                for shard_id, shard_data in shard_mapping.items():
                    if self.stop_event.is_set():
                        break
                    # Process weights with proper FP8 handling
                    processed = fp8_handler.process_weights(shard_data)
                    self.queue.put((shard_id, processed))
                self.queue.put(None)  # Signal completion
            except Exception as e:
                self.error = e
                self.queue.put(None)

        self.thread = threading.Thread(target=_prefetch_worker)
        self.thread.start()

    def get_next_shard(self) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """Get next processed shard."""
        if self.error:
            raise RuntimeError(f"Error in prefetch worker: {self.error}")
        
        if self.current_shard is None:
            self.current_shard = self.queue.get()
        return self.current_shard

    def advance(self):
        """Move to next shard."""
        self.current_shard = None

    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()
        if self.thread:
            self.thread.join()
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import List, Optional, Dict, Any, Tuple
import os
import queue
import threading
from dataclasses import dataclass
import numpy as np
from safetensors.torch import load_file

@dataclass
class FP8Format:
    """Configuration for FP8 format."""
    e4m3: bool = True  # True for e4m3, False for e5m2
    scale: float = 1.0
    bias: int = 7  # 7 for e4m3, 15 for e5m2
    max_value: float = 448.0  # 448.0 for e4m3, 57344.0 for e5m2

class AsyncShardLoader:
    """Handles asynchronous loading of weight shards."""
    def __init__(self, num_prefetch: int = 2):
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.current_shard = None
        self.thread = None

    def start_prefetch(self, shard_mapping: Dict, fp8_handler, device: torch.device):
        """Start prefetch thread for shards."""
        def _prefetch_worker():
            for shard_id, shard_data in shard_mapping.items():
                if self.stop_event.is_set():
                    break
                # Pre-process shard
                processed = fp8_handler.dequantize_weights(shard_data)
                self.queue.put((shard_id, processed))
            self.queue.put(None)  # Signal completion

        self.thread = threading.Thread(target=_prefetch_worker)
        self.thread.start()

    def get_next_shard(self) -> Optional[Tuple[str, Dict[str, torch.Tensor]]]:
        """Get next preprocessed shard."""
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

class FP8QuantizationHandler:
    """Enhanced FP8 quantization handler with proper FP8 format support."""
    
    def __init__(self, fp8_format: Optional[FP8Format] = None):
        self.fp8_format = fp8_format or FP8Format()
        
    def load_quantized_weights(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Load FP8 quantized weights with proper metadata."""
        try:
            weights = load_file(checkpoint_path)
            
            # Check if the weights include FP8 config
            if 'fp8_config' in weights:
                self.fp8_format = FP8Format(**weights['fp8_config'])
                
            # If weights are nested in model_state, extract them
            if 'model_state' in weights:
                weights = weights['model_state']
                
            return weights
            
        except Exception as e:
            raise RuntimeError(f"Failed to load safetensors file: {e}")

    def dequantize_weights(self, quantized_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Dequantize FP8 weights to FP16/FP32."""
        dequantized = {}
        
        for name, tensor in quantized_weights.items():
            if isinstance(tensor, dict) and 'quantized' in tensor:
                # Handle case where tensor includes metadata
                quantized = tensor['quantized']
                scale = tensor.get('scale', self.fp8_format.scale)
            else:
                quantized = tensor
                scale = self.fp8_format.scale

            # First convert FP8 to FP32, then apply scaling
            if quantized.dtype == torch.float8_e4m3fn:
                # Convert to FP32 first
                fp32_tensor = quantized.to(torch.float32)
                # Apply scaling in FP32
                scaled_tensor = fp32_tensor * scale
                # Convert to FP16 for efficiency
                dequantized[name] = scaled_tensor.to(torch.float16)
            elif quantized.dtype == torch.float8_e5m2:
                # Handle e5m2 format similarly
                fp32_tensor = quantized.to(torch.float32)
                scaled_tensor = fp32_tensor * scale
                dequantized[name] = scaled_tensor.to(torch.float16)
            else:
                # For non-FP8 tensors, process normally
                dequantized[name] = (quantized * scale).to(torch.float16)
            
        return dequantized

class ShardedFP8ModelLoader:
    def __init__(
        self,
        model_cls: nn.Module,
        device_ids: Optional[List[int]] = None,
        use_ddp: bool = True,
        memory_efficient: bool = True,
        mixed_precision: bool = True,
        shard_size_gb: float = 4.0,
        num_prefetch_shards: int = 2
    ):
        """Enhanced initialization with prefetch support."""
        super().__init__()
        self.model_cls = model_cls
        self.device_ids = device_ids or self._get_available_devices()
        self.use_ddp = use_ddp
        self.memory_efficient = memory_efficient
        self.mixed_precision = mixed_precision
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.fp8_handler = FP8QuantizationHandler()
        self.async_loader = AsyncShardLoader(num_prefetch_shards)
        
        if self.use_ddp and not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized, but use_ddp=True.")

    def _get_available_devices(self) -> List[int]:
        """Get list of available CUDA devices."""
        if not torch.cuda.is_available():
            return [0]  # CPU only
        return list(range(torch.cuda.device_count()))

    def _create_shard_mapping(self, weights: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create mapping of weights to shards based on size constraints.
        
        Args:
            weights: Dictionary of model weights
            
        Returns:
            Dictionary mapping shard IDs to weight dictionaries
        """
        shard_mapping = {}
        current_shard = {}
        current_shard_size = 0
        shard_id = 0
        
        # Sort weights by size for better distribution
        sorted_weights = sorted(
            weights.items(),
            key=lambda x: x[1].numel() * x[1].element_size(),
            reverse=True
        )
        
        for name, tensor in sorted_weights:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If tensor is larger than shard size, split it
            if tensor_size > self.shard_size_bytes:
                # Calculate number of splits needed
                num_splits = int(np.ceil(tensor_size / self.shard_size_bytes))
                splits = torch.chunk(tensor, num_splits)
                
                for i, split in enumerate(splits):
                    split_name = f"{name}_split_{i}"
                    shard_mapping[shard_id] = {split_name: split}
                    shard_id += 1
                continue
            
            # If adding tensor would exceed shard size, create new shard
            if current_shard_size + tensor_size > self.shard_size_bytes and current_shard:
                shard_mapping[shard_id] = current_shard
                current_shard = {}
                current_shard_size = 0
                shard_id += 1
            
            current_shard[name] = tensor
            current_shard_size += tensor_size
        
        # Add remaining weights to final shard
        if current_shard:
            shard_mapping[shard_id] = current_shard
        
        return shard_mapping

    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Enhanced model loading with async shard processing and meta tensor handling."""
        model = self.model_cls()
        fp8_weights = self.fp8_handler.load_quantized_weights(checkpoint_path)
        
        if not torch.cuda.is_available():
            weights_fp = self.fp8_handler.dequantize_weights(fp8_weights)
            model.load_state_dict(weights_fp)
            return model
        
        # Create shard mapping and start prefetch
        self.shard_mapping = self._create_shard_mapping(fp8_weights)
        primary_device = torch.device(f'cuda:{self.device_ids[0]}')
        
        # Handle meta tensors by using to_empty() first
        try:
            model.to_empty(device=primary_device)
        except AttributeError:
            # Fallback for models that don't support to_empty
            model.to(primary_device)
        
        # Start async loading
        self.async_loader.start_prefetch(self.shard_mapping, self.fp8_handler, primary_device)
        
        # Setup CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        try:
            # Process shards asynchronously
            while True:
                shard = self.async_loader.get_next_shard()
                if shard is None:
                    break
                    
                shard_id, processed_weights = shard
                
                # Load shard weights into model
                with torch.cuda.stream(torch.cuda.Stream()):
                    self._load_shard_weights(model, processed_weights)
                    
                self.async_loader.advance()
                
                if self.memory_efficient:
                    torch.cuda.empty_cache()
            
            # Setup DDP if requested
            if self.use_ddp and len(self.device_ids) > 1:
                model = self._setup_ddp(model)
            
        finally:
            self.async_loader.stop()
        
        # Record timing
        end_event.record()
        torch.cuda.synchronize()
        
        loading_time = start_event.elapsed_time(end_event) / 1000
        print(f"Model loaded in {loading_time:.2f} seconds")
        
        return model

    def _load_shard_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """Load shard weights into model efficiently."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])

    def _setup_ddp(self, model: nn.Module) -> nn.Module:
        """Setup DDP with optimized settings."""
        return DDP(
            model,
            device_ids=[self.device_ids[0]],
            output_device=self.device_ids[0],
            broadcast_buffers=False,
            bucket_cap_mb=25,
            gradient_as_bucket_view=True,
            static_graph=True
        )
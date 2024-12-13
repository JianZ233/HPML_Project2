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
        
    def _quantize_to_fp8(self, fp32_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert FP32 tensor to FP8 format."""
        abs_max = torch.max(torch.abs(fp32_tensor)).item()
        scale = abs_max / self.fp8_format.max_value
        
        scaled = fp32_tensor / scale
        scaled = torch.clamp(scaled, -self.fp8_format.max_value, self.fp8_format.max_value)
        
        # Simulate FP8 quantization
        if self.fp8_format.e4m3:
            # e4m3: 4-bit exponent, 3-bit mantissa
            exp_bits = 4
            man_bits = 3
        else:
            # e5m2: 5-bit exponent, 2-bit mantissa
            exp_bits = 5
            man_bits = 2
            
        total_bits = exp_bits + man_bits + 1  # +1 for sign bit
        
        # Quantize to FP8 precision
        quantum = 2.0 ** (-man_bits)
        quantized = torch.round(scaled / quantum) * quantum
        
        return quantized, torch.tensor(scale)

    def load_quantized_weights(self, checkpoint_path: str) -> Dict[str, torch.Tensor]:
        """Load FP8 quantized weights with proper metadata."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            weights = checkpoint['model_state']
            if 'fp8_config' in checkpoint:
                self.fp8_format = FP8Format(**checkpoint['fp8_config'])
        else:
            weights = checkpoint
            
        return weights

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
                
            # Convert to FP16 for efficiency
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

    def load_model(self, checkpoint_path: str) -> nn.Module:
        """Enhanced model loading with async shard processing."""
        model = self.model_cls()
        fp8_weights = self.fp8_handler.load_quantized_weights(checkpoint_path)
        
        if not torch.cuda.is_available():
            weights_fp = self.fp8_handler.dequantize_weights(fp8_weights)
            model.load_state_dict(weights_fp)
            return model
        
        # Create shard mapping and start prefetch
        self.shard_mapping = self._create_shard_mapping(fp8_weights)
        primary_device = torch.device(f'cuda:{self.device_ids[0]}')
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
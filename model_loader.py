import torch
import gc
import os
import json
import time
from torch import nn
from typing import Dict, List, Optional
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file
from fp8_format import FP8Format
from quantization import FP8QuantizationHandler
from gpu_utils import GPUMemoryTracker
from shard_loader import AsyncShardLoader

class ShardedFP8ModelLoader:
    """Main loader class for handling sharded FP8 models."""
    
    def __init__(
        self,
        model_dir: str,
        device_ids: Optional[List[int]] = None,
        use_ddp: bool = False,
        memory_efficient: bool = True,
        mixed_precision: bool = True,
        shard_size_gb: float = 4.0,
        fp8_format: Optional[FP8Format] = None
    ):
        """
        Initialize the FP8 model loader.
        
        Args:
            model_dir: Directory containing model files
            device_ids: List of GPU device IDs to use
            use_ddp: Whether to use DistributedDataParallel
            memory_efficient: Whether to use memory-efficient loading
            mixed_precision: Whether to use mixed precision
            shard_size_gb: Size of each shard in gigabytes
            fp8_format: FP8 format configuration
        """
        self.model_dir = model_dir
        self.device_ids = device_ids or [0]
        self.use_ddp = use_ddp
        self.memory_efficient = memory_efficient
        self.mixed_precision = mixed_precision
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.fp8_handler = FP8QuantizationHandler(fp8_format=fp8_format)
        self.async_loader = AsyncShardLoader()
        self.memory_tracker = GPUMemoryTracker(self.device_ids)

    def load_model(self) -> nn.Module:
        """Load and initialize model from safetensors shards."""
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Load config and create model instance
            config = AutoConfig.from_pretrained(self.model_dir)
            
            # Initialize model with mixed precision on CPU first
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32
            ).cpu()

            if self.memory_efficient:
                self._convert_to_meta(model)
            
            # Get shard files
            shard_files = sorted([
                f for f in os.listdir(self.model_dir) 
                if f.startswith("model-") and f.endswith(".safetensors")
            ])
            
            print(f"Found {len(shard_files)} shards: {shard_files}")
            
            # Load index file if it exists
            index_path = os.path.join(self.model_dir, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
                print("Loaded index file")
            
            # Load shards
            start_time = time.time()
            load_device = torch.device("cpu")
            
            for shard_file in shard_files:
                shard_path = os.path.join(self.model_dir, shard_file)
                print(f"\nLoading shard: {shard_path}")
                
                # Load shard weights using safetensors
                shard_weights = load_file(shard_path)
                
                # Process weights with FP8 handler if needed
                if self.fp8_handler:
                    shard_weights = self.fp8_handler.process_weights(shard_weights)
                
                # Load weights into model
                self._load_shard_weights(model, shard_weights, device=load_device)
                
                # Clear memory
                del shard_weights
                if self.memory_efficient:
                    gc.collect()
                    torch.cuda.empty_cache()

            # Ensure no parameters remain on meta device
            self._ensure_no_meta_left(model, load_device)
            
            # Distribute model across GPUs
            model = self._distribute_model_across_gpus(model)

            # Optionally wrap with DDP if multiple GPUs are used
            if self.use_ddp and len(self.device_ids) > 1:
                model = nn.parallel.DistributedDataParallel(
                    model, 
                    device_ids=[self.device_ids[0]]
                )

            end_time = time.time()
            print(f"\nModel loaded in {end_time - start_time:.2f} seconds")
            
            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        finally:
            self.async_loader.stop()

    def _create_shard_mapping(self, weights: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """Create a mapping of weights to shards based on shard_size_bytes."""
        shard_mapping = {}
        current_shard = {}
        current_size = 0
        shard_id = 0

        # Sort weights by size for better distribution
        sorted_weights = sorted(
            weights.items(), 
            key=lambda x: x[1].numel() * x[1].element_size(), 
            reverse=True
        )

        for name, tensor in sorted_weights:
            tensor_size = tensor.numel() * tensor.element_size()
            if current_size + tensor_size > self.shard_size_bytes and current_shard:
                shard_mapping[shard_id] = current_shard
                current_shard = {}
                current_size = 0
                shard_id += 1

            current_shard[name] = tensor
            current_size += tensor_size

        if current_shard:
            shard_mapping[shard_id] = current_shard

        return shard_mapping

    def _convert_to_meta(self, model: nn.Module):
        """
        Convert model parameters to meta tensors with proper type handling.
        """
        updates = []
        with torch.no_grad():
            for full_name, param in list(model.named_parameters(recurse=True)):
                if param.is_meta:
                    continue
                
                # Determine appropriate meta dtype
                orig_dtype = param.dtype
                meta_dtype = torch.float16 if str(orig_dtype).startswith('torch.float8') else orig_dtype
                
                try:
                    new_param = nn.Parameter(
                        torch.empty(param.shape, dtype=meta_dtype, device='meta'),
                        requires_grad=param.requires_grad
                    )
                except RuntimeError as e:
                    print(f"Failed to convert {full_name}: {e}, shape={param.shape}, dtype={param.dtype}")
                    continue
                
                # Find parent module and schedule update
                parent = model
                attrs = full_name.split('.')
                for attr in attrs[:-1]:
                    parent = getattr(parent, attr)
                param_name = attrs[-1]
                updates.append((parent, param_name, new_param))
        
        # Apply all updates
        for parent, param_name, new_param in updates:
            setattr(parent, param_name, new_param)

    def _load_shard_weights(self, model: nn.Module, weights: Dict[str, torch.Tensor], device: torch.device):
        """Load weights from a shard into the model."""
        unexpected_keys = []
        missing_keys = []
        
        for name, tensor in weights.items():
            try:
                param = None
                obj = model
                parts = name.split('.')
                
                # Navigate to correct parameter
                for attr in parts[:-1]:
                    if hasattr(obj, attr):
                        obj = getattr(obj, attr)
                    else:
                        param = None
                        break
                
                if param is None and hasattr(obj, parts[-1]):
                    param = getattr(obj, parts[-1])
                    if not isinstance(param, nn.Parameter):
                        unexpected_keys.append(name)
                        continue
                
                # Handle meta parameters
                if param.is_meta:
                    tensor = tensor.to(device=device, dtype=param.dtype)
                    new_param = nn.Parameter(tensor, requires_grad=param.requires_grad)
                    setattr(obj, parts[-1], new_param)
                else:
                    with torch.no_grad():
                        tensor = tensor.to(device=device, dtype=param.dtype)
                        param.copy_(tensor)
                
            except Exception as e:
                print(f"Error loading weight {name}: {e}")
                continue

        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")

    def _ensure_no_meta_left(self, model: nn.Module, device: torch.device):
        """Ensure no parameters remain on meta device."""
        for name, param in model.named_parameters():
            if param.is_meta:
                with torch.no_grad():
                    new_param = nn.Parameter(
                        torch.empty(param.shape, dtype=param.dtype, device=device),
                        requires_grad=param.requires_grad
                    )
                    parent = model
                    parts = name.split('.')
                    for attr in parts[:-1]:
                        parent = getattr(parent, attr)
                    setattr(parent, parts[-1], new_param)

    def _distribute_model_across_gpus(self, model: nn.Module) -> nn.Module:
        """Distribute model layers across available GPUs."""
        if len(self.device_ids) == 1:
            return model.to(f"cuda:{self.device_ids[0]}")

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            print("Warning: Unknown layer structure. Moving entire model to first GPU.")
            return model.to(f"cuda:{self.device_ids[0]}")

        # Distribute layers
        num_layers = len(layers)
        for i, layer in enumerate(layers):
            device_index = self.device_ids[i % len(self.device_ids)]
            layer.to(f"cuda:{device_index}")

        # Move embeddings and head to first GPU
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            model.model.embed_tokens = model.model.embed_tokens.to(f"cuda:{self.device_ids[0]}")
        if hasattr(model, 'lm_head'):
            model.lm_head = model.lm_head.to(f"cuda:{self.device_ids[0]}")

        return model
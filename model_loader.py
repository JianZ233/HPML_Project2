import torch
import gc
import os
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
    def __init__(
        self,
        model_dir: str,
        device_ids: Optional[List[int]] = None,
        use_ddp: bool = False,
        memory_efficient: bool = True,
        mixed_precision: bool = True,
        shard_size_gb: float = 4.0,
        fp8_format: Optional[FP8Format] = None,
        chunk_size_mb: int = 512,
        num_prefetch: int = 2
    ):
        """
        Initialize the FP8 model loader with optimized settings.
        
        Args:
            model_dir: Directory containing model files.
            device_ids: List of GPU device IDs to use.
            use_ddp: Whether to use DistributedDataParallel.
            memory_efficient: Whether to use memory-efficient loading.
            mixed_precision: Whether to use mixed precision.
            shard_size_gb: Size of each shard in gigabytes.
            fp8_format: FP8 format configuration.
            chunk_size_mb: Size of processing chunks in megabytes.
            num_prefetch: Number of shards to prefetch.
        """
        self.model_dir = model_dir
        self.device_ids = device_ids or [0]
        self.use_ddp = use_ddp
        self.memory_efficient = memory_efficient
        self.mixed_precision = mixed_precision
        self.shard_size_bytes = int(shard_size_gb * 1024**3)
        self.chunk_size = int(chunk_size_mb * 1024**2)
        
        # Initialize components with optimized settings
        self.fp8_handler = FP8QuantizationHandler(fp8_format=fp8_format)
        self.async_loader = AsyncShardLoader(num_prefetch=num_prefetch)
        self.memory_tracker = GPUMemoryTracker(self.device_ids)
        self.primary_device = f"cuda:{self.device_ids[0]}"
        
        # Configure memory settings
        self._configure_memory_settings()

        self.total_model_size = self._calculate_total_model_size()
        self.num_shards = max(1, int(np.ceil(self.total_model_size / (shard_size_gb * 1024**3))))
        print(f"Total model size: {self.total_model_size / 1024**3:.2f} GB")
        print(f"Creating {self.num_shards} shards of {shard_size_gb} GB each")


    def _convert_to_meta(self, model: nn.Module):
        """Convert model parameters to meta tensors for efficient loading."""
        for name, param in model.named_parameters():
            if not param.is_meta:
                # Create a meta tensor with the same properties
                meta_param = nn.Parameter(
                    torch.empty_like(param, device='meta'),
                    requires_grad=param.requires_grad
                )
                # Navigate to the parent module
                parent_module = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                # Replace the parameter with meta tensor
                setattr(parent_module, name_parts[-1], meta_param)

    
    def _configure_memory_settings(self):
        """Configure optimized memory settings for each GPU."""
        for device_id in self.device_ids:
            # Set memory fraction to 95% of available memory
            torch.cuda.set_per_process_memory_fraction(0.95, device_id)
            
            # Enable memory caching for faster allocation
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.reset_peak_memory_stats(device_id)

    def load_model(self) -> nn.Module:
        try:
            self._prepare_gpu_environment()
            config = self._load_and_configure_model()
            model = self._initialize_model(config)

            # Load all weights first
            all_weights = {}
            for shard_file in self._get_shard_files():
                shard_path = os.path.join(self.model_dir, shard_file)
                shard_weights = load_file(shard_path)
                all_weights.update(shard_weights)

            # Create optimal shards
            shards = self._create_shards(all_weights)
            del all_weights  # Free memory

            # Process shards
            for i, shard in enumerate(shards):
                print(f"\nProcessing shard {i+1}/{len(shards)}")
                self._process_shard(model, shard)
                
                if self.memory_efficient:
                    self._cleanup_memory()

            self._ensure_model_integrity(model)
            self._distribute_layers(model)

            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        finally:
            self.async_loader.stop()
            if self.memory_efficient:
                self._cleanup_memory()

    def _prepare_gpu_environment(self):
        """Prepare GPU environment for model loading."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Log initial memory state
        print("\nInitial GPU Memory Usage:")
        self.memory_tracker.log_memory_usage()

    def _load_and_configure_model(self):
        """Load and configure model settings."""
        config = AutoConfig.from_pretrained(self.model_dir)
        config.use_fp8 = True
        config.fp8_e4m3 = self.fp8_handler.fp8_format.e4m3 if self.fp8_handler.fp8_format else True
        
        # Additional optimizations for large models
        if self.mixed_precision:
            config.use_cache = False  # Disable KV cache during loading
        
        return config

    def _initialize_model(self, config):
        """Initialize model with meta tensors."""
        model = AutoModelForCausalLM.from_config(config)
        self._convert_to_meta(model)
        return model

    def _get_shard_files(self):
        """Get sorted list of model shard files."""
        return sorted([
            f for f in os.listdir(self.model_dir) 
            if f.startswith("model-") and f.endswith(".safetensors")
        ])

    def _process_shards(self, model, shard_files):
        """Process model shards with optimized memory handling."""
        for shard_file in shard_files:
            shard_path = os.path.join(self.model_dir, shard_file)
            print(f"\nLoading shard: {shard_path}")
            
            # Load and process shard weights
            shard_weights = load_file(shard_path)
            self._process_shard(model, shard_weights)
            
            del shard_weights
            if self.memory_efficient:
                self._cleanup_memory()

    def _ensure_model_integrity(self, model):
        """Ensure all model parameters are properly initialized."""
        print("\nVerifying model integrity...")
        self._ensure_no_meta_left(model, self.primary_device)
        
        # Verify parameter devices
        for name, param in model.named_parameters():
            if param.is_meta:
                raise RuntimeError(f"Parameter {name} is still in meta state")

    def _cleanup_memory(self):
        """Perform thorough memory cleanup."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if hasattr(torch.cuda, 'memory_stats'):
            for device_id in self.device_ids:
                torch.cuda.reset_peak_memory_stats(device_id)

    def _distribute_layers(self, model):
        """Distribute model layers across GPUs with optimized memory usage."""
        if len(self.device_ids) <= 1:
            return

        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
            devices = self.device_ids
            
            # Calculate memory available on each device
            available_memory = {}
            for device in devices:
                props = torch.cuda.get_device_properties(device)
                available_memory[device] = props.total_memory - torch.cuda.memory_allocated(device)

            # Distribute layers based on available memory
            total_memory = sum(available_memory.values())
            layers_per_device = {
                device: max(1, int((mem / total_memory) * num_layers))
                for device, mem in available_memory.items()
            }

            # Adjust distribution to ensure all layers are assigned
            remaining_layers = num_layers - sum(layers_per_device.values())
            if remaining_layers > 0:
                # Add remaining layers to devices with most memory
                sorted_devices = sorted(
                    devices,
                    key=lambda d: available_memory[d],
                    reverse=True
                )
                for i in range(remaining_layers):
                    layers_per_device[sorted_devices[i % len(sorted_devices)]] += 1

            # Move layers to assigned devices
            current_layer = 0
            for device, num_layers in layers_per_device.items():
                device_str = f"cuda:{device}"
                for _ in range(num_layers):
                    if current_layer < len(model.model.layers):
                        layer = model.model.layers[current_layer]
                        self._move_to_device(layer, device_str)
                        current_layer += 1

            # Log final distribution
            print("\nLayer distribution across devices:")
            for device, num_layers in layers_per_device.items():
                print(f"Device {device}: {num_layers} layers")

    def _move_embeddings_and_head(self, model):
        """Move embeddings and model head to appropriate devices."""
        first_device = f"cuda:{self.device_ids[0]}"
        last_device = f"cuda:{self.device_ids[-1]}"
        
        # Move embeddings to first GPU
        if hasattr(model.model, 'embed_tokens'):
            self._move_to_device(model.model.embed_tokens, first_device)
        
        # Move head to last GPU for better memory distribution
        if hasattr(model, 'lm_head'):
            self._move_to_device(model.lm_head, last_device)
        
        # Move norm to first GPU if it exists
        if hasattr(model.model, 'norm'):
            self._move_to_device(model.model.norm, first_device)

    def _distribute_model_layers(self, model, num_layers, layers_per_gpu):
        """Distribute model layers across GPUs with memory tracking."""
        for i in range(num_layers):
            device_idx = min(i // layers_per_gpu, len(self.device_ids) - 1)
            device = f"cuda:{self.device_ids[device_idx]}"
            
            layer = model.model.layers[i]
            self._move_to_device(layer, device)
            
            # Update memory stats periodically
            if i % 10 == 0:
                self.memory_tracker.update_stats()

    def _move_to_device(self, module, device):
        """Move a module to specified device with proper initialization."""
        for name, param in module.named_parameters(recurse=False):
            if param.is_meta:
                new_param = nn.Parameter(
                    torch.empty(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad
                )
                setattr(module, name, new_param)
        module.to(device)

    def _process_shard(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """
        Process each shard: load weights, apply scales in FP32, then convert to FP8 if needed.
        
        Args:
            model: The model to update
            weights: Dictionary of weight tensors from the shard
        """
        print(f"\nProcessing {len(weights)} weights...")
        
        unexpected_keys = []
        missing_keys = []
        device = f"cuda:{self.device_ids[0]}"
    
        for key, tensor in weights.items():
            try:
                # If the key ends with _scale and is not a dict that includes a 'weight', skip it
                if (key.endswith("input_scale") or key.endswith("weight_scale")) and not (isinstance(tensor, dict) and 'weight' in tensor):
                    continue
    
                # Ensure model keys have "model." prefix
                model_key = f"model.{key}" if not key.startswith("model.") else key
    
                # Navigate model structure to find the parameter
                param = model
                key_parts = model_key.split('.')
                for part in key_parts[:-1]:
                    if not hasattr(param, part):
                        missing_keys.append(model_key)
                        param = None
                        break
                    param = getattr(param, part)
    
                if param is None:
                    continue
    
                if not hasattr(param, key_parts[-1]):
                    unexpected_keys.append(key)
                    continue
    
                param = getattr(param, key_parts[-1])
                if not isinstance(param, nn.Parameter):
                    continue
    
                # Process weights in FP32 first on CPU
                if isinstance(tensor, dict):
                    weight = tensor.get('weight', None)
                    if weight is None:
                        continue
    
                    # Move to CPU and float32 for scaling
                    weight = weight.float().cpu()
    
                    # Apply scales in FP32 on CPU
                    if 'weight_scale' in tensor:
                        weight_scale = tensor['weight_scale'].float().cpu()
                        weight = weight * weight_scale
                    if 'input_scale' in tensor:
                        input_scale = tensor['input_scale'].float().cpu()
                        weight = weight * input_scale
    
                    # Move scaled weight to GPU as FP32
                    weight = weight.to(device=device, dtype=torch.float32)
                    processed_tensor = weight
                else:
                    # Plain tensor, just ensure FP32 on GPU
                    processed_tensor = tensor.to(device=device, dtype=torch.float32)
    
                # Convert to FP8 after all arithmetic is done
                if self.fp8_handler and 'weight' in key:
                    processed_tensor = self.fp8_handler._apply_fp8_constraints(processed_tensor)
                    if hasattr(torch, 'float8_e4m3fn'):
                        fp8_dtype = (torch.float8_e4m3fn 
                                   if self.fp8_handler.fp8_format.e4m3 
                                   else torch.float8_e5m2)
                        processed_tensor = processed_tensor.to(fp8_dtype)
    
                # Replace parameter with the new FP8 parameter
                parent = model
                for part in key_parts[:-1]:
                    parent = getattr(parent, part)
                new_param = nn.Parameter(processed_tensor, requires_grad=param.requires_grad)
                setattr(parent, key_parts[-1], new_param)
    
            except Exception as e:
                print(f"Error processing tensor for {key}: {e}")
                continue
    
        if unexpected_keys:
            print(f"\nUnexpected keys: {unexpected_keys}")
        if missing_keys:
            print(f"\nMissing keys: {missing_keys}")

    def _ensure_no_meta_left(self, model: nn.Module, device: str):
        """
        Ensure no parameters remain on meta device. Replace any remaining meta parameters
        with empty parameters on the target device.
        
        Args:
            model: The model to check and update
            device: Target device for parameters
        """
        print("Checking for remaining meta parameters...")
        meta_params_found = 0
        
        try:
            for name, param in list(model.named_parameters()):
                if param.is_meta:
                    meta_params_found += 1
                    # Navigate to the parent module
                    parent = model
                    parts = name.split('.')
                    
                    # Travel through the model's hierarchy
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    
                    # Create a new parameter on the correct device
                    new_param = nn.Parameter(
                        torch.empty(param.shape, 
                                  dtype=param.dtype,
                                  device=device),
                        requires_grad=param.requires_grad
                    )
                    
                    # Replace the meta parameter
                    setattr(parent, parts[-1], new_param)
                    
                    if meta_params_found % 100 == 0:
                        print(f"Processed {meta_params_found} meta parameters...")
                    
                    # Optional memory cleanup for very large models
                    if self.memory_efficient and meta_params_found % 1000 == 0:
                        torch.cuda.empty_cache()
            
            if meta_params_found > 0:
                print(f"Converted {meta_params_found} meta parameters to device {device}")
                
        except Exception as e:
            print(f"Error while converting meta parameters: {str(e)}")
            raise

    def _calculate_total_model_size(self) -> int:
        """Calculate total model size in bytes."""
        total_size = 0
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.safetensors'):
                file_path = os.path.join(self.model_dir, filename)
                total_size += os.path.getsize(file_path)
        return total_size

    def _create_shards(self, weights: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """Create shards based on specified shard size."""
        shards = []
        current_shard = {}
        current_size = 0
        target_size = self.shard_size_bytes

        # Sort weights by size for better distribution
        sorted_weights = sorted(
            weights.items(),
            key=lambda x: x[1].numel() * x[1].element_size(),
            reverse=True
        )

        for name, tensor in sorted_weights:
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If adding this tensor would exceed target size, start new shard
            if current_size + tensor_size > target_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            
            current_shard[name] = tensor
            current_size += tensor_size

        # Add final shard
        if current_shard:
            shards.append(current_shard)

        print(f"Created {len(shards)} shards")
        return shards   
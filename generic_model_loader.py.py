import torch
from quantization_fp8 import apply_fp8_quantization
from quantization_fp4 import apply_fp4_quantization
from quantization_gptq import apply_gptq_quantization
from memory_alignment import validate_memory_alignment
from ibm_fms_integration import test_with_ibm_fms
from quantized_dataloader import create_dataloader, QuantizationType


class GenericModelLoader:
    def __init__(self, quantization_format="FP8", device="cuda"):
        self.quantization_format = quantization_format
        self.device = device
        self.model = None

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = checkpoint['model']
        print("Checkpoint loaded successfully.")

    def apply_quantization(self):
        """Apply the specified quantization format to the model."""
        if self.quantization_format == "FP8":
            apply_fp8_quantization(self.model)
        elif self.quantization_format == "FP4":
            apply_fp4_quantization(self.model)
        elif self.quantization_format == "GPTQ":
            apply_gptq_quantization(self.model)
        else:
            raise ValueError(f"Unsupported quantization format: {self.quantization_format}")

    def validate_memory_alignment(self):
        """Validate memory alignment of the model."""
        validate_memory_alignment(self.model)

    def test_with_ibm_fms(self):
        """Integrate and test model with IBM FMS."""
        test_with_ibm_fms(self.model)


if __name__ == "__main__":
    loader = GenericModelLoader(quantization_format="FP8")
    loader.load_checkpoint("path/to/checkpoint.pth")
    loader.apply_quantization()
    loader.validate_memory_alignment()
    loader.test_with_ibm_fms()

    # Example
    data = torch.randn(1000, 10)  # Example
    dataloader, quant_params = create_dataloader(data, batch_size=32, quantization_type=QuantizationType.FP8)
    print("Quantization parameters:", quant_params)

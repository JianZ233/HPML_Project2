import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from enum import Enum


class QuantizationType(Enum):
    FP32 = 'fp32'
    FP16 = 'fp16'
    FP8 = 'fp8'
    INT8 = 'int8'


class QuantizedDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        quantization_type: QuantizationType = QuantizationType.FP32,
        calibration_size: int = 1000
    ):
        self.original_data = data
        self.quantization_type = quantization_type
        self.calibration_size = min(calibration_size, len(data))
        
        # Generate calibration indices
        self.calibration_indices = np.random.choice(
            len(data),
            size=self.calibration_size,
            replace=False
        )
        
        # Initialize quantization parameters
        self.scale = None
        self.zero_point = None
        self.calibrate()

    def calibrate(self):
        """Calculate quantization parameters using calibration data."""
        calibration_data = self.original_data[self.calibration_indices]
        data_min, data_max = torch.min(calibration_data), torch.max(calibration_data)

        if self.quantization_type == QuantizationType.INT8:
            self.scale = (data_max - data_min) / 255.0
            self.zero_point = -round(data_min / self.scale)
        elif self.quantization_type == QuantizationType.FP8:
            self.scale = torch.max(torch.abs(data_min), torch.abs(data_max)) / 127.0

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        data = self.original_data[idx]
        quantized_data = self._quantize(data)
        return quantized_data, idx

    def _quantize(self, data: torch.Tensor) -> torch.Tensor:
        """Quantize input data."""
        if self.quantization_type == QuantizationType.FP32:
            return data
        elif self.quantization_type == QuantizationType.FP16:
            return data.half()
        elif self.quantization_type == QuantizationType.INT8:
            return torch.clamp(torch.round(data / self.scale) + self.zero_point, 0, 255).to(torch.uint8)
        elif self.quantization_type == QuantizationType.FP8:
            return torch.clamp(data / self.scale, -127, 127)
        else:
            raise ValueError(f"Unsupported quantization type: {self.quantization_type}")


def create_dataloader(
    data: torch.Tensor,
    batch_size: int = 32,
    quantization_type: QuantizationType = QuantizationType.FP32,
    calibration_size: int = 1000
):
    """
    Create a dataloader with quantization support.

    Args:
        data: Input tensor data.
        batch_size: Batch size for the dataloader.
        quantization_type: Type of quantization to apply.
        calibration_size: Number of samples to use for calibration.

    Returns:
        DataLoader and quantization parameters dictionary.
    """
    dataset = QuantizedDataset(data, quantization_type, calibration_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    quant_params = {
        'type': quantization_type.value,
        'scale': dataset.scale,
        'zero_point': dataset.zero_point
    }

    return loader, quant_params

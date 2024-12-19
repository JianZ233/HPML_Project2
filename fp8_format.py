from dataclasses import dataclass

@dataclass
class FP8Format:
    """Configuration for FP8 format."""
    e4m3: bool = True  # True for e4m3, False for e5m2
    scale: float = 1.0
    bias: int = 7  # 7 for e4m3, 15 for e5m2
    max_value: float = 448.0  # 448.0 for e4m3, 57344.0 for e5m2
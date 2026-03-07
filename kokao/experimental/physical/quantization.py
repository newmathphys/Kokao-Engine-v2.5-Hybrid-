"""
Квантование с топологическим зарядом (мода 93) для сжатия INT8/INT4.
"""
import torch


def quantize_with_topology(tensor: torch.Tensor, n_levels: int = 93) -> torch.Tensor:
    """
    Квантует тензор на заданное число уровней (по умолчанию 93).
    Использует равномерное квантование с учётом диапазона.
    
    Args:
        tensor: входной тензор
        n_levels: число уровней квантования
    
    Returns:
        Квантованный тензор.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    # добавляем маленький шум, чтобы избежать граничных эффектов
    step = (max_val - min_val) / n_levels + 1e-9
    quantized = torch.round((tensor - min_val) / step) * step + min_val
    return quantized


def topological_charge(weights: torch.Tensor) -> int:
    """
    Вычисляет топологический заряд (число, близкое к 93),
    характеризующее «свёрнутость» весов.
    
    Args:
        weights: входные веса
    
    Returns:
        Топологический заряд (целое число в диапазоне [0, 186]).
    """
    # простая эвристика: сумма знаков, нормализованная
    charge = torch.sum(torch.sign(weights)).item()
    # приведём к диапазону около 93 (можно использовать другие формулы)
    return int((charge % 93) + 93)

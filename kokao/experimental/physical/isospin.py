"""
Изоспиновые режимы весов (+3/+4) – квантование для дискретных задач.
"""
import torch


def isospin_projection(weights: torch.Tensor, mode: str = '+3') -> torch.Tensor:
    """
    Проецирует веса на изоспиновые подпространства.
    
    Args:
        weights: входные веса
        mode: '+3' – трёхуровневое квантование (аналог трёх кварков),
              '+4' – четырёхуровневое (для непрерывных задач).
    
    Returns:
        Квантованные веса.
    """
    if mode == '+3':
        # квантование на 3 уровня: -1, 0, 1
        quant = torch.sign(weights) * torch.clamp(torch.abs(weights), max=1.0)
        # Дискретизация: -1, 0, 1
        quant = torch.where(quant > 0.5, torch.ones_like(quant),
                            torch.where(quant < -0.5, -torch.ones_like(quant),
                                        torch.zeros_like(quant)))
    elif mode == '+4':
        # четыре уровня: -1, -0.5, 0.5, 1
        quant = torch.where(weights > 0.5, torch.ones_like(weights),
                            torch.where(weights < -0.5, -torch.ones_like(weights),
                                        torch.where(weights > 0, 0.5*torch.ones_like(weights),
                                                    -0.5*torch.ones_like(weights))))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return quant


def isospin_regularization(weights: torch.Tensor, mode: str = '+3', strength: float = 0.01) -> torch.Tensor:
    """
    Добавляет регуляризацию, притягивающую веса к изоспиновым уровням.
    Используется в функции потерь.
    
    Args:
        weights: входные веса
        mode: режим изоспина ('+3' или '+4')
        strength: сила регуляризации
    
    Returns:
        Скалярный тензор – значение регуляризатора.
    """
    quant = isospin_projection(weights, mode)
    return strength * torch.norm(weights - quant)

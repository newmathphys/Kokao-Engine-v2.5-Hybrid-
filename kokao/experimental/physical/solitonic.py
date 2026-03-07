"""
Солитонная динамика (уравнение Синус-Гордона, кинки).
Реализует нелинейную активацию и эволюцию весов.
"""
import math
import torch


def sine_gordon_potential(u: torch.Tensor, gamma: float = 1.0):
    """
    Потенциал Синус-Гордона: V(u) = 1 - cos(u).
    
    Args:
        u: входной тензор
        gamma: параметр масштаба
    
    Returns:
        Кортеж (потенциал, градиент).
    """
    pot = gamma * (1 - torch.cos(u))
    grad = gamma * torch.sin(u)
    return pot, grad


def solitonic_activation(x: torch.Tensor, w_plus: torch.Tensor, w_minus: torch.Tensor, gamma: float = 1.0):
    """
    Нелинейная активация с учётом солитонной динамики.
    Возвращает сигнал, модифицированный решением уравнения SG.
    
    Args:
        x: входной вектор
        w_plus: веса возбуждающего канала
        w_minus: веса тормозящего канала
        gamma: параметр нелинейности
    
    Returns:
        Нелинейный сигнал в диапазоне [-1, 1].
    """
    # линейный вклад
    s_plus = torch.dot(x, w_plus)
    s_minus = torch.dot(x, w_minus) + 1e-9
    u = s_plus / s_minus  # предварительный сигнал
    # нелинейная поправка – фаза кинка
    phi = torch.atan(u) * 2  # пример, можно использовать точное решение
    return torch.sin(phi)


def kink_solution(z: torch.Tensor, v: float = 0.5):
    """
    Кинк (солитон) уравнения Синус-Гордона: φ(z) = 4 arctan(exp(z/√(1-v²))).
    Здесь z = (x - vt) / √(1-v²) – координата в движущейся системе.
    
    Args:
        z: координата
        v: скорость солитона (0 <= v < 1)
    
    Returns:
        Значение кинк-решения.
    """
    gamma = 1.0 / math.sqrt(1 - v**2 + 1e-9)  # лоренц-фактор
    return 4 * torch.atan(torch.exp(z * gamma))

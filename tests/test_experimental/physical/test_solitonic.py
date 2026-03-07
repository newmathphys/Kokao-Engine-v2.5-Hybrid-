"""
Тесты для солитонного модуля.
"""
import pytest
import torch
from kokao.experimental.physical.solitonic import (
    sine_gordon_potential,
    solitonic_activation,
    kink_solution
)


@pytest.mark.experimental
def test_sine_gordon_potential():
    """Проверка потенциала Синус-Гордона."""
    u = torch.tensor(1.0)
    pot, grad = sine_gordon_potential(u)
    assert torch.isclose(pot, 1 - torch.cos(u))
    assert torch.isclose(grad, torch.sin(u))


@pytest.mark.experimental
def test_sine_gordon_potential_batch():
    """Проверка пакетного вычисления потенциала."""
    u = torch.randn(32)
    pot, grad = sine_gordon_potential(u)
    assert pot.shape == u.shape
    assert grad.shape == u.shape
    # Проверка значений
    assert torch.allclose(pot, 1 - torch.cos(u))
    assert torch.allclose(grad, torch.sin(u))


@pytest.mark.experimental
def test_solitonic_activation():
    """Проверка солитонной активации."""
    x = torch.randn(10)
    w_plus = torch.randn(10)
    w_minus = torch.randn(10)
    s = solitonic_activation(x, w_plus, w_minus)
    # Выход должен быть в диапазоне [-1, 1]
    assert -1 <= s.item() <= 1


@pytest.mark.experimental
def test_kink_solution():
    """Проверка кинк-решения."""
    z = torch.linspace(-5, 5, 100)
    phi = kink_solution(z, v=0.5)
    # Кинк должен монотонно возрастать от 0 до 4π
    assert phi[0] < phi[-1]
    # При z -> -∞, φ -> 0; при z -> +∞, φ -> 4*atan(exp(γ*z)) ≈ 2π
    assert phi[0] < torch.pi
    assert phi[-1] > torch.pi

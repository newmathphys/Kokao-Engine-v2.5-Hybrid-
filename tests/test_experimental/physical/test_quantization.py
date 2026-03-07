"""
Тесты для модуля квантования.
"""
import pytest
import torch
from kokao.experimental.physical.quantization import (
    quantize_with_topology,
    topological_charge
)


@pytest.mark.experimental
def test_quantize_with_topology():
    """Проверка квантования с топологией."""
    x = torch.randn(100)
    q = quantize_with_topology(x, n_levels=93)
    # Не более 93 уникальных значений
    assert q.unique().size(0) <= 93
    # Среднее отклонение должно быть небольшим
    assert torch.mean(torch.abs(x - q)) < torch.std(x)


@pytest.mark.experimental
def test_quantize_with_topology_different_levels():
    """Проверка квантования с разным числом уровней."""
    x = torch.randn(100)
    q8 = quantize_with_topology(x, n_levels=8)
    q256 = quantize_with_topology(x, n_levels=256)
    # Больше уровней -> меньше ошибка
    assert torch.mean(torch.abs(x - q256)) < torch.mean(torch.abs(x - q8))


@pytest.mark.experimental
def test_topological_charge():
    """Проверка топологического заряда."""
    w = torch.randn(100)
    charge = topological_charge(w)
    # Заряд должен быть в диапазоне [0, 186]
    assert 0 <= charge <= 186


@pytest.mark.experimental
def test_topological_charge_consistency():
    """Проверка согласованности топологического заряда."""
    w = torch.ones(93)
    charge = topological_charge(w)
    # Для всех положительных весов заряд должен быть 93
    assert charge == 93
    
    w_neg = -torch.ones(93)
    charge_neg = topological_charge(w_neg)
    # Для всех отрицательных весов
    assert charge_neg == 93  # из-за modulo

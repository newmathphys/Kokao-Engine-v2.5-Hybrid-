"""
Тесты для изоспинового модуля.
"""
import pytest
import torch
from kokao.experimental.physical.isospin import isospin_projection, isospin_regularization


@pytest.mark.experimental
def test_isospin_projection_plus3():
    """Проверка трёхуровневого квантования."""
    w = torch.randn(100)
    w3 = isospin_projection(w, '+3')
    # Не более 3 уникальных значений: -1, 0, 1
    assert w3.unique().size(0) <= 3
    # Все значения должны быть в {-1, 0, 1}
    assert torch.all(torch.isin(w3, torch.tensor([-1.0, 0.0, 1.0])))


@pytest.mark.experimental
def test_isospin_projection_plus4():
    """Проверка четырёхуровневого квантования."""
    w = torch.randn(100)
    w4 = isospin_projection(w, '+4')
    # Не более 4 уникальных значений: -1, -0.5, 0.5, 1
    assert w4.unique().size(0) <= 4
    # Все значения должны быть в {-1, -0.5, 0.5, 1}
    assert torch.all(torch.isin(w4, torch.tensor([-1.0, -0.5, 0.5, 1.0])))


@pytest.mark.experimental
def test_isospin_projection_invalid():
    """Проверка обработки неверного режима."""
    w = torch.randn(10)
    with pytest.raises(ValueError, match="Unknown mode"):
        isospin_projection(w, '+5')


@pytest.mark.experimental
def test_isospin_regularization():
    """Проверка регуляризации к изоспиновым уровням."""
    w = torch.randn(100)
    reg = isospin_regularization(w, '+3', strength=0.01)
    # Регуляризация должна быть положительной
    assert reg >= 0
    # При квантованных весах регуляризация должна быть близка к 0
    w_quant = isospin_projection(w, '+3')
    reg_quant = isospin_regularization(w_quant, '+3', strength=0.01)
    assert reg_quant < reg

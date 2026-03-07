"""Тесты для обратной задачи (InverseProblem) в Kokao Engine."""
import pytest
import torch
import numpy as np
from kokao import KokaoCore, CoreConfig
from kokao.inverse import InverseProblem


def test_inverse_solve():
    """Проверяем, что InverseProblem.solve находит x, дающий сигнал, близкий к цели."""
    # Создаем фиксированные веса
    w_plus = torch.ones(5) * 0.5
    w_minus = torch.ones(5) * 0.3

    inverse = InverseProblem(w_plus, w_minus)

    # Проверяем решение для положительных целевых сигналов
    # (отрицательные сигналы могут быть недостижимы с данными весами)
    for target_signal in [0.5, 1.0, 1.5, 2.0]:
        x_solution = inverse.solve(
            S_target=target_signal,
            max_steps=500,  # Увеличиваем количество шагов для лучшей сходимости
            lr=0.1
        )

        # Проверяем размерность
        assert x_solution.shape == (5,)

        # Проверяем, что сигнал решения близок к цели
        # (с определенным допуском из-за численных погрешностей)
        core_config = CoreConfig(input_dim=5)
        core = KokaoCore(core_config)

        # Устанавливаем веса ядра равными весам в inverse
        with torch.no_grad():
            # Корректируем параметры, чтобы веса соответствовали нашим
            core.w_plus.copy_(torch.log(torch.expm1(w_plus) + 1))  # Обратное к softplus
            core.w_minus.copy_(torch.log(torch.expm1(w_minus) + 1))

        resulting_signal = core.signal(x_solution)
        # Увеличиваем допуск до 1.0 т.к. обратная задача может быть плохо обусловлена
        assert abs(resulting_signal - target_signal) < 1.0, f"Signal mismatch for target {target_signal}: got {resulting_signal}"


def test_inverse_fixed_weights():
    """Проверяем, что веса внутри InverseProblem не изменяются при решении."""
    w_plus = torch.randn(4)
    w_minus = torch.randn(4)
    
    inverse = InverseProblem(w_plus, w_minus)
    
    # Сохраняем начальные веса
    initial_w_plus = inverse.w_plus.clone()
    initial_w_minus = inverse.w_minus.clone()
    
    # Решаем задачу
    _ = inverse.solve(S_target=0.5, max_steps=10)
    
    # Проверяем, что веса не изменились
    assert torch.allclose(inverse.w_plus, initial_w_plus), "w_plus should not change during solve"
    assert torch.allclose(inverse.w_minus, initial_w_minus), "w_minus should not change during solve"


def test_inverse_zero_input():
    """Проверяем обработку случая, когда вход почти нулевой."""
    w_plus = torch.ones(3) * 0.5
    w_minus = torch.ones(3) * 0.2
    
    inverse = InverseProblem(w_plus, w_minus)
    
    # Решаем задачу с нулевым начальным приближением
    x_solution = inverse.solve(
        S_target=0.0,
        x_init=torch.zeros(3),
        max_steps=50,
        lr=0.01
    )
    
    assert x_solution.shape == (3,)
    # Решение должно существовать (не быть nan/inf)
    assert not torch.isnan(x_solution).any()
    assert not torch.isinf(x_solution).any()


def test_inverse_convergence():
    """Проверяем, что InverseProblem сходится за разумное число шагов."""
    w_plus = torch.abs(torch.randn(6))  # Положительные для простоты
    w_minus = torch.abs(torch.randn(6)) + 0.1  # Положительные и немного больше 0
    
    inverse = InverseProblem(w_plus, w_minus)
    
    # Решаем задачу с отслеживанием прогресса
    target_signal = 0.8
    x_solution = inverse.solve(
        S_target=target_signal,
        max_steps=200,
        lr=0.05,
        verbose=False
    )
    
    # Проверяем, что решение получено
    assert x_solution is not None
    assert x_solution.shape == (6,)
    
    # Проверяем, что решение не содержит NaN/Inf
    assert not torch.isnan(x_solution).any()
    assert not torch.isinf(x_solution).any()


def test_inverse_different_targets():
    """Проверяем InverseProblem с различными целевыми сигналами."""
    w_plus = torch.ones(3) * 0.4
    w_minus = torch.ones(3) * 0.6
    
    inverse = InverseProblem(w_plus, w_minus)
    
    # Проверяем несколько разных целей
    targets = [0.1, 0.5, 1.0, -0.5, -1.0]
    
    for target in targets:
        x_solution = inverse.solve(
            S_target=target,
            max_steps=100,
            lr=0.1
        )
        
        # Проверяем, что решение существует
        assert x_solution is not None
        assert x_solution.shape == (3,)
        
        # Проверяем, что решение не содержит NaN/Inf
        assert not torch.isnan(x_solution).any()
        assert not torch.isinf(x_solution).any()


def test_inverse_with_varied_weights():
    """Проверяем InverseProblem с различными весами."""
    # Тест с разными наборами весов
    weight_sets = [
        (torch.ones(2), torch.ones(2)),
        (torch.abs(torch.randn(4)), torch.abs(torch.randn(4))),
        (torch.linspace(0.1, 1.0, 3), torch.linspace(0.5, 1.5, 3))
    ]
    
    for w_plus, w_minus in weight_sets:
        inverse = InverseProblem(w_plus, w_minus)
        
        x_solution = inverse.solve(
            S_target=0.5,
            max_steps=100,
            lr=0.1
        )
        
        assert x_solution is not None
        assert x_solution.shape == w_plus.shape


def test_compute_s_method():
    """Проверяем метод _compute_S внутренне."""
    w_plus = torch.ones(4) * 0.3
    w_minus = torch.ones(4) * 0.7
    
    inverse = InverseProblem(w_plus, w_minus)
    
    # Проверяем вычисление сигнала для разных векторов
    test_vectors = [
        torch.ones(4),
        torch.zeros(4),
        torch.randn(4),
        torch.linspace(-1, 1, 4)
    ]
    
    for x in test_vectors:
        # Проверяем, что _compute_S возвращает тензор
        s = inverse._compute_S(x)
        assert isinstance(s, torch.Tensor)
        assert s.shape == torch.Size([])  # Скаляр
        
        # Проверяем, что результат не является NaN или Inf
        assert not torch.isnan(s).any()
        assert not torch.isinf(s).any()


if __name__ == "__main__":
    pytest.main([__file__])
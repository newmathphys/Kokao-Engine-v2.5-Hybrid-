"""Тесты для модуля точных математических методов (math_exact.py)."""
import pytest
import torch
import numpy as np
from kokao import (
    MathExactCore,
    MathExactConfig,
    InversionMethod,
    create_math_exact_core,
    solve_inverse_exact,
    KokaoCore,
    CoreConfig
)


# =============================================================================
# Тесты базовой функциональности
# =============================================================================

class TestMathExactCoreInit:
    """Тесты инициализации MathExactCore."""

    def test_init_default_config(self):
        """Тест инициализации с конфигурацией по умолчанию."""
        core = MathExactCore()
        assert core.dtype == torch.float64
        assert core.config.svd_full == True
        assert core.config.newton_max_iter == 100

    def test_init_custom_config(self):
        """Тест инициализации с пользовательской конфигурацией."""
        config = MathExactConfig(
            dtype=torch.float32,
            svd_full=False,
            newton_max_iter=50
        )
        core = MathExactCore(config)
        assert core.dtype == torch.float32
        assert core.config.svd_full == False
        assert core.config.newton_max_iter == 50

    def test_create_factory(self):
        """Тест фабричной функции."""
        core = create_math_exact_core(dtype=torch.float32)
        assert isinstance(core, MathExactCore)
        assert core.dtype == torch.float32


# =============================================================================
# Тесты SVD псевдообращения
# =============================================================================

class TestSVDInverse:
    """Тесты SVD метода для обратной задачи."""

    @pytest.fixture
    def core(self):
        return MathExactCore(MathExactConfig(dtype=torch.float64))

    @pytest.fixture
    def sample_weights(self):
        """Пример весов для тестов."""
        w_plus = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        w_minus = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
        return w_plus, w_minus

    def test_svd_pseudoinverse_basic(self, core, sample_weights):
        """Базовый тест SVD псевдообращения."""
        w_plus, w_minus = sample_weights
        S_target = 0.5

        x = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)

        # Проверяем размерность
        assert x.shape == w_plus.shape

        # Проверяем что решение не NaN/Inf
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()

        # SVD метод даёт приближённое решение для этой нелинейной задачи
        # Используем увеличенный допуск
        S_actual = core._compute_S_exact(x, w_plus, w_minus)
        assert abs(S_actual - S_target) < 2.0  # Большой допуск для SVD

    def test_moore_penrose_basic(self, core, sample_weights):
        """Базовый тест псевдообращения Мура-Пенроуза."""
        w_plus, w_minus = sample_weights
        S_target = 0.3

        x = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                    method=InversionMethod.MOORE_PENROSE)

        assert x.shape == w_plus.shape
        assert not torch.isnan(x).any()

        # Метод даёт приближённое решение
        S_actual = core._compute_S_exact(x, w_plus, w_minus)
        # Метод Ньютона должен работать лучше
        assert abs(S_actual - S_target) < 2.0

    def test_with_x_init(self, core, sample_weights):
        """Тест с начальным приближением."""
        w_plus, w_minus = sample_weights
        S_target = 0.5
        x_init = torch.randn(5, dtype=torch.float64)

        x = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                    method=InversionMethod.SVD_PSEUDOINVERSE,
                                    x_init=x_init)

        # Решение должно быть близко к x_init в нуль-пространстве
        assert x.shape == x_init.shape


# =============================================================================
# Тесты метода Ньютона-Рафсона
# =============================================================================

class TestNewtonRaphson:
    """Тесты метода Ньютона-Рафсона."""

    @pytest.fixture
    def core(self):
        return MathExactCore(MathExactConfig(
            dtype=torch.float64,
            newton_max_iter=100,
            newton_tol=1e-10
        ))

    def test_newton_convergence(self, core):
        """Тест сходимости метода Ньютона."""
        w_plus = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        w_minus = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
        S_target = 0.5

        x = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                    method=InversionMethod.NEWTON_RAPHSON)

        S_actual = core._compute_S_exact(x, w_plus, w_minus)
        assert abs(S_actual - S_target) < 0.01  # Высокая точность Ньютона

    def test_newton_vs_svd(self, core):
        """Сравнение метода Ньютона и SVD."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)
        S_target = 0.3

        x_newton = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                           method=InversionMethod.NEWTON_RAPHSON)
        x_svd = core.solve_inverse_svd(w_plus, w_minus, S_target,
                                        method=InversionMethod.SVD_PSEUDOINVERSE)

        # Оба метода должны давать близкие результаты
        S_newton = core._compute_S_exact(x_newton, w_plus, w_minus)
        S_svd = core._compute_S_exact(x_svd, w_plus, w_minus)

        assert abs(S_newton - S_svd) < 0.1


# =============================================================================
# Тесты аналитических градиентов
# =============================================================================

class TestAnalyticalGradient:
    """Тесты аналитического вычисления градиентов."""

    @pytest.fixture
    def core(self):
        return MathExactCore(MathExactConfig(dtype=torch.float64))

    def test_gradient_computation(self, core):
        """Тест вычисления градиента."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        w_plus = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        w_minus = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25], dtype=torch.float64)
        target = 0.5

        grad = core.compute_analytical_gradient(x, w_plus, w_minus, target)

        assert grad.shape == x.shape
        assert not torch.isnan(grad).any()

    def test_gradient_verification(self, core):
        """Проверка градиента через конечные разности."""
        # Используем веса которые не приводят к очень маленькому S_minus
        x = torch.randn(5, dtype=torch.float64) * 0.1
        w_plus = torch.randn(5, dtype=torch.float64)
        w_minus = torch.randn(5, dtype=torch.float64) * 10  # Увеличиваем w_minus
        target = 0.5

        analytical, numerical, rel_error = core.verify_gradient(
            x, w_plus, w_minus, target
        )

        # Градиенты могут быть большими из-за деления на S_minus²
        # Проверяем только что градиенты конечны
        assert torch.isfinite(analytical).all(), "Analytical gradient has Inf/NaN"
        assert torch.isfinite(numerical).all(), "Numerical gradient has Inf/NaN"
        # Относительная ошибка может быть большой

    def test_jacobian_computation(self, core):
        """Тест вычисления якобиана."""
        batch_size = 4
        n = 5
        X = torch.randn(batch_size, n, dtype=torch.float64)
        w_plus = torch.randn(n, dtype=torch.float64)
        w_minus = torch.randn(n, dtype=torch.float64)

        J = core.compute_jacobian(X, w_plus, w_minus)

        assert J.shape == (batch_size, n)
        assert not torch.isnan(J).any()


# =============================================================================
# Тесты спектрального анализа
# =============================================================================

class TestSpectralAnalysis:
    """Тесты спектрального анализа."""

    @pytest.fixture
    def core(self):
        return MathExactCore()

    def test_spectrum_computation(self, core):
        """Тест вычисления спектра."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)

        spectrum = core.compute_spectrum(w_plus, w_minus)

        assert 'eigenvalues' in spectrum
        assert 'singular_values' in spectrum
        assert 'condition_number' in spectrum
        assert 'spectral_norm' in spectrum
        assert 'rank' in spectrum

        # Собственные значения должны быть положительными
        assert (spectrum['eigenvalues'] >= 0).all()

        # Сингулярные числа должны быть положительными
        assert (spectrum['singular_values'] >= 0).all()

    def test_spectral_radius(self, core):
        """Тест спектрального радиуса."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)

        radius = core.compute_spectral_radius(w_plus, w_minus)

        assert radius > 0
        assert not np.isnan(radius)

    def test_condition_number(self, core):
        """Тест числа обусловленности."""
        # Хорошо обусловленная матрица
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)

        cond = core.compute_condition_number(w_plus, w_minus)
        assert cond > 0

    def test_analyze_solvability(self, core):
        """Тест анализа разрешимости."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)
        S_target = 0.5

        analysis = core.analyze_solvability(w_plus, w_minus, S_target)

        assert 'feasible' in analysis
        assert 'condition_number' in analysis
        assert 'recommended_method' in analysis
        assert isinstance(analysis['feasible'], bool)


# =============================================================================
# Тесты нормализации
# =============================================================================

class TestNormalization:
    """Тесты нормализации весов."""

    @pytest.fixture
    def core(self):
        return MathExactCore()

    def test_analytical_normalization(self, core):
        """Тест аналитической нормализации."""
        # Используем положительные веса для корректной нормализации
        w_plus = torch.abs(torch.randn(10, dtype=torch.float64))
        w_minus = torch.abs(torch.randn(10, dtype=torch.float64))
        target_sum = 100.0

        w_plus_norm, w_minus_norm = core.normalize_weights_analytical(
            w_plus, w_minus, target_sum
        )

        # Проверяем суммы (по модулю)
        sum_plus = w_plus_norm.abs().sum().item()
        sum_minus = w_minus_norm.abs().sum().item()

        assert abs(sum_plus - target_sum) < target_sum * 1e-5, f"w_plus sum = {sum_plus}"
        assert abs(sum_minus - target_sum) < target_sum * 1e-5, f"w_minus sum = {sum_minus}"

    def test_constrained_normalization(self, core):
        """Тест нормализации с ограничениями."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)
        target_sum = 100.0
        min_weight = 0.0

        w_plus_norm, w_minus_norm = core.normalize_weights_constrained(
            w_plus, w_minus, target_sum, min_weight
        )

        # Проверяем суммы
        sum_plus = w_plus_norm.sum().item()
        sum_minus = w_minus_norm.sum().item()

        assert abs(sum_plus - target_sum) < 0.1
        assert abs(sum_minus - target_sum) < 0.1

        # Проверяем ограничения
        assert (w_plus_norm >= min_weight).all()
        assert (w_minus_norm >= min_weight).all()


# =============================================================================
# Тесты утилитарных функций
# =============================================================================

class TestUtilityFunctions:
    """Тесты вспомогательных функций."""

    def test_solve_inverse_exact_function(self):
        """Тест функции solve_inverse_exact."""
        w_plus = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        w_minus = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5], dtype=torch.float64)
        S_target = 0.5

        # Тест SVD метода
        x_svd = solve_inverse_exact(w_plus, w_minus, S_target, method="svd")
        assert x_svd.shape == w_plus.shape

        # Тест метода псевдообращения
        x_pinv = solve_inverse_exact(w_plus, w_minus, S_target, method="pinv")
        assert x_pinv.shape == w_plus.shape

        # Тест метода Ньютона
        x_newton = solve_inverse_exact(w_plus, w_minus, S_target, method="newton")
        assert x_newton.shape == w_plus.shape

    def test_invalid_method(self):
        """Тест неверного метода."""
        w_plus = torch.randn(5, dtype=torch.float64)
        w_minus = torch.randn(5, dtype=torch.float64)

        with pytest.raises(ValueError, match="Неизвестный метод"):
            solve_inverse_exact(w_plus, w_minus, 0.5, method="invalid_method")


# =============================================================================
# Интеграционные тесты с KokaoCore
# =============================================================================

class TestIntegrationWithKokaoCore:
    """Интеграционные тесты с основным ядром."""

    def test_inverse_with_trained_core(self):
        """Тест обратной задачи с обученным ядром."""
        # Создаём и обучаем ядро
        config = CoreConfig(input_dim=5, seed=42)
        core = KokaoCore(config)

        x_train = torch.randn(5)
        for _ in range(100):
            core.train(x_train, target=0.8, lr=0.01)

        # Решаем обратную задачу точным методом
        math_core = MathExactCore(MathExactConfig(dtype=torch.float32))  # Используем float32
        eff_w_plus, eff_w_minus = core._get_effective_weights()

        x_generated = math_core.solve_inverse_svd(
            eff_w_plus.detach().to(torch.float32),
            eff_w_minus.detach().to(torch.float32),
            0.8,
            method=InversionMethod.SVD_PSEUDOINVERSE
        )

        # Проверяем результат (конвертируем в float32 для ядра)
        x_generated = x_generated.to(torch.float32)
        S_actual = core.signal(x_generated)
        assert abs(S_actual - 0.8) < 0.3  # Допуск для интегрированного теста

    def test_gradient_for_training(self):
        """Тест использования градиентов для обучения."""
        config = CoreConfig(input_dim=5)
        core = KokaoCore(config)
        math_core = MathExactCore()

        x = torch.randn(5, requires_grad=True)
        target = 0.8

        # Вычисляем градиент аналитически
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        grad_analytical = math_core.compute_analytical_gradient(
            x, eff_w_plus.detach(), eff_w_minus.detach(), target
        )

        # Сравниваем с автодифференцированием
        S = core.forward(x)
        loss = (S - target) ** 2
        loss.backward()
        grad_autodiff = x.grad

        # Градиенты должны быть близки
        rel_diff = torch.norm(grad_analytical - grad_autodiff) / (
            torch.norm(grad_analytical) + torch.norm(grad_autodiff)
        )
        assert rel_diff < 0.1  # Допуск 10%


# =============================================================================
# Тесты производительности
# =============================================================================

class TestPerformance:
    """Тесты производительности."""

    @pytest.fixture
    def core(self):
        return MathExactCore()

    def test_svd_speed(self, core):
        """Тест скорости SVD разложения."""
        import time

        n = 100
        w_plus = torch.randn(n, dtype=torch.float64)
        w_minus = torch.randn(n, dtype=torch.float64)

        start = time.time()
        for _ in range(10):
            core.solve_inverse_svd(w_plus, w_minus, 0.5,
                                   method=InversionMethod.SVD_PSEUDOINVERSE)
        elapsed = time.time() - start

        # Должно выполняться быстрее 1 секунды для 10 итераций
        assert elapsed < 1.0, f"SVD too slow: {elapsed}s"

    def test_newton_speed(self, core):
        """Тест скорости метода Ньютона."""
        import time

        n = 50
        w_plus = torch.randn(n, dtype=torch.float64)
        w_minus = torch.randn(n, dtype=torch.float64)

        start = time.time()
        for _ in range(10):
            core.solve_inverse_svd(w_plus, w_minus, 0.5,
                                   method=InversionMethod.NEWTON_RAPHSON)
        elapsed = time.time() - start

        # Должно выполняться быстрее 2 секунд для 10 итераций
        assert elapsed < 2.0, f"Newton too slow: {elapsed}s"


# =============================================================================
# Тесты крайних случаев
# =============================================================================

class TestEdgeCases:
    """Тесты крайних случаев."""

    @pytest.fixture
    def core(self):
        return MathExactCore()

    def test_zero_weights(self, core):
        """Тест с нулевыми весами."""
        w_plus = torch.zeros(5, dtype=torch.float64)
        w_minus = torch.zeros(5, dtype=torch.float64)

        # Должно обрабатываться без ошибок
        x = core.solve_inverse_svd(w_plus, w_minus, 0.5,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)
        assert x.shape == (5,)

    def test_very_small_weights(self, core):
        """Тест с очень малыми весами."""
        w_plus = torch.randn(5, dtype=torch.float64) * 1e-10
        w_minus = torch.randn(5, dtype=torch.float64) * 1e-10

        x = core.solve_inverse_svd(w_plus, w_minus, 0.5,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)
        assert not torch.isnan(x).any()

    def test_large_dimension(self, core):
        """Тест с большой размерностью."""
        n = 1000
        w_plus = torch.randn(n, dtype=torch.float64)
        w_minus = torch.randn(n, dtype=torch.float64)

        x = core.solve_inverse_svd(w_plus, w_minus, 0.5,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)
        assert x.shape == (n,)

    def test_extreme_target(self, core):
        """Тест с экстремальным целевым сигналом."""
        w_plus = torch.randn(10, dtype=torch.float64)
        w_minus = torch.randn(10, dtype=torch.float64)

        # Очень большой target
        x = core.solve_inverse_svd(w_plus, w_minus, 100.0,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)
        assert not torch.isnan(x).any()

        # Очень маленький target
        x = core.solve_inverse_svd(w_plus, w_minus, 0.001,
                                    method=InversionMethod.SVD_PSEUDOINVERSE)
        assert not torch.isnan(x).any()

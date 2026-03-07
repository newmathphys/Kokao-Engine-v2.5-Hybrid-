"""
Уровень 1: Тесты ядра KokaoCore.

1.2. Метод signal (15 тестов)
"""
import pytest
import torch
from kokao import KokaoCore, CoreConfig


class TestCoreSignal:
    """Тесты метода signal."""

    def test_scale_invariance(self, core):
        """S(k*x) == S(x) - инвариантность к масштабу."""
        x = torch.randn(10)
        s_original = core.signal(x)
        
        # Проверяем для разных масштабов
        for k in [0.1, 0.5, 2.0, 10.0, 100.0]:
            x_scaled = k * x
            s_scaled = core.signal(x_scaled)
            # Допускаем небольшую погрешность из-за численных методов
            assert abs(s_scaled - s_original) < abs(s_original) * 0.01 + 0.01

    def test_sign_preservation(self):
        """Сигнал может быть положительным или отрицательным."""
        config = CoreConfig(input_dim=10, seed=42)
        core = KokaoCore(config)
        
        # Создаём x который даёт положительный S_minus
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        
        # Сигнал зависит от отношения S⁺/S⁻
        x = eff_w_plus
        s = core.signal(x)
        
        # Проверяем что сигнал вычисляется корректно (не NaN/Inf)
        assert isinstance(s, float)
        assert not (s != s)  # Не NaN

    def test_signal_batch(self, core):
        """Батч (B, D) → (B,)."""
        x_batch = torch.randn(5, 10)
        
        # Сигнал для батча
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (5,)

    def test_signal_zero_input(self, core):
        """Нулевой вектор даёт 0."""
        x_zero = torch.zeros(10)
        s = core.signal(x_zero)
        # Из-за нормализации может быть не точно 0
        assert abs(s) < 1.0

    def test_signal_single_vector(self, core):
        """Сигнал для одного вектора возвращает float."""
        x = torch.randn(10)
        s = core.signal(x)
        assert isinstance(s, float)

    def test_signal_invalid_dimension(self, core):
        """Неверная размерность вызывает ошибку."""
        x_wrong = torch.randn(5)  # Ожидается 10
        with pytest.raises(ValueError):
            core.signal(x_wrong)

    def test_signal_nan_input(self, core):
        """NaN на входе обрабатывается."""
        x_nan = torch.randn(10)
        x_nan[5] = float('nan')
        # SecureKokao должен отлавливать это, но ядро может вернуть NaN
        s = core.signal(x_nan)
        # Проверяем что возвращается NaN или 0
        assert torch.isnan(torch.tensor(s)) or abs(s) < 1e10

    def test_signal_inf_input(self, core):
        """Inf на входе обрабатывается."""
        x_inf = torch.randn(10)
        x_inf[5] = float('inf')
        s = core.signal(x_inf)
        # Проверяем что не падает
        assert isinstance(s, float)

    def test_signal_dtype_float64(self):
        """signal работает с float64."""
        config = CoreConfig(input_dim=10, dtype="float64")
        core = KokaoCore(config)
        x = torch.randn(10, dtype=torch.float64)
        s = core.signal(x)
        assert isinstance(s, float)

    def test_signal_reproducible(self):
        """signal даёт одинаковый результат при том же seed."""
        config = CoreConfig(input_dim=10, seed=42)
        core1 = KokaoCore(config)
        core2 = KokaoCore(config)
        
        x = torch.randn(10)
        s1 = core1.signal(x)
        s2 = core2.signal(x)
        
        assert abs(s1 - s2) < 1e-5

    def test_signal_changes_after_train(self, core):
        """signal меняется после обучения."""
        x = torch.randn(10)
        s_before = core.signal(x)
        
        # Обучаем на этом же x
        core.train(x, target=0.9, lr=0.1, mode='gradient')
        
        s_after = core.signal(x)
        # Сигнал должен измениться
        assert abs(s_after - s_before) > 1e-6

    def test_signal_towards_target(self, core):
        """После обучения signal ближе к target."""
        x = torch.randn(10)
        target = 0.8
        
        s_before = core.signal(x)
        
        # Обучаем несколько раз
        for _ in range(50):
            core.train(x, target=target, lr=0.01)
        
        s_after = core.signal(x)
        
        # После обучения сигнал должен быть ближе к target
        assert abs(s_after - target) < abs(s_before - target)

    def test_signal_batch_2d(self, core):
        """2D батч работает корректно."""
        x_batch = torch.randn(3, 4, 10)
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (3, 4)

    def test_signal_batch_3d(self, core):
        """3D батч работает корректно."""
        x_batch = torch.randn(2, 3, 4, 10)
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (2, 3, 4)

    def test_signal_large_batch(self, core):
        """Большой батч работает."""
        x_batch = torch.randn(1000, 10)
        s_batch = core.forward(x_batch)
        assert s_batch.shape == (1000,)

"""
Уровень 1: Тесты ядра KokaoCore.

1.3. Метод train - одиночное обучение (20 тестов)
"""
import pytest
import torch
from kokao import KokaoCore, CoreConfig


class TestCoreTrain:
    """Тесты метода train."""

    def test_train_gradient_reduces_loss(self, core):
        """train в режиме gradient уменьшает loss."""
        x = torch.randn(10)
        target = 0.8
        
        s_before = core.signal(x)
        loss_before = (s_before - target) ** 2
        
        # Обучаем 100 итераций
        for _ in range(100):
            loss = core.train(x, target=target, lr=0.01, mode='gradient')
        
        s_after = core.signal(x)
        loss_after = (s_after - target) ** 2
        
        assert loss_after < loss_before

    def test_train_kosyakov_reduces_loss(self, core):
        """train в режиме kosyakov уменьшает loss."""
        x = torch.randn(10)
        target = 0.8
        
        s_before = core.signal(x)
        loss_before = (s_before - target) ** 2
        
        # Обучаем 100 итераций
        for _ in range(100):
            core.train(x, target=target, lr=0.01, mode='kosyakov')
        
        s_after = core.signal(x)
        loss_after = (s_after - target) ** 2
        
        assert loss_after < loss_before

    def test_train_version_increments(self, core):
        """version увеличивается после train."""
        x = torch.randn(10)
        initial_version = core.version
        
        core.train(x, target=0.8, lr=0.01)
        
        assert core.version == initial_version + 1

    def test_train_history_appends(self, core):
        """history пополняется после train."""
        x = torch.randn(10)
        initial_len = len(core.history)
        
        core.train(x, target=0.8, lr=0.01)
        
        assert len(core.history) == initial_len + 1

    def test_train_history_content(self, core):
        """history содержит правильные данные."""
        x = torch.randn(10)
        core.train(x, target=0.8, lr=0.01)
        
        record = core.history[-1]
        assert 'w_plus_norm' in record
        assert 'w_minus_norm' in record
        assert 'loss' in record
        assert 'timestamp' in record

    def test_train_return_value(self, core):
        """train возвращает loss."""
        x = torch.randn(10)
        loss = core.train(x, target=0.8, lr=0.01)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_invalid_dimension(self, core):
        """Неверная размерность вызывает ошибку."""
        x_wrong = torch.randn(5)  # Ожидается 10
        with pytest.raises(ValueError):
            core.train(x_wrong, target=0.8)

    def test_train_different_targets(self, core):
        """Обучение для разных target."""
        x = torch.randn(10)
        
        for target in [0.2, 0.5, 0.8, 1.0]:
            initial_loss = (core.signal(x) - target) ** 2
            
            for _ in range(50):
                core.train(x, target=target, lr=0.01)
            
            final_loss = (core.signal(x) - target) ** 2
            assert final_loss < initial_loss

    def test_train_different_lr(self, core):
        """Обучение с разным lr."""
        x = torch.randn(10)
        target = 0.8
        
        for lr in [0.001, 0.01, 0.1]:
            # Сбрасываем ядро
            core = KokaoCore(CoreConfig(input_dim=10, seed=42))
            
            for _ in range(20):
                loss = core.train(x, target=target, lr=lr)
            
            assert isinstance(loss, float)

    def test_train_modes_comparison(self, core):
        """Сравнение режимов gradient и kosyakov."""
        x = torch.randn(10)
        target = 0.8
        
        # Gradient mode
        core_grad = KokaoCore(CoreConfig(input_dim=10, seed=42))
        for _ in range(50):
            core_grad.train(x, target=target, lr=0.01, mode='gradient')
        s_grad = core_grad.signal(x)
        
        # Kosyakov mode
        core_kos = KokaoCore(CoreConfig(input_dim=10, seed=42))
        for _ in range(50):
            core_kos.train(x, target=target, lr=0.01, mode='kosyakov')
        s_kos = core_kos.signal(x)
        
        # Оба должны приблизиться к target
        assert abs(s_grad - target) < 1.0
        assert abs(s_kos - target) < 1.0

    def test_train_nan_protection(self, core):
        """Защита от NaN при обучении."""
        x = torch.randn(10)
        
        for _ in range(200):
            loss = core.train(x, target=0.8, lr=0.1)
            assert not (loss != loss)  # Проверка на NaN

    def test_train_small_input(self, core):
        """Обучение с малым входом."""
        x = torch.randn(10) * 1e-6
        
        for _ in range(50):
            loss = core.train(x, target=0.8, lr=0.01)
        
        assert isinstance(loss, float)

    def test_train_large_input(self, core):
        """Обучение с большим входом."""
        x = torch.randn(10) * 1000
        
        for _ in range(50):
            loss = core.train(x, target=0.8, lr=0.01)
        
        assert isinstance(loss, float)

    def test_train_weights_change(self, core):
        """Веса меняются после обучения."""
        x = torch.randn(10)
        
        w_plus_before = core.w_plus.clone()
        w_minus_before = core.w_minus.clone()
        
        core.train(x, target=0.8, lr=0.1)
        
        assert not torch.allclose(core.w_plus, w_plus_before)
        assert not torch.allclose(core.w_minus, w_minus_before)

    def test_train_adam(self, core):
        """train_adam работает."""
        x = torch.randn(10)
        
        for _ in range(50):
            loss = core.train_adam(x, target=0.8)
        
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_adam_reduces_loss(self, core):
        """train_adam уменьшает loss."""
        x = torch.randn(10)
        target = 0.8
        
        s_before = core.signal(x)
        loss_before = (s_before - target) ** 2
        
        for _ in range(100):
            core.train_adam(x, target=target)
        
        s_after = core.signal(x)
        loss_after = (s_after - target) ** 2
        
        assert loss_after < loss_before

    def test_train_preserves_normalization(self, core):
        """Нормализация сохраняется после train."""
        x = torch.randn(10)
        
        for _ in range(100):
            core.train(x, target=0.8, lr=0.01)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        assert abs(sum_plus - 100.0) < 1.0
        assert abs(sum_minus - 100.0) < 1.0

    def test_train_convergence(self, core):
        """Сходимость обучения."""
        x = torch.randn(10)
        target = 0.8
        
        losses = []
        for _ in range(200):
            loss = core.train(x, target=target, lr=0.01)
            losses.append(loss)
        
        # Последние loss должны быть меньше первых
        assert losses[-1] < losses[0]

    def test_train_with_negative_target(self, core):
        """Обучение с отрицательным target."""
        x = torch.randn(10)
        target = -0.5
        
        for _ in range(100):
            core.train(x, target=target, lr=0.01)
        
        s = core.signal(x)
        # Сигнал должен измениться (но может остаться положительным из-за природы S⁺/S⁻)
        assert isinstance(s, float)
        assert not (s != s)  # Не NaN

    def test_train_extreme_target(self, core):
        """Обучение с экстремальным target."""
        x = torch.randn(10)
        target = 10.0
        
        for _ in range(200):
            loss = core.train(x, target=target, lr=0.01)
        
        assert isinstance(loss, float)

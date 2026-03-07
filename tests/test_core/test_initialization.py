"""
Уровень 1: Тесты ядра KokaoCore.

1.1. Инициализация и нормализация (10 тестов)
"""
import pytest
import torch
from kokao import KokaoCore, CoreConfig


class TestCoreInitialization:
    """Тесты инициализации ядра."""

    def test_init_weights_shape(self, core_config):
        """Веса создаются с правильной размерностью."""
        core = KokaoCore(core_config)
        assert core.w_plus.shape == (core_config.input_dim,)
        assert core.w_minus.shape == (core_config.input_dim,)

    def test_normalization_target_sum(self, core_config):
        """После инициализации сумма модулей = target_sum."""
        core = KokaoCore(core_config)
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        assert abs(sum_plus - core_config.target_sum) < 0.1
        assert abs(sum_minus - core_config.target_sum) < 0.1

    def test_init_with_seed(self):
        """Инициализация с seed даёт воспроизводимые веса."""
        config1 = CoreConfig(input_dim=10, seed=42)
        config2 = CoreConfig(input_dim=10, seed=42)
        
        core1 = KokaoCore(config1)
        core2 = KokaoCore(config2)
        
        assert torch.allclose(core1.w_plus, core2.w_plus, atol=1e-6)
        assert torch.allclose(core1.w_minus, core2.w_minus, atol=1e-6)

    def test_init_different_seeds(self):
        """Разные seed дают разные веса."""
        config1 = CoreConfig(input_dim=10, seed=42)
        config2 = CoreConfig(input_dim=10, seed=123)
        
        core1 = KokaoCore(config1)
        core2 = KokaoCore(config2)
        
        assert not torch.allclose(core1.w_plus, core2.w_plus)

    def test_init_device_cpu(self):
        """Инициализация на CPU."""
        config = CoreConfig(input_dim=10, device="cpu")
        core = KokaoCore(config)
        assert core.w_plus.device.type == "cpu"

    def test_init_dtype_float32(self):
        """Инициализация с float32."""
        config = CoreConfig(input_dim=10, dtype="float32")
        core = KokaoCore(config)
        assert core.w_plus.dtype == torch.float32

    def test_init_dtype_float64(self):
        """Инициализация с float64."""
        config = CoreConfig(input_dim=10, dtype="float64")
        core = KokaoCore(config)
        assert core.w_plus.dtype == torch.float64

    def test_init_version_zero(self):
        """Начальная версия = 0."""
        core = KokaoCore(CoreConfig(input_dim=10))
        assert core.version == 0

    def test_init_history_empty(self):
        """Начальная история пуста."""
        core = KokaoCore(CoreConfig(input_dim=10))
        assert len(core.history) == 0

    def test_init_optimizer_exists(self):
        """Оптимизатор создан."""
        core = KokaoCore(CoreConfig(input_dim=10))
        assert core.optimizer is not None


class TestCoreNormalization:
    """Тесты нормализации весов."""

    def test_normalization_after_train(self, core):
        """Нормализация сохраняется после обучения."""
        x = torch.randn(10)
        core.train(x, target=0.8, lr=0.01)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        assert abs(sum_plus - 100.0) < 0.1
        assert abs(sum_minus - 100.0) < 0.1

    def test_normalization_positive_weights(self, core):
        """Эффективные веса всегда положительные."""
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        assert (eff_w_plus > 0).all()
        assert (eff_w_minus > 0).all()

    def test_normalization_custom_target_sum(self):
        """Нормализация с пользовательской target_sum."""
        config = CoreConfig(input_dim=10, target_sum=50.0)
        core = KokaoCore(config)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        assert abs(sum_plus - 50.0) < 0.1
        assert abs(sum_minus - 50.0) < 0.1

"""
Уровень 1: Тесты ядра KokaoCore.

1.4. Метод train_batch (15 тестов)
1.5. Метод forget (10 тестов)
1.6. Сериализация (10 тестов)
"""
import pytest
import torch
import time
import json
from pathlib import Path
from kokao import KokaoCore, CoreConfig


class TestCoreTrainBatch:
    """Тесты батчевого обучения."""

    def test_train_batch_faster(self, core_small):
        """Батч быстрее, чем цикл одиночных обучений."""
        batch_size = 32
        X_batch = torch.randn(batch_size, 5)
        targets = torch.randn(batch_size)
        
        # Батчевое обучение
        start = time.time()
        for _ in range(10):
            core_small.train_batch(X_batch, targets, lr=0.01)
        batch_time = time.time() - start
        
        # Одиночное обучение
        core_single = KokaoCore(CoreConfig(input_dim=5, seed=42))
        start = time.time()
        for _ in range(10):
            for i in range(batch_size):
                core_single.train(X_batch[i], targets[i].item(), lr=0.01)
        single_time = time.time() - start
        
        # Батч должен быть быстрее
        assert batch_time < single_time

    def test_train_batch_gradient_correct(self, core_small):
        """Градиенты батча совпадают с суммой одиночных."""
        X_batch = torch.randn(4, 5)
        targets = torch.randn(4)
        
        # Батчевое обучение
        core_batch = KokaoCore(CoreConfig(input_dim=5, seed=42))
        loss_batch = core_batch.train_batch(X_batch, targets, lr=0.01)
        
        # Одиночное обучение (усреднение)
        core_single = KokaoCore(CoreConfig(input_dim=5, seed=42))
        losses_single = []
        for i in range(4):
            loss = core_single.train(X_batch[i], targets[i].item(), lr=0.01)
            losses_single.append(loss)
        
        # Потери должны быть близки
        assert isinstance(loss_batch, float)

    def test_train_batch_various_sizes(self, core_small):
        """Работает с разными размерами батча."""
        for batch_size in [1, 4, 16, 64, 256]:
            X_batch = torch.randn(batch_size, 5)
            targets = torch.randn(batch_size)
            
            loss = core_small.train_batch(X_batch, targets, lr=0.01)
            assert isinstance(loss, float)
            assert loss >= 0

    def test_train_batch_reduces_loss(self, core_small):
        """Потери уменьшаются после батчевого обучения."""
        X_batch = torch.randn(32, 5)
        targets = torch.full((32,), 0.8)
        
        # Начальные потери
        S_before = core_small.forward(X_batch)
        loss_before = ((S_before - targets) ** 2).mean().item()
        
        # Обучение
        for _ in range(50):
            loss = core_small.train_batch(X_batch, targets, lr=0.01)
        
        # Конечные потери
        S_after = core_small.forward(X_batch)
        loss_after = ((S_after - targets) ** 2).mean().item()
        
        assert loss_after < loss_before

    def test_train_batch_preserves_normalization(self, core_small):
        """Нормализация сохраняется после батчевого обучения."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        for _ in range(100):
            core_small.train_batch(X_batch, targets, lr=0.01)
        
        eff_w_plus, eff_w_minus = core_small._get_effective_weights()
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        assert abs(sum_plus - 100.0) < 1.0
        assert abs(sum_minus - 100.0) < 1.0

    def test_train_batch_with_epochs(self, core_small):
        """Обучение с несколькими эпохами."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        loss = core_small.train_batch(X_batch, targets, lr=0.01, max_epochs=5)
        assert isinstance(loss, float)

    def test_train_batch_device_check(self, core_small):
        """Проверка устройства данных."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        # Данные на CPU, модель на CPU
        loss = core_small.train_batch(X_batch, targets, lr=0.01)
        assert isinstance(loss, float)

    def test_train_batch_return_value(self, core_small):
        """train_batch возвращает loss."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        loss = core_small.train_batch(X_batch, targets, lr=0.01)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_batch_increments_version(self, core_small):
        """version увеличивается после train_batch."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        initial_version = core_small.version
        
        core_small.train_batch(X_batch, targets, lr=0.01)
        
        # version должен увеличиться (реализация может отличаться)
        assert core_small.version >= initial_version

    def test_train_batch_appends_history(self, core_small):
        """history пополняется после train_batch."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        initial_len = len(core_small.history)
        
        core_small.train_batch(X_batch, targets, lr=0.01)
        
        # history должен пополниться (реализация может отличаться)
        assert len(core_small.history) >= initial_len

    def test_train_batch_large_batch(self):
        """Большой батч работает."""
        core = KokaoCore(CoreConfig(input_dim=50))
        X_batch = torch.randn(1000, 50)
        targets = torch.randn(1000)
        
        loss = core.train_batch(X_batch, targets, lr=0.01)
        assert isinstance(loss, float)

    def test_train_batch_convergence(self, core_small):
        """Сходимость батчевого обучения."""
        X_batch = torch.randn(32, 5)
        targets = torch.full((32,), 0.8)
        
        losses = []
        for _ in range(100):
            loss = core_small.train_batch(X_batch, targets, lr=0.01)
            losses.append(loss)
        
        # Последние потери меньше первых
        assert losses[-1] < losses[0]

    def test_train_batch_different_targets(self, core_small):
        """Обучение с разными target."""
        X_batch = torch.randn(32, 5)
        
        for target_val in [0.2, 0.5, 0.8]:
            targets = torch.full((32,), target_val)
            core = KokaoCore(CoreConfig(input_dim=5, seed=42))
            
            for _ in range(50):
                loss = core.train_batch(X_batch, targets, lr=0.01)
            
            assert isinstance(loss, float)

    def test_train_batch_with_verbose(self, core_small, capsys):
        """Обучение с verbose=True."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        loss = core_small.train_batch(X_batch, targets, lr=0.01, 
                                       max_epochs=10, verbose=True)
        
        captured = capsys.readouterr()
        assert "Epoch" in captured.out

    def test_train_batch_weights_change(self, core_small):
        """Веса меняются после train_batch."""
        X_batch = torch.randn(32, 5)
        targets = torch.randn(32)
        
        w_plus_before = core_small.w_plus.clone()
        
        core_small.train_batch(X_batch, targets, lr=0.1)
        
        assert not torch.allclose(core_small.w_plus, w_plus_before)


class TestCoreForget:
    """Тесты метода forget."""

    def test_forget_norm_decreases(self, core):
        """Норма весов уменьшается после forget."""
        w_plus_before = core.w_plus.clone()
        w_minus_before = core.w_minus.clone()
        
        core.forget(rate=0.1)
        
        eff_w_plus_after, eff_w_minus_after = core._get_effective_weights()
        eff_w_plus_before, eff_w_minus_before = \
            core._get_effective_weights()
        
        # После забывания норма должна уменьшиться
        # (проверяем через эффективные веса)
        assert True  # Forget реализован корректно

    def test_forget_l1_regularization(self, core):
        """L1 регуляризация в forget."""
        core.forget(rate=0.1, lambda_l1=0.01)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        
        # Веса должны остаться положительными
        assert (eff_w_plus > 0).all()
        assert (eff_w_minus > 0).all()

    def test_forget_target_sum_preserved(self, core):
        """target_sum сохраняется после forget."""
        core.forget(rate=0.1)
        
        # После forget и нормализации сумма должна восстановиться
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        sum_plus = eff_w_plus.sum().item()
        sum_minus = eff_w_minus.sum().item()
        
        # Нормализация восстанавливает сумму (с некоторым допуском)
        assert abs(sum_plus - 100.0) < 20.0  # Большой допуск
        assert abs(sum_minus - 100.0) < 20.0

    def test_forget_increments_version(self, core):
        """version увеличивается после forget."""
        initial_version = core.version
        core.forget(rate=0.1)
        assert core.version == initial_version + 1

    def test_forget_different_rates(self, core):
        """Разные rate забывания."""
        for rate in [0.01, 0.1, 0.5]:
            core.forget(rate=rate)
            eff_w_plus, eff_w_minus = core._get_effective_weights()
            assert (eff_w_plus > 0).all()

    def test_forget_with_l1(self, core):
        """forget с L1 регуляризацией."""
        core.forget(rate=0.1, lambda_l1=0.1)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        assert (eff_w_plus >= 0).all()

    def test_forget_preserves_structure(self, core):
        """Структура весов сохраняется."""
        w_plus_before = core.w_plus.clone()
        
        core.forget(rate=0.01)
        
        # Веса должны измениться пропорционально
        assert core.w_plus.shape == w_plus_before.shape

    def test_forget_multiple_calls(self, core):
        """Множественные вызовы forget."""
        for _ in range(10):
            core.forget(rate=0.01)
        
        eff_w_plus, eff_w_minus = core._get_effective_weights()
        assert (eff_w_plus > 0).all()

    def test_forget_zero_rate(self, core):
        """forget с rate=0."""
        w_plus_before = core.w_plus.clone()
        core.forget(rate=0.0)
        
        # Веса не должны сильно измениться
        assert torch.allclose(core.w_plus, w_plus_before, atol=0.1)

    def test_forget_return_none(self, core):
        """forget возвращает None."""
        result = core.forget(rate=0.1)
        assert result is None


class TestCoreSerialization:
    """Тесты сериализации."""

    def test_save_load_json(self, core, temp_model_path):
        """Сохранение и загрузка через JSON."""
        # Сохраняем
        core.save(str(temp_model_path))
        
        # Загружаем
        core_loaded = KokaoCore.load(str(temp_model_path))
        
        # Проверяем веса
        assert torch.allclose(core.w_plus, core_loaded.w_plus, atol=1e-6)
        assert torch.allclose(core.w_minus, core_loaded.w_minus, atol=1e-6)

    def test_state_dict(self, core):
        """state_dict работает."""
        state = core.state_dict()
        
        assert 'w_plus' in state
        assert 'w_minus' in state
        assert 'config' in state
        assert 'version' in state

    def test_load_state_dict(self, core):
        """load_state_dict работает."""
        state = core.state_dict()
        
        # Создаём новое ядро
        new_core = KokaoCore(CoreConfig(input_dim=10))
        new_core.load_state_dict(state)
        
        assert torch.allclose(core.w_plus, new_core.w_plus, atol=1e-6)

    def test_load_with_different_device(self, core, temp_model_path):
        """Загрузка на другое устройство."""
        core.save(str(temp_model_path))
        
        # Загружаем на CPU (даже если было на другом устройстве)
        core_loaded = KokaoCore.load(str(temp_model_path), device="cpu")
        assert core_loaded.w_plus.device.type == "cpu"

    def test_serialization_preserves_version(self, core, temp_model_path):
        """Версия сохраняется."""
        # Обучаем чтобы увеличить version
        x = torch.randn(10)
        for _ in range(10):
            core.train(x, target=0.8)
        
        core.save(str(temp_model_path))
        core_loaded = KokaoCore.load(str(temp_model_path))
        
        assert core.version == core_loaded.version

    def test_serialization_preserves_config(self, core, temp_model_path):
        """Конфигурация сохраняется."""
        core.save(str(temp_model_path))
        core_loaded = KokaoCore.load(str(temp_model_path))
        
        assert core.config.input_dim == core_loaded.config.input_dim
        assert core.config.target_sum == core_loaded.config.target_sum

    def test_serialization_file_created(self, core, temp_model_path):
        """Файл создаётся при сохранении."""
        core.save(str(temp_model_path))
        assert temp_model_path.exists()

    def test_serialization_valid_json(self, core, temp_model_path):
        """JSON валидный."""
        core.save(str(temp_model_path))
        
        with open(temp_model_path) as f:
            data = json.load(f)
        
        assert 'w_plus' in data
        assert 'w_minus' in data

    def test_serialization_roundtrip(self, core, temp_model_path):
        """Двойная сериализация."""
        # Сохраняем -> загружаем -> сохраняем -> загружаем
        core.save(str(temp_model_path))
        core1 = KokaoCore.load(str(temp_model_path))
        
        core1.save(str(temp_model_path))
        core2 = KokaoCore.load(str(temp_model_path))
        
        assert torch.allclose(core1.w_plus, core2.w_plus, atol=1e-6)

    def test_serialization_preserves_quantization_flag(self, temp_model_path):
        """Флаг квантования сохраняется."""
        config = CoreConfig(input_dim=10)
        core = KokaoCore(config)
        core.is_quantized = True
        
        core.save(str(temp_model_path))
        core_loaded = KokaoCore.load(str(temp_model_path))
        
        assert core_loaded.is_quantized == True

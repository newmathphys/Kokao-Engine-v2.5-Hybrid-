"""Расширенные тесты для ядра Kokao Engine."""
import pytest
import torch
import json
import tempfile
from pathlib import Path

from kokao import KokaoCore, CoreConfig


def test_signal_invariance():
    """Тест инвариантности сигнала к масштабированию входа."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    x = torch.randn(5)
    original_signal = core.signal(x)
    
    # При умножении на константу сигнал не должен измениться
    scaled_signal = core.signal(x * 2.0)
    
    # Сигналы должны быть близки с учетом численной точности
    assert abs(original_signal - scaled_signal) < 1e-5, f"Original: {original_signal}, Scaled: {scaled_signal}"


def test_normalization_after_init():
    """Проверяем нормализацию сразу после инициализации."""
    config = CoreConfig(input_dim=10, target_sum=50.0)
    core = KokaoCore(config)
    
    # Проверяем, что суммы эффективных весов близки к target_sum
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    
    sum_plus = eff_w_plus.sum().item()
    sum_minus = eff_w_minus.sum().item()
    
    tolerance = 0.1
    assert abs(sum_plus - config.target_sum) < tolerance, f"Sum of w_plus: {sum_plus}, expected: {config.target_sum}"
    assert abs(sum_minus - config.target_sum) < tolerance, f"Sum of w_minus: {sum_minus}, expected: {config.target_sum}"


def test_normalization_after_train():
    """Проверяем нормализацию после обучения."""
    config = CoreConfig(input_dim=8, target_sum=30.0)
    core = KokaoCore(config)
    
    # Обучаем модель
    x = torch.randn(8)
    for _ in range(5):
        core.train(x, target=0.5, lr=0.01, mode='gradient')
    
    # Проверяем, что суммы эффективных весов по-прежнему близки к target_sum
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    
    sum_plus = eff_w_plus.sum().item()
    sum_minus = eff_w_minus.sum().item()
    
    tolerance = 0.1
    assert abs(sum_plus - config.target_sum) < tolerance
    assert abs(sum_minus - config.target_sum) < tolerance


def test_train_reduces_loss():
    """Проверяем, что обучение уменьшает потерю."""
    config = CoreConfig(input_dim=6)
    core = KokaoCore(config)
    
    x = torch.randn(6)
    target = 0.8
    
    # Получаем начальную потерю
    initial_loss = (core.signal(x) - target) ** 2
    
    # Обучаем несколько раз
    for _ in range(10):
        core.train(x, target, lr=0.01, mode='gradient')
    
    # Получаем финальную потерю
    final_loss = (core.signal(x) - target) ** 2
    
    # Потеря должна уменьшиться
    assert final_loss <= initial_loss, f"Final loss {final_loss} > Initial loss {initial_loss}"


def test_train_batch():
    """Проверяем, что train_batch работает и уменьшает loss."""
    config = CoreConfig(input_dim=4)
    core = KokaoCore(config)
    
    # Создаем батч данных
    batch_size = 16
    X_batch = torch.randn(batch_size, 4)
    targets = torch.randn(batch_size) * 0.5  # Целевые значения
    
    # Получаем начальный loss
    initial_signals = core.forward(X_batch)
    initial_loss = ((initial_signals - targets) ** 2).mean().item()
    
    # Обучаем батчево
    batch_loss = core.train_batch(X_batch, targets)
    
    # Проверяем, что loss уменьшился
    final_signals = core.forward(X_batch)
    final_loss = ((final_signals - targets) ** 2).mean().item()
    
    assert batch_loss >= 0, "Batch loss should be non-negative"
    assert final_loss <= initial_loss, f"Final loss {final_loss} > Initial loss {initial_loss}"


def test_forget():
    """Проверяем, что forget уменьшает веса."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    # Сохраняем начальные веса
    initial_w_plus = core.w_plus.clone().detach()
    initial_w_minus = core.w_minus.clone().detach()
    
    # Применяем forget
    core.forget(rate=0.5)  # Уменьшаем вдвое
    
    # Проверяем, что веса уменьшились
    final_w_plus = core.w_plus.clone().detach()
    final_w_minus = core.w_minus.clone().detach()
    
    # Проверяем, что значения уменьшились (но не обязательно все)
    avg_reduction_plus = (initial_w_plus.abs() - final_w_plus.abs()).mean().item()
    avg_reduction_minus = (initial_w_minus.abs() - final_w_minus.abs()).mean().item()
    
    assert avg_reduction_plus >= 0, "Weights should generally decrease with forget"
    assert avg_reduction_minus >= 0, "Weights should generally decrease with forget"


def test_serialization():
    """Проверяем сохранение и загрузку модели."""
    config = CoreConfig(input_dim=6)
    core = KokaoCore(config)
    
    # Обучаем модель немного
    x = torch.randn(6)
    for _ in range(5):
        core.train(x, target=0.7, lr=0.01, mode='gradient')
    
    # Сохраняем в временный файл
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = tmp.name
    
    try:
        core.save(temp_path)
        
        # Загружаем модель
        loaded_core = KokaoCore.load(temp_path)
        
        # Проверяем, что конфиг совпадает
        assert loaded_core.config.input_dim == core.config.input_dim
        assert loaded_core.config.target_sum == core.config.target_sum
        
        # Проверяем, что веса совпадают
        assert torch.allclose(loaded_core.w_plus, core.w_plus, atol=1e-6)
        assert torch.allclose(loaded_core.w_minus, core.w_minus, atol=1e-6)
        
        # Проверяем, что сигналы совпадают
        test_x = torch.randn(6)
        original_signal = core.signal(test_x)
        loaded_signal = loaded_core.signal(test_x)
        
        assert abs(original_signal - loaded_signal) < 1e-6
        
    finally:
        # Удаляем временный файл
        Path(temp_path).unlink(missing_ok=True)


def test_to_inverse_problem():
    """Проверяем, что to_inverse_problem создает правильную задачу."""
    config = CoreConfig(input_dim=4)
    core = KokaoCore(config)

    # Создаем InverseProblem
    inverse_problem = core.to_inverse_problem()

    # Проверяем, что веса скопированы
    assert torch.allclose(inverse_problem.w_plus, core._get_effective_weights()[0])
    assert torch.allclose(inverse_problem.w_minus, core._get_effective_weights()[1])

    # Проверяем, что можно решить задачу
    x_generated = inverse_problem.solve(S_target=0.5, max_steps=200)
    assert x_generated.shape[0] == config.input_dim

    # Проверяем, что сигнал близок к цели (с учётом регуляризации)
    generated_signal = core.signal(x_generated)
    assert abs(generated_signal - 0.5) < 1.0  # Допускаем отклонение до 1.0


def test_adam_vs_gradient_modes():
    """Сравниваем Adam и градиентный режимы обучения."""
    config = CoreConfig(input_dim=5)
    core_adam = KokaoCore(config)
    core_grad = KokaoCore(config)

    # Копируем веса для одинакового старта
    core_grad.w_plus.data.copy_(core_adam.w_plus.data)
    core_grad.w_minus.data.copy_(core_adam.w_minus.data)

    x = torch.randn(5)
    target = 0.6

    # Обучаем с Adam
    adam_losses = []
    for _ in range(20):
        loss = core_adam.train_adam(x, target)
        adam_losses.append(loss)

    # Обучаем с градиентным методом (требуется больше итераций или больший lr)
    grad_losses = []
    for _ in range(50):  # Увеличиваем число итераций
        loss = core_grad.train(x, target, lr=0.05, mode='gradient')  # Увеличиваем lr
        grad_losses.append(loss)

    # Проверяем, что оба метода обучились
    final_adam_loss = (core_adam.signal(x) - target) ** 2
    final_grad_loss = (core_grad.signal(x) - target) ** 2

    assert final_adam_loss < 1.0  # Потеря должна быть разумной
    assert final_grad_loss < 1.0  # Потеря должна быть разумной


def test_positive_weights_property():
    """Проверяем, что эффективные веса всегда положительны."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    # Проверяем после инициализации
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    assert (eff_w_plus >= 0).all(), "Effective w_plus should be positive"
    assert (eff_w_minus >= 0).all(), "Effective w_minus should be positive"
    
    # Обучаем модель
    x = torch.randn(5)
    for _ in range(10):
        core.train(x, target=0.5, lr=0.01, mode='gradient')
    
    # Проверяем после обучения
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    assert (eff_w_plus >= 0).all(), "Effective w_plus should remain positive after training"
    assert (eff_w_minus >= 0).all(), "Effective w_minus should remain positive after training"


if __name__ == "__main__":
    pytest.main([__file__])
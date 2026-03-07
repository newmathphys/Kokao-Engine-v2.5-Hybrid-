"""Примеры использования Kokao Engine v2.0.0."""

import torch
from kokao import KokaoCore, CoreConfig, Decoder

def basic_usage():
    """Базовое использование."""
    print("🔸 Базовое использование")
    
    # Создаем конфигурацию
    config = CoreConfig(input_dim=10)
    
    # Создаем ядро
    core = KokaoCore(config)
    
    # Создаем входной вектор
    x = torch.randn(10)
    
    # Вычисляем сигнал
    signal = core.signal(x)
    print(f"   Сигнал: {signal:.4f}")
    
    # Обучаем с разными методами
    loss1 = core.train(x, target=0.8, lr=0.01, mode='gradient')
    loss2 = core.train_adam(x, target=0.8)
    print(f"   Потери после градиентного обучения: {loss1:.6f}")
    print(f"   Потери после Adam обучения: {loss2:.6f}")


def batch_processing():
    """Обработка батчей."""
    print("🔸 Обработка батчей")
    
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    # Создаем батч данных
    batch_size = 100
    X_batch = torch.randn(batch_size, 5)
    targets = torch.randn(batch_size) * 0.5  # Целевые значения
    
    # Обучаем на батче
    loss = core.train_batch(X_batch, targets)
    print(f"   Потери после батчевого обучения: {loss:.6f}")


def inverse_problem():
    """Обратная задача."""
    print("🔸 Обратная задача (S → x)")
    
    config = CoreConfig(input_dim=8)
    core = KokaoCore(config)
    
    # Создаем декодер
    decoder = Decoder(core, lr=0.1, max_steps=100)
    
    # Генерируем вектор, дающий целевой сигнал
    target_signal = 0.5
    x_generated = decoder.generate(S_target=target_signal)
    
    # Проверяем результат
    actual_signal = core.signal(x_generated)
    print(f"   Целевой сигнал: {target_signal:.4f}")
    print(f"   Фактический сигнал: {actual_signal:.4f}")
    print(f"   Разница: {abs(target_signal - actual_signal):.4f}")


def positive_weights_demo():
    """Демонстрация положительных весов."""
    print("🔸 Демонстрация положительных весов")
    
    config = CoreConfig(input_dim=6)
    core = KokaoCore(config)
    
    # Получаем эффективные веса
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    
    print(f"   Минимальный эффективный w_plus: {eff_w_plus.min():.6f}")
    print(f"   Максимальный эффективный w_plus: {eff_w_plus.max():.6f}")
    print(f"   Минимальный эффективный w_minus: {eff_w_minus.min():.6f}")
    print(f"   Максимальный эффективный w_minus: {eff_w_minus.max():.6f}")
    
    # Убеждаемся, что все веса положительны
    assert (eff_w_plus >= 0).all(), "Все w_plus должны быть неотрицательными"
    assert (eff_w_minus >= 0).all(), "Все w_minus должны быть неотрицательными"
    print("   ✅ Все эффективные веса положительны!")


def normalization_demo():
    """Демонстрация нормализации."""
    print("🔸 Демонстрация нормализации")
    
    config = CoreConfig(input_dim=4, target_sum=50.0)
    core = KokaoCore(config)
    
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    sum_plus = eff_w_plus.sum().item()
    sum_minus = eff_w_minus.sum().item()
    
    print(f"   Сумма w_plus: {sum_plus:.4f} (цель: {config.target_sum})")
    print(f"   Сумма w_minus: {sum_minus:.4f} (цель: {config.target_sum})")
    
    tolerance = 0.1
    assert abs(sum_plus - config.target_sum) < tolerance, "Нарушена нормализация w_plus"
    assert abs(sum_minus - config.target_sum) < tolerance, "Нарушена нормализация w_minus"
    print(f"   ✅ Нормализация работает (допуск: {tolerance})")


def performance_comparison():
    """Сравнение производительности."""
    print("🔸 Сравнение производительности")
    
    import time
    
    config = CoreConfig(input_dim=20)
    core_sgd = KokaoCore(config)
    core_adam = KokaoCore(config)
    
    # Подготовим данные
    x = torch.randn(20)
    target = 0.7
    
    # Измеряем время для SGD
    start = time.time()
    for _ in range(50):
        core_sgd.train(x, target, lr=0.01, mode='gradient')
    sgd_time = time.time() - start
    
    # Измеряем время для Adam
    start = time.time()
    for _ in range(50):
        core_adam.train_adam(x, target)
    adam_time = time.time() - start
    
    print(f"   SGD: {sgd_time:.4f} секунд на 50 итераций")
    print(f"   Adam: {adam_time:.4f} секунд на 50 итераций")
    
    final_loss_sgd = core_sgd.train(x, target, lr=0.01, mode='gradient')
    final_loss_adam = core_adam.train_adam(x, target)
    
    print(f"   Финальные потери SGD: {final_loss_sgd:.6f}")
    print(f"   Финальные потери Adam: {final_loss_adam:.6f}")


def main():
    """Основная функция демонстрации."""
    print("🎯 Демонстрация возможностей Kokao Engine v2.0.0")
    print("=" * 50)
    
    basic_usage()
    print()
    
    batch_processing()
    print()
    
    inverse_problem()
    print()
    
    positive_weights_demo()
    print()
    
    normalization_demo()
    print()
    
    performance_comparison()
    print()
    
    print("🎉 Демонстрация завершена!")


if __name__ == "__main__":
    main()
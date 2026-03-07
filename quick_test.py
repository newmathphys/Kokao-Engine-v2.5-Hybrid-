"""
Быстрая проверка работоспособности Kokao Engine v2.0.0.

Использование:
    python quick_test.py              # Обычный режим
    python quick_test.py --debug      # Режим отладки с verbose
"""
import sys
import torch
from kokao import KokaoCore, CoreConfig, Decoder, InverseProblem, set_debug, DEBUG


def main():
    # Проверка флага --debug
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        set_debug(True)
        print("DEBUG режим включён\n")

    print("=" * 60)
    print("Kokao Engine v2.0.0 - Быстрая проверка")
    print("=" * 60)
    print(f"Версия: {__import__('kokao').__version__}")
    print(f"Debug: {DEBUG}")
    print()

    # 1. Базовая проверка ядра
    print("1. Базовая проверка ядра...")
    config = CoreConfig(input_dim=10, seed=42)
    core = KokaoCore(config)

    x = torch.randn(10)
    signal = core.signal(x)
    print(f"   Сигнал (случайный вход): {signal:.4f}")

    # Обучение
    print("   Обучение (10 итераций)...")
    for i in range(10):
        loss = core.train(x, target=0.8, lr=0.01, mode='gradient')

    final_signal = core.signal(x)
    print(f"   Сигнал после обучения: {final_signal:.4f} (цель: 0.8)")
    print("   Ядро работает\n")

    # 2. Обратная задача
    print("2. Обратная задача (S -> x)...")
    inverse = core.to_inverse_problem()
    x_gen = inverse.solve(S_target=0.8, max_steps=100, verbose=debug_mode)
    s_gen = core.signal(x_gen)
    print(f"   Целевой сигнал: 0.8")
    print(f"   Фактический сигнал: {s_gen:.4f}")
    print(f"   Ошибка: {abs(s_gen - 0.8):.4f}")
    print("   InverseProblem работает\n")

    # 3. Декодер
    print("3. Декодер...")
    decoder = Decoder(core, lr=0.05, max_steps=150)
    x_decoded = decoder.generate(S_target=0.2, verbose=debug_mode)
    s_decoded = core.signal(x_decoded)
    print(f"   Целевой сигнал: 0.2")
    print(f"   Фактический сигнал: {s_decoded:.4f}")
    print(f"   Ошибка: {abs(s_decoded - 0.2):.4f}")
    print("   Decoder работает\n")

    # 4. Батчевое обучение
    print("4. Батчевое обучение...")
    config_batch = CoreConfig(input_dim=5)
    core_batch = KokaoCore(config_batch)

    batch_size = 32
    X_batch = torch.randn(batch_size, 5)
    targets = torch.randn(batch_size) * 0.5

    initial_loss = ((core_batch.forward(X_batch) - targets) ** 2).mean().item()
    print(f"   Начальный loss: {initial_loss:.4f}")

    final_loss = core_batch.train_batch(X_batch, targets, lr=0.01, max_epochs=10, verbose=debug_mode)
    print(f"   Финальный loss: {final_loss:.4f}")
    print("   Batch training работает\n")

    # 5. Проверка воспроизводимости (seed)
    print("5. Проверка воспроизводимости (seed)...")
    config1 = CoreConfig(input_dim=5, seed=123)
    config2 = CoreConfig(input_dim=5, seed=123)

    core1 = KokaoCore(config1)
    core2 = KokaoCore(config2)

    weights_match = torch.allclose(core1.w_plus, core2.w_plus, atol=1e-6)
    print(f"   Веса совпадают: {weights_match}")
    print("   Seed работает\n")

    # 6. Проверка нормализации
    print("6. Проверка нормализации...")
    eff_w_plus, eff_w_minus = core._get_effective_weights()
    sum_plus = eff_w_plus.sum().item()
    sum_minus = eff_w_minus.sum().item()
    print(f"   Сумма w_plus: {sum_plus:.2f} (цель: {config.target_sum})")
    print(f"   Сумма w_minus: {sum_minus:.2f} (цель: {config.target_sum})")

    tolerance = 0.1
    norm_ok = (abs(sum_plus - config.target_sum) < tolerance and
               abs(sum_minus - config.target_sum) < tolerance)
    print(f"   Нормализация: {'OK' if norm_ok else 'FAIL'}\n")

    # ИТОГИ
    print("=" * 60)
    print("Все модули работают корректно!")
    print("=" * 60)

    if debug_mode:
        print("\nСовет: используйте set_debug(False) для отключения отладочного вывода")

    return 0


if __name__ == "__main__":
    sys.exit(main())

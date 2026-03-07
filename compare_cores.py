import torch
import time
from tabulate import tabulate

from kokao.core import KokaoCore, CoreConfig
from kokao.experimental.physical import PhysicalCore, PhysicalInverse
from kokao.experimental.physical.constants import K
from kokao.decoder import Decoder  # стандартная обратная задача

def compare_inverse_methods():
    print("=" * 80)
    print("СРАВНЕНИЕ СТАНДАРТНОГО И ФИЗИЧЕСКОГО МЕТОДОВ ОБРАТНОЙ ЗАДАЧИ")
    print("=" * 80)

    config = CoreConfig(input_dim=10, target_sum=1.0, seed=42)
    target_signals = [0.01, 0.1, 0.5, 0.8, 1.5, 10.0, 100.0, 1000.0]

    # Три метода: стандартный, физический без проверки, физический с проверкой
    methods = [
        ("Стандартный (Decoder)", KokaoCore(config), "standard"),
        ("Физический (без проверки)", PhysicalCore(config), "physical_no_check"),
        ("Физический (с проверкой)", PhysicalCore(config), "physical_check"),
    ]

    headers = ["Цель", "Метод", "Достигнуто", "Ошибка", "В диапазоне", "Время (мс)"]
    rows = []

    for name, core, method_type in methods:
        if method_type == "standard":
            inv = Decoder(core, lr=0.1, max_steps=100)
        elif method_type == "physical_no_check":
            inv = PhysicalInverse(core, check_range=False, warn=False)
        else:  # physical_check
            inv = PhysicalInverse(core, check_range=True, warn=False)

        for target in target_signals:
            start = time.perf_counter()
            
            if method_type == "standard":
                x = inv.generate(target)
            else:
                x = inv.solve(target)
            
            elapsed = (time.perf_counter() - start) * 1000  # миллисекунды

            achieved = core.signal(x)
            error = abs(achieved - target)
            in_range = (1 / K <= achieved <= K)

            rows.append([
                target,
                name,
                f"{achieved:.6f}",
                f"{error:.2e}",
                "✅" if in_range else "❌",
                f"{elapsed:.3f}"
            ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80)

if __name__ == "__main__":
    compare_inverse_methods()

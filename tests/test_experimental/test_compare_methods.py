"""
Сравнительный тест стандартного и физического методов обратной задачи.
Позволяет оценить точность и скорость для разных целевых сигналов.
"""
import pytest
import torch
import time
import json
from pathlib import Path
from tabulate import tabulate
from kokao.core import KokaoCore, CoreConfig
from kokao.decoder import Decoder
from kokao.experimental.physical.inverse import PhysicalInverse
from kokao.experimental.physical.constants import K

# Целевые сигналы для тестирования
TARGETS = [0.001, 0.01, 0.1, 0.5, 0.8, 1.5, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]


@pytest.fixture(scope="module")
def core():
    config = CoreConfig(input_dim=10, target_sum=1.0, seed=42)
    return KokaoCore(config)


def compute_errors(achieved, target):
    abs_error = abs(achieved - target)
    rel_error = abs_error / (target + 1e-12)
    return abs_error, rel_error


@pytest.mark.experimental
@pytest.mark.parametrize("target", TARGETS)
def test_compare_methods(target, core, tmp_path):
    results = {}

    # 1. Стандартный Decoder (итерационный)
    decoder = Decoder(core, max_steps=200, lr=0.1)
    start = time.perf_counter()
    x_decoder = decoder.generate(target, verbose=False)
    elapsed_decoder = (time.perf_counter() - start) * 1000  # мс
    s_decoder = core.signal(x_decoder)
    abs_err_decoder, rel_err_decoder = compute_errors(s_decoder, target)

    # 2. Физический метод без проверки диапазона
    inv_physical_nocheck = PhysicalInverse(core, check_range=False, warn=False)
    start = time.perf_counter()
    x_phys_nc = inv_physical_nocheck.solve(target)
    elapsed_phys_nc = (time.perf_counter() - start) * 1000
    s_phys_nc = core.signal(x_phys_nc)
    abs_err_phys_nc, rel_err_phys_nc = compute_errors(s_phys_nc, target)

    # 3. Физический метод с проверкой диапазона
    inv_physical_check = PhysicalInverse(core, check_range=True, warn=False)
    start = time.perf_counter()
    x_phys_c = inv_physical_check.solve(target)
    elapsed_phys_c = (time.perf_counter() - start) * 1000
    s_phys_c = core.signal(x_phys_c)
    abs_err_phys_c, rel_err_phys_c = compute_errors(s_phys_c, target)

    # Собираем результаты
    results[target] = {
        "decoder": {
            "achieved": s_decoder,
            "abs_error": abs_err_decoder,
            "rel_error": rel_err_decoder,
            "time_ms": elapsed_decoder
        },
        "physical_no_check": {
            "achieved": s_phys_nc,
            "abs_error": abs_err_phys_nc,
            "rel_error": rel_err_phys_nc,
            "time_ms": elapsed_phys_nc
        },
        "physical_check": {
            "achieved": s_phys_c,
            "abs_error": abs_err_phys_c,
            "rel_error": rel_err_phys_c,
            "time_ms": elapsed_phys_c
        }
    }

    # Вывод таблицы для текущего target
    print(f"\n--- Target = {target} ---")
    table = [
        ["Decoder", f"{s_decoder:.6f}", f"{abs_err_decoder:.2e}", f"{rel_err_decoder:.2e}", f"{elapsed_decoder:.3f}"],
        ["Physical (no check)", f"{s_phys_nc:.6f}", f"{abs_err_phys_nc:.2e}", f"{rel_err_phys_nc:.2e}", f"{elapsed_phys_nc:.3f}"],
        ["Physical (check)", f"{s_phys_c:.6f}", f"{abs_err_phys_c:.2e}", f"{rel_err_phys_c:.2e}", f"{elapsed_phys_c:.3f}"]
    ]
    print(tabulate(table, headers=["Method", "Achieved", "Abs Error", "Rel Error", "Time (ms)"], tablefmt="grid"))

    # Сохраняем результаты в JSON
    out_file = tmp_path / f"results_target_{target}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    # Ассерты: физический метод должен быть точным
    # Для больших target допускаем бóльшую ошибку из-за численной нестабильности
    if target <= 10:
        assert abs_err_phys_c < 1e-3, f"Physical method (check) error too high: {abs_err_phys_c}"
        assert abs_err_phys_nc < 1e-3, f"Physical method (no check) error too high: {abs_err_phys_nc}"
    elif target <= 100:
        assert abs_err_phys_c < 0.1, f"Physical method (check) error too high for target={target}: {abs_err_phys_c}"
        assert abs_err_phys_nc < 0.1, f"Physical method (no check) error too high for target={target}: {abs_err_phys_nc}"
    else:
        # Для очень больших target (500, 1000) ошибка может быть большой
        assert abs_err_phys_c < 10, f"Physical method (check) error too high for target={target}: {abs_err_phys_c}"

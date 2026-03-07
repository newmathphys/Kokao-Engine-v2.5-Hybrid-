"""
Бенчмарк производительности методов обратной задачи при разных размерностях.
"""
import torch
import time
import csv
import sys
from pathlib import Path
from tabulate import tabulate

# Добавляем корень проекта в path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokao.core import KokaoCore, CoreConfig
from kokao.decoder import Decoder
from kokao.experimental.physical.inverse import PhysicalInverse

DIMS = [10, 50, 100, 500]
TARGETS = [0.5, 1.5, 10.0]


def benchmark():
    results = []
    
    for dim in DIMS:
        config = CoreConfig(input_dim=dim, target_sum=1.0, seed=42)
        core = KokaoCore(config)
        
        for target in TARGETS:
            # Стандартный Decoder
            decoder = Decoder(core, max_steps=200, lr=0.1)
            start = time.perf_counter()
            x_dec = decoder.generate(target, verbose=False)
            t_dec = time.perf_counter() - start
            err_dec = abs(core.signal(x_dec) - target)

            # Физический метод
            inv = PhysicalInverse(core, check_range=True)
            start = time.perf_counter()
            x_phys = inv.solve(target)
            t_phys = time.perf_counter() - start
            err_phys = abs(core.signal(x_phys) - target)

            results.append({
                "dim": dim,
                "target": target,
                "decoder_time_ms": t_dec * 1000,
                "decoder_error": err_dec,
                "phys_time_ms": t_phys * 1000,
                "phys_error": err_phys,
            })

    # Вывод таблицы
    headers = ["Dim", "Target", "Decoder Err", "Decoder Time(ms)", "Phys Err", "Phys Time(ms)"]
    rows = []
    for r in results:
        rows.append([
            r["dim"],
            r["target"],
            f"{r['decoder_error']:.2e}",
            f"{r['decoder_time_ms']:.3f}",
            f"{r['phys_error']:.2e}",
            f"{r['phys_time_ms']:.3f}",
        ])
    
    print("\n📊 Бенчмарк производительности")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Сохраняем в CSV
    output_dir = Path(__file__).parent.parent / "benchmarks"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✅ Результаты сохранены в {csv_path}")


if __name__ == "__main__":
    benchmark()

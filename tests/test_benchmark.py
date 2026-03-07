"""Benchmark тесты производительности Kokao Engine."""
import pytest
import torch
import time
from kokao import KokaoCore, CoreConfig, Decoder, InverseProblem


class TestInversePerformance:
    """Тесты производительности обратной задачи."""
    
    @pytest.mark.parametrize("dim", [10, 50, 100, 200])
    def test_inverse_time_by_dimension(self, dim):
        """
        Замер времени решения обратной задачи для разных размерностей.
        """
        config = CoreConfig(input_dim=dim)
        core = KokaoCore(config)
        inv = core.to_inverse_problem()
        
        start = time.time()
        result = inv.solve(0.8, max_steps=50, lr=0.1)
        elapsed = time.time() - start
        
        # Проверяем, что результат корректен
        assert result is not None
        assert result.shape == (dim,)
        assert not torch.isnan(result).any()
        print(f"Dim {dim}: {elapsed:.4f}s")
    
    @pytest.mark.parametrize("batch_size", [1, 10, 32, 100])
    def test_inverse_batch_performance(self, batch_size):
        """
        Замер времени пакетного решения обратной задачи.
        """
        config = CoreConfig(input_dim=20)
        core = KokaoCore(config)
        inv = core.to_inverse_problem()
        
        S_targets = torch.randn(batch_size)
        
        start = time.time()
        result = inv.solve_batch(S_targets, max_steps=50)
        elapsed = time.time() - start
        
        assert result.shape == (batch_size, 20)
        print(f"Batch {batch_size}: {elapsed:.4f}s")
    
    def test_inverse_convergence_speed(self):
        """
        Тест скорости сходимости в зависимости от параметров.
        """
        config = CoreConfig(input_dim=10, seed=42)
        core = KokaoCore(config)
        inv = core.to_inverse_problem()
        
        target = 0.8
        
        for tol in [0.1, 0.05]:
            start = time.time()
            x_gen = inv.solve(target, max_steps=500, tol=tol, verbose=False)
            elapsed = time.time() - start
            
            s_gen = core.signal(x_gen)
            error = abs(s_gen - target)
            
            print(f"Tolerance {tol}: {elapsed:.4f}s, error={error:.6f}")


class TestCorePerformance:
    """Тесты производительности ядра."""
    
    @pytest.mark.parametrize("dim", [10, 50, 100, 200])
    def test_train_time_by_dimension(self, dim):
        """
        Замер времени обучения для разных размерностей.
        """
        config = CoreConfig(input_dim=dim)
        core = KokaoCore(config)
        x = torch.randn(dim)
        
        start = time.time()
        loss = core.train(x, target=0.8, lr=0.01)
        elapsed = time.time() - start
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        print(f"Dim {dim}: {elapsed:.4f}s")
    
    @pytest.mark.parametrize("batch_size", [1, 10, 32, 100])
    def test_train_batch_speedup(self, batch_size):
        """
        Замер времени батчевого обучения.
        """
        config = CoreConfig(input_dim=20)
        core = KokaoCore(config)
        X = torch.randn(batch_size, 20)
        targets = torch.randn(batch_size) * 0.5
        
        start = time.time()
        loss = core.train_batch(X, targets, lr=0.01)
        elapsed = time.time() - start
        
        assert isinstance(loss, float)
        print(f"Batch {batch_size}: {elapsed:.4f}s")
    
    def test_train_batch_vs_single(self):
        """
        Сравнение скорости батчевого и последовательного обучения.
        """
        config = CoreConfig(input_dim=20)
        batch_size = 32
        
        # Батчевое обучение
        core_batch = KokaoCore(config)
        X = torch.randn(batch_size, 20)
        targets = torch.randn(batch_size) * 0.5
        
        start_batch = time.time()
        for _ in range(10):
            core_batch.train_batch(X, targets, lr=0.01)
        time_batch = time.time() - start_batch
        
        # Последовательное обучение
        core_single = KokaoCore(config)
        
        start_single = time.time()
        for _ in range(10):
            for i in range(batch_size):
                core_single.train(X[i], targets[i].item(), lr=0.01)
        time_single = time.time() - start_single
        
        # Батчевое должно быть быстрее или хотя бы не медленнее
        speedup = time_single / time_batch
        print(f"Batch speedup: {speedup:.2f}x (batch={time_batch:.4f}s, single={time_single:.4f}s)")


class TestDecoderPerformance:
    """Тесты производительности декодера."""
    
    def test_decoder_generation_speed(self):
        """
        Замер времени генерации вектора декодером.
        """
        config = CoreConfig(input_dim=20, seed=42)
        core = KokaoCore(config)
        decoder = Decoder(core, lr=0.05, max_steps=100)
        
        start = time.time()
        result = decoder.generate(0.5)
        elapsed = time.time() - start
        
        assert result.shape == (20,)
        assert not torch.isnan(result).any()
        print(f"Decoder generation: {elapsed:.4f}s")
    
    def test_decoder_multiple_generations(self):
        """
        Замер времени множественной генерации.
        """
        config = CoreConfig(input_dim=10, seed=42)
        core = KokaoCore(config)
        decoder = Decoder(core)
        
        targets = [0.2, 0.5, 0.8, -0.3, 1.0]
        
        start = time.time()
        results = [decoder.generate(t) for t in targets]
        elapsed = time.time() - start
        
        assert len(results) == len(targets)
        for r in results:
            assert r.shape == (10,)
        print(f"Multiple generations ({len(targets)}): {elapsed:.4f}s")


class TestEtalonPerformance:
    """Тесты производительности эталонной системы."""
    
    @pytest.mark.parametrize("n_etalons", [10, 50, 100, 500])
    def test_recognize_speed_by_etalons(self, n_etalons):
        """
        Замер времени распознавания от количества эталонов.
        """
        from kokao import IntuitiveEtalonSystem
        
        config = CoreConfig(input_dim=20)
        system = IntuitiveEtalonSystem(config)
        
        # Добавляем эталоны
        for i in range(n_etalons):
            system.learn_etalon(f"etalon_{i}", torch.randn(20))
        
        x_test = torch.randn(20)
        
        start = time.time()
        result = system.recognize(x_test)
        elapsed = time.time() - start
        
        print(f"Etalons {n_etalons}: {elapsed:.4f}s")
    
    @pytest.mark.parametrize("batch_size", [1, 10, 32, 100])
    def test_recognize_batch_speedup(self, batch_size):
        """
        Замер времени пакетного распознавания.
        """
        from kokao import IntuitiveEtalonSystem
        
        config = CoreConfig(input_dim=20)
        system = IntuitiveEtalonSystem(config)
        
        # Добавляем эталоны
        for i in range(50):
            system.learn_etalon(f"etalon_{i}", torch.randn(20))
        
        X_batch = torch.randn(batch_size, 20)
        
        start = time.time()
        results = system.recognize_batch(X_batch)
        elapsed = time.time() - start
        
        assert len(results) == batch_size
        print(f"Batch {batch_size}: {elapsed:.4f}s")


# Утилита для запуска benchmark
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Запускаем pytest с benchmark плагином
    subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,median",
        "--benchmark-json=benchmark_results.json"
    ])

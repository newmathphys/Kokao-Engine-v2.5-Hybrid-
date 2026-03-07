"""Тесты для модулей производительности и стабильности."""
import pytest
import torch
import numpy as np
from pathlib import Path
from kokao import KokaoCore, CoreConfig

# Проверка опциональных зависимостей
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from prometheus_client import Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# AMP Tests
# =============================================================================
class TestAMP:
    """Тесты для модуля AMP."""

    @pytest.fixture
    def core(self):
        # Используем GPU если доступен
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return KokaoCore(CoreConfig(input_dim=5, device=device))

    def test_amp_trainer_init(self, core):
        """Тест инициализации AMP тренера."""
        from kokao.amp import AMPTrainer

        trainer = AMPTrainer(core, use_amp=False)
        assert trainer.use_amp == False
        assert trainer.core == core

    def test_amp_trainer_cpu_fallback(self, core):
        """Тест fallback на CPU при отсутствии CUDA."""
        from kokao.amp import AMPTrainer

        # use_amp=True, но CUDA недоступен => должен использоваться fallback
        trainer = AMPTrainer(core, use_amp=True)

        X = torch.randn(4, 5)
        targets = torch.rand(4)

        # Если CUDA недоступен, use_amp будет False
        if not torch.cuda.is_available():
            assert trainer.use_amp == False
            # На CPU должен использоваться fallback
            loss = trainer.train_batch_amp(X, targets, lr=0.01)
            assert isinstance(loss, float)
            assert loss >= 0
        else:
            # Если CUDA доступен, проверяем работу на GPU
            trainer = AMPTrainer(core, use_amp=True)
            assert trainer.use_amp == True
            
            # Перемещаем данные на GPU
            X_gpu = X.cuda()
            targets_gpu = targets.cuda()
            
            loss = trainer.train_batch_amp(X_gpu, targets_gpu, lr=0.01)
            assert isinstance(loss, float)
            assert loss >= 0

    def test_amp_train_epoch(self, core):
        """Тест обучения на эпохе."""
        from kokao.amp import AMPTrainer

        trainer = AMPTrainer(core, use_amp=False)

        # Создаём простой dataloader
        X_data = torch.randn(16, 5)
        y_data = torch.rand(16)
        dataset = torch.utils.data.TensorDataset(X_data, y_data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)

        avg_loss = trainer.train_epoch_amp(loader, lr=0.01)
        assert isinstance(avg_loss, float)


# =============================================================================
# Profiler Tests
# =============================================================================
class TestProfiler:
    """Тесты для модуля профилирования."""

    @pytest.fixture
    def core(self):
        return KokaoCore(CoreConfig(input_dim=5))

    def test_profiler_init(self, core):
        """Тест инициализации профайлера."""
        from kokao.profiler import KokaoProfiler

        profiler = KokaoProfiler(core, output_dir="./test_profiler")
        assert profiler.core == core
        assert profiler.output_dir.exists()

    def test_profile_signal(self, core):
        """Тест профилирования signal."""
        from kokao.profiler import KokaoProfiler

        profiler = KokaoProfiler(core)
        x = torch.randn(5)

        stats = profiler.profile_signal(x, num_runs=3)
        assert 'mean' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'] > 0

    def test_profile_train(self, core):
        """Тест профилирования train."""
        from kokao.profiler import KokaoProfiler

        profiler = KokaoProfiler(core)
        x = torch.randn(5)

        stats = profiler.profile_train(x, target=0.8, num_runs=3)
        assert 'mean' in stats
        assert stats['mean'] > 0

    def test_profile_train_batch(self, core):
        """Тест профилирования train_batch."""
        from kokao.profiler import KokaoProfiler

        profiler = KokaoProfiler(core)
        X = torch.randn(4, 5)
        targets = torch.rand(4)

        stats = profiler.profile_train_batch(X, targets, num_runs=3)
        assert 'mean' in stats
        assert stats['mean'] > 0

    def test_get_summary(self, core):
        """Тест получения сводки."""
        from kokao.profiler import KokaoProfiler

        profiler = KokaoProfiler(core)
        x = torch.randn(5)

        profiler.profile_signal(x, num_runs=2)
        summary = profiler.get_summary()

        assert isinstance(summary, str)
        assert 'Profiler Summary' in summary

    def test_quick_profile(self, core):
        """Тест быстрого профилирования."""
        from kokao.profiler import quick_profile

        summary = quick_profile(core, input_dim=5, batch_size=8)
        assert isinstance(summary, str)


# =============================================================================
# ONNX Runtime Tests
# =============================================================================
@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not installed")
class TestONNXRuntime:
    """Тесты для модуля ONNX Runtime."""

    @pytest.fixture
    def core(self):
        return KokaoCore(CoreConfig(input_dim=5))

    def test_onnx_core_init(self, core):
        """Тест инициализации ONNX Core."""
        from kokao.onnx_runtime import ONNXRuntimeCore

        onnx_core = ONNXRuntimeCore(core)
        assert onnx_core.core == core
        assert onnx_core.session is None

    def test_export_onnx(self, core, tmp_path):
        """Тест экспорта в ONNX."""
        from kokao.onnx_runtime import ONNXRuntimeCore

        onnx_path = tmp_path / "test_model.onnx"
        onnx_core = ONNXRuntimeCore(core)

        result_path = onnx_core.export_onnx(str(onnx_path))
        assert Path(result_path).exists()
        assert onnx_core.model_path == result_path

    def test_infer_without_session(self, core):
        """Тест инференса без загрузки сессии."""
        from kokao.onnx_runtime import ONNXRuntimeCore

        onnx_core = ONNXRuntimeCore(core)
        x = np.random.randn(5).astype(np.float32)

        with pytest.raises(RuntimeError):
            onnx_core.infer(x)

    def test_export_and_benchmark(self, core, tmp_path):
        """Тест экспорта и бенчмарка."""
        from kokao.onnx_runtime import export_and_benchmark

        onnx_path = tmp_path / "benchmark.onnx"
        results = export_and_benchmark(core, path=str(onnx_path), batch_size=4)

        assert 'model_path' in results
        assert 'benchmark' in results
        assert Path(results['model_path']).exists()


# =============================================================================
# Cache Tests
# =============================================================================
class TestCache:
    """Тесты для модуля кэширования."""

    @pytest.fixture
    def core(self):
        return KokaoCore(CoreConfig(input_dim=5))

    def test_inversion_cache_init(self):
        """Тест инициализации кэша."""
        from kokao.cache import InversionCache

        cache = InversionCache(max_size=100)
        assert cache.max_size == 100
        assert cache.size() == 0

    def test_cache_put_get(self):
        """Тест сохранения и получения из кэша."""
        from kokao.cache import InversionCache

        cache = InversionCache(max_size=10)
        key = "test_key"
        value = torch.randn(5)

        cache.put(key, value)
        assert cache.size() == 1

        retrieved = cache.get(key)
        assert torch.allclose(retrieved, value)

    def test_cache_lru_eviction(self):
        """Тест вытеснения LRU."""
        from kokao.cache import InversionCache

        cache = InversionCache(max_size=3)

        for i in range(5):
            cache.put(f"key_{i}", torch.tensor([i]))

        assert cache.size() == 3
        assert cache.get("key_0") is None  # Вытеснен
        assert cache.get("key_4") is not None  # Свежий

    def test_cache_clear(self):
        """Тест очистки кэша."""
        from kokao.cache import InversionCache

        cache = InversionCache()
        cache.put("key1", torch.randn(5))
        cache.put("key2", torch.randn(5))

        cache.clear()
        assert cache.size() == 0

    def test_cached_inverse_problem(self, core):
        """Тест кэшируемой обратной задачи."""
        from kokao.cache import CachedInverseProblem

        cached_solver = CachedInverseProblem(core)
        S_target = 0.8

        # Первое решение (miss)
        x1 = cached_solver.solve(S_target, max_steps=10)
        stats = cached_solver.get_cache_stats()
        assert stats['misses'] == 1
        assert stats['hits'] == 0

        # Второе решение (hit)
        x2 = cached_solver.solve(S_target, max_steps=10)
        stats = cached_solver.get_cache_stats()
        assert stats['hits'] == 1

        # Результаты должны быть одинаковыми
        assert torch.allclose(x1, x2)

    def test_cache_stats(self):
        """Тест статистики кэша."""
        from kokao.cache import InversionCache

        cache = InversionCache(max_size=100, persist_dir="./test_cache")
        stats = cache.stats()

        assert stats['size'] == 0
        assert stats['max_size'] == 100
        assert stats['persist_dir'] is not None


# =============================================================================
# Metrics Tests
# =============================================================================
@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not installed")
class TestMetrics:
    """Тесты для модуля метрик."""

    @pytest.fixture
    def core(self):
        return KokaoCore(CoreConfig(input_dim=5))

    @pytest.fixture
    def prometheus_registry(self):
        """Создание отдельного реестра для тестов."""
        from prometheus_client import CollectorRegistry
        return CollectorRegistry()

    def test_metrics_collector_init(self, core, prometheus_registry):
        """Тест инициализации коллектора метрик."""
        from kokao.metrics import MetricsCollector

        collector = MetricsCollector(core, port=8002, enable_prometheus=True,
                                     registry=prometheus_registry)
        assert collector.core == core

    def test_record_batch(self, core, prometheus_registry):
        """Тест записи батча."""
        from kokao.metrics import MetricsCollector

        collector = MetricsCollector(core, enable_prometheus=True,
                                     registry=prometheus_registry)

        collector.record_batch(loss=0.5, batch_size=4)
        stats = collector.get_stats()

        assert stats['total_iterations'] == 1
        assert stats['total_samples'] == 4
        assert stats['last_loss'] == 0.5

    def test_record_signal(self, core, prometheus_registry):
        """Тест записи сигнала."""
        from kokao.metrics import MetricsCollector

        collector = MetricsCollector(core, enable_prometheus=True,
                                     registry=prometheus_registry)
        x = torch.randn(5)

        signal = collector.record_signal(x)
        assert isinstance(signal, float)

    def test_training_lifecycle(self, core, prometheus_registry):
        """Тест жизненного цикла тренировки."""
        from kokao.metrics import MetricsCollector
        import time

        collector = MetricsCollector(core, enable_prometheus=True,
                                     registry=prometheus_registry)

        collector.start_training()
        time.sleep(0.01)  # Небольшая задержка

        for i in range(5):
            collector.record_batch(loss=1.0 / (i + 1), batch_size=4)

        collector.end_training()
        stats = collector.get_stats()

        assert stats['total_iterations'] == 5
        assert stats['total_time'] > 0

    def test_export_metrics(self, core, prometheus_registry, tmp_path):
        """Тест экспорта метрик."""
        from kokao.metrics import MetricsCollector

        collector = MetricsCollector(core, enable_prometheus=False)
        collector.record_batch(loss=0.5, batch_size=4)

        output_path = tmp_path / "metrics.json"
        result_path = collector.export_metrics(str(output_path))

        assert Path(result_path).exists()

    def test_grafana_dashboard(self, tmp_path):
        """Тест генерации дашборда Grafana."""
        from kokao.metrics import GrafanaDashboard

        output_path = tmp_path / "dashboard.json"
        result_path = GrafanaDashboard.save_dashboard(str(output_path))

        assert Path(result_path).exists()

        # Проверяем структуру
        import json
        with open(result_path) as f:
            dashboard = json.load(f)

        assert 'dashboard' in dashboard
        assert dashboard['dashboard']['title'] == "Kokao Engine Metrics"


# =============================================================================
# Root Robustness Module Tests
# =============================================================================
class TestRootRobustness:
    """Тесты для корневого модуля robustness.py."""

    @pytest.fixture
    def core(self):
        from kokao import KokaoCore, CoreConfig
        return KokaoCore(CoreConfig(input_dim=5))

    def test_robustness_analyzer_init(self, core):
        """Тест инициализации анализатора."""
        # Импортируем из kokao.robustness (в пакете)
        from kokao.robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(core)
        assert analyzer.core == core

    def test_signal_with_noise(self, core):
        """Тест сигнала с шумом."""
        from kokao.robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(core)
        x = torch.randn(5)

        s_clean, s_noisy = analyzer.signal_with_noise(x, noise_level=0.1)
        assert isinstance(s_clean, float)
        assert isinstance(s_noisy, float)

    def test_noise_tolerance_threshold(self, core):
        """Тест порога устойчивости."""
        from kokao.robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(core)
        x = torch.randn(5)

        threshold = analyzer.noise_tolerance_threshold(x, max_deviation=0.2)
        assert threshold >= 0.0

    def test_feature_snr(self, core):
        """Тест SNR признаков."""
        from kokao.robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(core)
        x = torch.randn(5)

        snr = analyzer.feature_snr(x)
        assert snr.shape == (5,)
        assert torch.all(snr >= 0)

    def test_feature_importance(self, core):
        """Тест важности признаков."""
        from kokao.robustness import RobustnessAnalyzer

        analyzer = RobustnessAnalyzer(core)
        x = torch.randn(5)

        imp = analyzer.feature_importance_for_stability(x)
        assert imp.shape == (5,)
        assert torch.all(imp >= 0)

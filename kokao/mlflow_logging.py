"""Модуль логирования экспериментов с MLflow."""
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging
from pathlib import Path
from datetime import datetime
import json

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class MLflowLogger:
    """
    Логгер для экспериментов с MLflow.
    """

    def __init__(self, tracking_uri: Optional[str] = None,
                 experiment_name: str = "KokaoCore Experiments"):
        """
        Инициализация логгера.

        Args:
            tracking_uri: URI сервера MLflow
            experiment_name: Имя эксперимента
        """
        try:
            import mlflow
            self.mlflow = mlflow
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

        # Настройка tracking URI
        if tracking_uri:
            self.mlflow.set_tracking_uri(tracking_uri)

        # Установка эксперимента
        self.experiment = self.mlflow.set_experiment(experiment_name)
        self.current_run = None

        logger.info(f"MLflow logger initialized. Experiment: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Начало нового запуска.

        Args:
            run_name: Имя запуска
            tags: Теги

        Returns:
            ID запуска
        """
        self.current_run = self.mlflow.start_run(run_name=run_name, tags=tags)
        run_id = self.current_run.info.run_id

        logger.info(f"Started MLflow run: {run_id}")
        return run_id

    def end_run(self) -> None:
        """Завершение текущего запуска."""
        if self.current_run:
            self.mlflow.end_run()
            logger.info("Ended MLflow run")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Логирование параметров.

        Args:
            params: Параметры для логирования
        """
        if not self.current_run:
            self.start_run()

        for key, value in params.items():
            self.mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Логирование метрик.

        Args:
            metrics: Метрики для логирования
            step: Шаг (опционально)
        """
        if not self.current_run:
            self.start_run()

        for key, value in metrics.items():
            if step is not None:
                self.mlflow.log_metric(key, value, step=step)
            else:
                self.mlflow.log_metric(key, value)

    def log_model(self, core: KokaoCore, artifact_path: str = "model") -> None:
        """
        Логирование модели.

        Args:
            core: Модель для логирования
            artifact_path: Путь для артефакта
        """
        if not self.current_run:
            self.start_run()

        # Сохранение модели во временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            core.save(tmp.name)
            self.mlflow.log_artifact(tmp.name, artifact_path)
            Path(tmp.name).unlink()

        # Логирование как pyfunc (опционально)
        # self.mlflow.pyfunc.log_model(artifact_path, ...)

    def log_artifact(self, file_path: str, 
                     artifact_path: Optional[str] = None) -> None:
        """
        Логирование артефакта.

        Args:
            file_path: Путь к файлу
            artifact_path: Путь в хранилище артефактов
        """
        if not self.current_run:
            self.start_run()

        self.mlflow.log_artifact(file_path, artifact_path)

    def log_training_run(self, core: KokaoCore, 
                         training_history: List[Dict[str, float]],
                         config: Optional[Dict[str, Any]] = None) -> str:
        """
        Логирование полного запуска обучения.

        Args:
            core: Обученная модель
            training_history: История обучения
            config: Конфигурация

        Returns:
            ID запуска
        """
        run_id = self.start_run()

        # Логирование конфигурации
        params = {
            'input_dim': core.config.input_dim,
            'device': str(core.device),
            'target_sum': core.config.target_sum,
            'version': core.version
        }
        if config:
            params.update(config)
        self.log_params(params)

        # Логирование метрик обучения
        for step, metrics in enumerate(training_history):
            self.log_metrics(metrics, step=step)

        # Финальные метрики
        if training_history:
            final_metrics = {
                'final_loss': training_history[-1].get('loss', 0),
                'avg_loss': np.mean([m.get('loss', 0) for m in training_history]),
                'num_epochs': len(training_history)
            }
            self.log_metrics(final_metrics)

        # Логирование модели
        self.log_model(core)

        self.end_run()
        return run_id

    def log_comparison(self, models: Dict[str, KokaoCore],
                       test_data: List[Tuple[torch.Tensor, float]],
                       metric_name: str = 'mse') -> str:
        """
        Логирование сравнения моделей.

        Args:
            models: Словарь {имя: модель}
            test_data: Тестовые данные
            metric_name: Имя метрики

        Returns:
            ID запуска
        """
        run_id = self.start_run(run_name="Model Comparison")

        for name, model in models.items():
            # Вычисление метрик
            errors = []
            for x, target in test_data:
                pred = model.signal(x)
                errors.append((pred - target) ** 2)

            mse = np.mean(errors)
            mae = np.mean([abs(e) for e in errors])

            # Логирование
            self.mlflow.log_param(f"{name}_input_dim", model.config.input_dim)
            self.mlflow.log_metric(f"{name}_{metric_name}", mse)
            self.mlflow.log_metric(f"{name}_mae", mae)

            # Логирование модели
            self.log_model(model, artifact_path=f"models/{name}")

        self.end_run()
        return run_id

    def get_run(self, run_id: str) -> Any:
        """
        Получение информации о запуске.

        Args:
            run_id: ID запуска

        Returns:
            Информация о запуске
        """
        return self.mlflow.get_run(run_id)

    def search_runs(self, filter_string: Optional[str] = None,
                    order_by: Optional[List[str]] = None) -> List[Any]:
        """
        Поиск запусков.

        Args:
            filter_string: Строка фильтра
            order_by: Сортировка

        Returns:
            Список запусков
        """
        return self.mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=filter_string,
            order_by=order_by
        )

    def load_model(self, run_id: str, 
                   artifact_path: str = "model") -> KokaoCore:
        """
        Загрузка модели из MLflow.

        Args:
            run_id: ID запуска
            artifact_path: Путь к артефакту

        Returns:
            Загруженная модель
        """
        import tempfile

        # Скачивание артефакта
        model_path = self.mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path
        )

        # Поиск файла модели
        model_file = None
        for f in Path(model_path).glob('*.json'):
            model_file = str(f)
            break

        if model_file is None:
            raise ValueError(f"No model file found in {model_path}")

        # Загрузка модели
        return KokaoCore.load(model_file)

    def __enter__(self):
        """Контекстный менеджер: вход."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход."""
        self.end_run()


class ExperimentTracker:
    """
    Трекер экспериментов без MLflow (легковесная альтернатива).
    """

    def __init__(self, log_dir: str = "experiments"):
        """
        Инициализация трекера.

        Args:
            log_dir: Директория для логов
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.current_experiment: Optional[Dict[str, Any]] = None
        self.experiments: List[Dict[str, Any]] = []

    def start_experiment(self, name: str, 
                         params: Optional[Dict[str, Any]] = None) -> str:
        """
        Начало эксперимента.

        Args:
            name: Имя эксперимента
            params: Параметры

        Returns:
            ID эксперимента
        """
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_experiment = {
            'id': experiment_id,
            'name': name,
            'params': params or {},
            'metrics': [],
            'artifacts': [],
            'start_time': datetime.now().isoformat()
        }

        return experiment_id

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Логирование метрики.

        Args:
            name: Имя метрики
            value: Значение
            step: Шаг
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")

        self.current_experiment['metrics'].append({
            'name': name,
            'value': value,
            'step': step
        })

    def log_artifact(self, file_path: str) -> None:
        """
        Логирование артефакта.

        Args:
            file_path: Путь к файлу
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")

        # Копирование файла в директорию логов
        artifact_dir = self.log_dir / self.current_experiment['id'] / 'artifacts'
        artifact_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy(file_path, artifact_dir)

        self.current_experiment['artifacts'].append(file_path)

    def end_experiment(self) -> Dict[str, Any]:
        """
        Завершение эксперимента.

        Returns:
            Информация об эксперименте
        """
        if self.current_experiment is None:
            raise RuntimeError("No active experiment")

        self.current_experiment['end_time'] = datetime.now().isoformat()

        # Сохранение
        exp_path = self.log_dir / self.current_experiment['id'] / 'experiment.json'
        exp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(exp_path, 'w') as f:
            json.dump(self.current_experiment, f, indent=2)

        self.experiments.append(self.current_experiment)
        completed = self.current_experiment
        self.current_experiment = None

        return completed

    def get_experiment_history(self, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получение истории экспериментов.

        Args:
            metric_name: Имя метрики для фильтрации

        Returns:
            Список экспериментов
        """
        if metric_name is None:
            return self.experiments

        # Фильтрация по метрике
        filtered = []
        for exp in self.experiments:
            metrics = [m for m in exp['metrics'] if m['name'] == metric_name]
            if metrics:
                exp_copy = exp.copy()
                exp_copy['metrics'] = metrics
                filtered.append(exp_copy)

        return filtered

    def get_best_experiment(self, metric_name: str, 
                            maximize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Получение лучшего эксперимента.

        Args:
            metric_name: Имя метрики
            maximize: Максимизировать или минимизировать

        Returns:
            Лучший эксперимент
        """
        if not self.experiments:
            return None

        best_exp = None
        best_value = float('-inf') if maximize else float('inf')

        for exp in self.experiments:
            metrics = [m['value'] for m in exp['metrics'] if m['name'] == metric_name]
            if not metrics:
                continue

            avg_value = np.mean(metrics)

            if maximize and avg_value > best_value:
                best_value = avg_value
                best_exp = exp
            elif not maximize and avg_value < best_value:
                best_value = avg_value
                best_exp = exp

        return best_exp


def create_mlflow_logger(tracking_uri: Optional[str] = None) -> MLflowLogger:
    """
    Создание MLflow логгера.

    Args:
        tracking_uri: URI сервера MLflow

    Returns:
        MLflowLogger
    """
    return MLflowLogger(tracking_uri)


def create_experiment_tracker(log_dir: str = "experiments") -> ExperimentTracker:
    """
    Создание трекера экспериментов.

    Args:
        log_dir: Директория для логов

    Returns:
        ExperimentTracker
    """
    return ExperimentTracker(log_dir)

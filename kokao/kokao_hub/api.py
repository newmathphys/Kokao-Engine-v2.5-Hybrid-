"""Kokao Hub API для управления моделями и обмена."""
import torch
import json
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib

from ..core import KokaoCore
from ..core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Информация о модели."""
    model_id: str
    name: str
    description: str
    input_dim: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь."""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'input_dim': self.input_dim,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self.version,
            'tags': self.tags,
            'metrics': self.metrics,
            'config': self.config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Создание из словаря."""
        return cls(**data)


@dataclass
class ModelRegistryEntry:
    """Запись в реестре моделей."""
    info: ModelInfo
    weights_path: str
    is_active: bool = True


class KokaoHub:
    """
    Локальный хаб для управления моделями KokaoCore.
    """

    def __init__(self, hub_dir: str = "~/.kokao_hub"):
        """
        Инициализация хаба.

        Args:
            hub_dir: Директория хаба
        """
        self.hub_dir = Path(hub_dir).expanduser()
        self.models_dir = self.hub_dir / "models"
        self.registry_path = self.hub_dir / "registry.json"

        # Создание директорий
        self.hub_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Загрузка реестра
        self.registry: Dict[str, ModelRegistryEntry] = {}
        self._load_registry()

        logger.info(f"KokaoHub initialized at {self.hub_dir}")

    def _load_registry(self) -> None:
        """Загрузка реестра."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for model_id, entry_data in data.items():
                    info = ModelInfo.from_dict(entry_data['info'])
                    self.registry[model_id] = ModelRegistryEntry(
                        info=info,
                        weights_path=entry_data['weights_path'],
                        is_active=entry_data.get('is_active', True)
                    )

    def _save_registry(self) -> None:
        """Сохранение реестра."""
        data = {
            model_id: {
                'info': entry.info.to_dict(),
                'weights_path': entry.weights_path,
                'is_active': entry.is_active
            }
            for model_id, entry in self.registry.items()
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_model_id(self, core: KokaoCore) -> str:
        """
        Генерация уникального ID модели.

        Args:
            core: Модель

        Returns:
            Уникальный ID
        """
        # Хеш от весов
        with torch.no_grad():
            w_plus_hash = hashlib.md5(
                core.w_plus.cpu().numpy().tobytes()
            ).hexdigest()[:8]
            w_minus_hash = hashlib.md5(
                core.w_minus.cpu().numpy().tobytes()
            ).hexdigest()[:8]

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"kokao_{core.config.input_dim}d_{timestamp}_{w_plus_hash}{w_minus_hash}"

    def register_model(self, core: KokaoCore, name: str,
                       description: str = "",
                       tags: Optional[List[str]] = None,
                       metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Регистрация модели в хабе.

        Args:
            core: Модель для регистрации
            name: Имя модели
            description: Описание
            tags: Теги
            metrics: Метрики качества

        Returns:
            ID модели
        """
        model_id = self._generate_model_id(core)

        # Сохранение весов
        weights_path = self.models_dir / f"{model_id}.json"
        core.save(str(weights_path))

        # Создание записи
        info = ModelInfo(
            model_id=model_id,
            name=name,
            description=description,
            input_dim=core.config.input_dim,
            tags=tags or [],
            metrics=metrics or {},
            config=core.config.dict(),
            version=1
        )

        entry = ModelRegistryEntry(
            info=info,
            weights_path=str(weights_path),
            is_active=True
        )

        self.registry[model_id] = entry
        self._save_registry()

        logger.info(f"Registered model: {model_id}")
        return model_id

    def load_model(self, model_id: str, 
                   device: Optional[str] = None) -> KokaoCore:
        """
        Загрузка модели из хаба.

        Args:
            model_id: ID модели
            device: Устройство для загрузки

        Returns:
            Загруженная модель
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found in registry")

        entry = self.registry[model_id]

        if not Path(entry.weights_path).exists():
            raise ValueError(f"Weights file not found: {entry.weights_path}")

        core = KokaoCore.load(entry.weights_path, device=device)
        logger.info(f"Loaded model: {model_id}")
        return core

    def list_models(self, tags: Optional[List[str]] = None,
                    active_only: bool = True) -> List[ModelInfo]:
        """
        Список моделей.

        Args:
            tags: Фильтр по тегам
            active_only: Только активные модели

        Returns:
            Список информации о моделях
        """
        models = []

        for model_id, entry in self.registry.items():
            if active_only and not entry.is_active:
                continue

            if tags:
                if not any(tag in entry.info.tags for tag in tags):
                    continue

            models.append(entry.info)

        return models

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Получение информации о модели."""
        if model_id in self.registry:
            return self.registry[model_id].info
        return None

    def update_model_metrics(self, model_id: str,
                             metrics: Dict[str, float]) -> None:
        """
        Обновление метрик модели.

        Args:
            model_id: ID модели
            metrics: Новые метрики
        """
        if model_id not in self.registry:
            raise ValueError(f"Model {model_id} not found")

        self.registry[model_id].info.metrics.update(metrics)
        self.registry[model_id].info.updated_at = datetime.now().isoformat()
        self._save_registry()

    def delete_model(self, model_id: str) -> bool:
        """
        Удаление модели.

        Args:
            model_id: ID модели

        Returns:
            True если удалена
        """
        if model_id not in self.registry:
            return False

        entry = self.registry[model_id]

        # Удаление файла весов
        if Path(entry.weights_path).exists():
            Path(entry.weights_path).unlink()

        # Удаление из реестра
        del self.registry[model_id]
        self._save_registry()

        logger.info(f"Deleted model: {model_id}")
        return True

    def export_registry(self, output_path: str) -> None:
        """
        Экспорт реестра.

        Args:
            output_path: Путь для экспорта
        """
        data = {
            model_id: entry.info.to_dict()
            for model_id, entry in self.registry.items()
            if entry.is_active
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def import_registry(self, input_path: str, 
                        merge: bool = False) -> int:
        """
        Импорт реестра.

        Args:
            input_path: Путь к файлу реестра
            merge: Объединить с существующим

        Returns:
            Количество импортированных моделей
        """
        with open(input_path) as f:
            data = json.load(f)

        if not merge:
            self.registry.clear()

        count = 0
        for model_id, info_data in data.items():
            if model_id not in self.registry:
                info = ModelInfo.from_dict(info_data)
                self.registry[model_id] = ModelRegistryEntry(
                    info=info,
                    weights_path="",  # Пустой путь для импортированных
                    is_active=True
                )
                count += 1

        self._save_registry()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики хаба.

        Returns:
            Статистика
        """
        active_models = sum(1 for e in self.registry.values() if e.is_active)

        return {
            'total_models': len(self.registry),
            'active_models': active_models,
            'hub_dir': str(self.hub_dir),
            'registry_size': self.registry_path.stat().st_size if self.registry_path.exists() else 0
        }


class ModelZoo:
    """
    Каталог предобученных моделей.
    """

    def __init__(self):
        """Инициализация каталога."""
        self.models: Dict[str, Dict[str, Any]] = {}
        self._load_default_models()

    def _load_default_models(self) -> None:
        """Загрузка моделей по умолчанию."""
        self.models = {
            'random_classifier': {
                'description': 'Случайный классификатор для тестирования',
                'input_dim': 10,
                'tags': ['test', 'random']
            },
            'anomaly_detector': {
                'description': 'Детектор аномалий на основе KokaoCore',
                'input_dim': 16,
                'tags': ['anomaly', 'detection']
            }
        }

    def list_available(self) -> List[Dict[str, Any]]:
        """Список доступных моделей."""
        return list(self.models.values())

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Получение конфигурации модели."""
        return self.models.get(model_name)


def create_hub(hub_dir: str = "~/.kokao_hub") -> KokaoHub:
    """
    Создание KokaoHub.

    Args:
        hub_dir: Директория хаба

    Returns:
        KokaoHub
    """
    return KokaoHub(hub_dir)


def quick_register(core: KokaoCore, name: str,
                   hub_dir: str = "~/.kokao_hub") -> str:
    """
    Быстрая регистрация модели.

    Args:
        core: Модель
        name: Имя модели
        hub_dir: Директория хаба

    Returns:
        ID модели
    """
    hub = KokaoHub(hub_dir)
    return hub.register_model(core, name)

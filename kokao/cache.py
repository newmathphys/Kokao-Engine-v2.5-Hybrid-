"""
Модуль кэширования результатов инверсии.
Основан на lru_cache для ускорения повторных вычислений.
"""

import torch
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from .core import KokaoCore
from .inverse import InverseProblem


class InversionCache:
    """Кэш для результатов обратной задачи."""

    def __init__(self, max_size: int = 1000, persist_dir: Optional[str] = None):
        """
        Инициализация кэша.

        Args:
            max_size: Максимальный размер кэша
            persist_dir: Директория для постоянного хранения (опционально)
        """
        self.max_size = max_size
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, weights: Tuple[torch.Tensor, torch.Tensor],
                  S_target: float, clamp_range: Tuple[float, float]) -> str:
        """Создание хеш-ключа для кэша."""
        key_data = {
            'w_plus': weights[0].detach().cpu().numpy().tobytes(),
            'w_minus': weights[1].detach().cpu().numpy().tobytes(),
            'S_target': S_target,
            'clamp_range': clamp_range
        }
        key_json = json.dumps({
            'w_plus_hash': hashlib.md5(key_data['w_plus']).hexdigest(),
            'w_minus_hash': hashlib.md5(key_data['w_minus']).hexdigest(),
            'S_target': S_target,
            'clamp_range': clamp_range
        }, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Получение значения из кэша.

        Args:
            key: Кэш-ключ

        Returns:
            Кэшированный вектор или None
        """
        if key in self.cache:
            # LRU: перемещаем в конец (свежий)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        """
        Сохранение значения в кэш.

        Args:
            key: Кэш-ключ
            value: Вектор для сохранения
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Удаляем старые записи при переполнении
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def save_to_disk(self, key: str, value: torch.Tensor) -> str:
        """
        Сохранение значения на диск.

        Args:
            key: Кэш-ключ
            value: Вектор для сохранения

        Returns:
            Путь к файлу
        """
        if not self.persist_dir:
            raise ValueError("persist_dir not set")

        file_path = self.persist_dir / f"{key}.pt"
        torch.save(value, file_path)
        return str(file_path)

    def load_from_disk(self, key: str) -> Optional[torch.Tensor]:
        """
        Загрузка значения с диска.

        Args:
            key: Кэш-ключ

        Returns:
            Загруженный вектор или None
        """
        if not self.persist_dir:
            return None

        file_path = self.persist_dir / f"{key}.pt"
        if file_path.exists():
            return torch.load(file_path, weights_only=True)
        return None

    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()

    def size(self) -> int:
        """Текущий размер кэша."""
        return len(self.cache)

    def stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'persist_dir': str(self.persist_dir) if self.persist_dir else None
        }


class CachedInverseProblem:
    """Обёртка для InverseProblem с кэшированием."""

    def __init__(self, core: KokaoCore, cache: Optional[InversionCache] = None):
        """
        Инициализация кэшируемой обратной задачи.

        Args:
            core: Экземпляр KokaoCore
            cache: Кэш (если None, создаётся новый)
        """
        self.core = core
        self.cache = cache or InversionCache()
        self.hits = 0
        self.misses = 0

    def solve(
        self,
        S_target: float,
        x_init: Optional[torch.Tensor] = None,
        lr: float = 0.1,
        max_steps: int = 100,
        clamp_range: Tuple[float, float] = (-1.0, 1.0),
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Решение обратной задачи с кэшированием.

        Args:
            S_target: Целевой сигнал
            x_init: Начальное приближение
            lr: Скорость обучения
            max_steps: Максимум шагов
            clamp_range: Диапазон ограничений
            use_cache: Использовать ли кэш

        Returns:
            Вектор решения
        """
        if use_cache:
            # Создаём ключ кэша
            eff_w_plus, eff_w_minus = self.core._get_effective_weights()
            key = self.cache._hash_key(
                (eff_w_plus, eff_w_minus),
                S_target,
                clamp_range
            )

            # Проверяем кэш
            cached_result = self.cache.get(key)
            if cached_result is not None:
                self.hits += 1
                return cached_result.clone()

            self.misses += 1

        # Решаем обратную задачу
        inverse = self.core.to_inverse_problem()
        result = inverse.solve(
            S_target=S_target,
            x_init=x_init,
            lr=lr,
            max_steps=max_steps,
            clamp_range=clamp_range
        )

        # Сохраняем в кэш
        if use_cache:
            self.cache.put(key, result.clone())

        return result

    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэширования."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': self.cache.size()
        }

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

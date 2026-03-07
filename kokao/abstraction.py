"""
Модуль абстрактного мышления – выделение общих признаков.
Основан на книге Косякова Ю.Б., п.4.3.3.
"""

import torch
from typing import List
from .core import KokaoCore


class AbstractionEngine:
    """Создание абстрактных прототипов из набора примеров."""

    def __init__(self, core: KokaoCore):
        """
        Инициализация движка абстракции.

        Args:
            core: Экземпляр KokaoCore
        """
        self.core = core

    def extract_prototype(
        self,
        examples: List[torch.Tensor],
        method: str = "mean",
        refine_steps: int = 20
    ) -> torch.Tensor:
        """
        Создаёт прототип из списка примеров.

        Args:
            examples: Список векторов примеров
            method: Метод экстракции ("mean", "median", "pca")
            refine_steps: Шагов уточнения через обратную задачу

        Returns:
            Вектор прототипа

        Raises:
            ValueError: Если метод неизвестен
        """
        if not examples:
            return torch.zeros(self.core.config.input_dim)

        stack = torch.stack(examples)

        if method == "mean":
            proto = stack.mean(dim=0)
        elif method == "median":
            proto = stack.median(dim=0).values
        elif method == "pca":
            mean = stack.mean(dim=0)
            centered = stack - mean
            cov = centered.T @ centered / (len(examples) - 1)
            _, _, v = torch.svd_lowrank(cov, q=1)
            proto = mean + v[:, 0] * torch.std(stack)
        else:
            raise ValueError(f"Unknown method: {method}")

        solver = self.core.to_inverse_problem()
        target_s = self.core.signal(proto)
        return solver.solve(S_target=target_s, max_steps=refine_steps)

    def hierarchical_abstraction(
        self,
        groups: List[List[torch.Tensor]],
        levels: int = 2
    ) -> List[torch.Tensor]:
        """
        Строит иерархию абстракций.

        Args:
            groups: Список групп примеров
            levels: Количество уровней иерархии

        Returns:
            Список уровней прототипов
        """
        current = [self.extract_prototype(g) for g in groups]
        prototypes = [current]

        for _ in range(1, levels):
            if len(current) == 1:
                break

            new_protos = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    combined = self.extract_prototype([current[i], current[i + 1]])
                else:
                    combined = current[i]
                new_protos.append(combined)

            prototypes.append(new_protos)
            current = new_protos

        return prototypes

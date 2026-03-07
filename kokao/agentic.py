"""Модуль Agentic AI для создания автономных агентов на основе KokaoCore."""
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from .core import KokaoCore
from .core_base import CoreConfig
from .decoder import Decoder

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Состояния агента."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    LEARNING = "learning"


@dataclass
class AgentMemory:
    """Память агента."""
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    max_experiences: int = 1000

    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool) -> None:
        """Добавление опыта."""
        if len(self.experiences) >= self.max_experiences:
            self.experiences.pop(0)

        self.experiences.append({
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'done': done
        })

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Сэмплирование случайного батча опытов."""
        if len(self.experiences) < batch_size:
            return self.experiences.copy()

        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def clear(self) -> None:
        """Очистка памяти."""
        self.experiences.clear()


class KokaoAgent:
    """
    Автономный агент на основе KokaoCore.
    Использует сигнал ядра для оценки действий и принятия решений.
    """

    def __init__(self, state_dim: int, num_actions: int, 
                 hidden_dim: int = 32, lr: float = 0.01):
        """
        Инициализация агента.

        Args:
            state_dim: Размерность пространства состояний
            num_actions: Количество возможных действий
            hidden_dim: Размерность скрытого представления
            lr: Скорость обучения
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Входное преобразование
        self.state_encoder = torch.nn.Linear(state_dim, hidden_dim)

        # Ядро для каждого действия
        self.cores = nn.ModuleList([
            KokaoCore(CoreConfig(input_dim=hidden_dim))
            for _ in range(num_actions)
        ])

        # Декодеры для генерации действий
        self.decoders = [Decoder(core, lr=lr) for core in self.cores]

        # Память
        self.memory = AgentMemory()

        # Состояние
        self.current_state: Optional[np.ndarray] = None
        self.state = AgentState.IDLE

        # Статистика
        self.total_reward = 0.0
        self.episode_count = 0

    def select_action(self, state: np.ndarray, 
                      exploration_rate: float = 0.1) -> int:
        """
        Выбор действия на основе состояния.

        Args:
            state: Текущее состояние среды
            exploration_rate: Вероятность случайного действия (epsilon-greedy)

        Returns:
            Индекс выбранного действия
        """
        self.state = AgentState.THINKING

        # Epsilon-greedy стратегия
        if np.random.random() < exploration_rate:
            self.state = AgentState.ACTING
            return np.random.randint(self.num_actions)

        # Преобразование состояния
        state_tensor = torch.tensor(state, dtype=torch.float32)
        hidden = torch.relu(self.state_encoder(state_tensor))

        # Вычисление сигналов для каждого действия
        signals = []
        for core in self.cores:
            signal = core.signal(hidden)
            signals.append(signal)

        # Выбор действия с максимальным сигналом
        action = int(torch.argmax(torch.tensor(signals)).item())

        self.state = AgentState.ACTING
        return action

    def act(self, action: int, env: Any) -> Tuple[float, np.ndarray, bool]:
        """
        Выполнение действия в среде.

        Args:
            action: Индекс действия
            env: Среда

        Returns:
            (reward, next_state, done)
        """
        self.state = AgentState.ACTING
        reward, next_state, done, _ = env.step(action)
        self.state = AgentState.OBSERVING
        return reward, next_state, done

    def observe(self, state: np.ndarray, action: int, 
                reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Наблюдение результата действия.

        Args:
            state: Предыдущее состояние
            action: Выполненное действие
            reward: Полученная награда
            next_state: Новое состояние
            done: Флаг завершения эпизода
        """
        self.memory.add(state, action, reward, next_state, done)
        self.total_reward += reward

        if done:
            self.episode_count += 1
            logger.info(f"Episode {self.episode_count} completed. "
                       f"Total reward: {self.total_reward:.2f}")
            self.total_reward = 0.0

        self.state = AgentState.IDLE

    def learn(self, batch_size: int = 32, target_reward: float = 1.0) -> float:
        """
        Обучение на основе накопленного опыта.

        Args:
            batch_size: Размер батча для обучения
            target_reward: Целевая награда для обучения

        Returns:
            Среднее значение потерь
        """
        self.state = AgentState.LEARNING

        if len(self.memory.experiences) < batch_size:
            return 0.0

        # Сэмплирование батча
        batch = self.memory.sample(batch_size)

        total_loss = 0.0
        for experience in batch:
            state = np.array(experience['state'])
            action = experience['action']
            reward = experience['reward']

            # Преобразование состояния
            state_tensor = torch.tensor(state, dtype=torch.float32)
            hidden = torch.relu(self.state_encoder(state_tensor))

            # Обучение ядра для выбранного действия
            target = target_reward if reward > 0 else -target_reward
            loss = self.cores[action].train_adam(hidden, target)
            total_loss += loss

        self.state = AgentState.IDLE
        return total_loss / len(batch)

    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """
        Получение политики для данного состояния.

        Args:
            state: Состояние

        Returns:
            Вероятности действий (softmax от сигналов)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        hidden = torch.relu(self.state_encoder(state_tensor))

        signals = []
        for core in self.cores:
            signal = core.signal(hidden)
            signals.append(signal)

        # Softmax для получения вероятностей
        signals_tensor = torch.tensor(signals)
        probs = torch.softmax(signals_tensor, dim=0).numpy()

        return probs

    def save(self, path: str) -> None:
        """Сохранение агента."""
        state_dict = {
            'state_encoder': self.state_encoder.state_dict(),
            'cores': [core.state_dict() for core in self.cores],
            'memory': self.memory.experiences,
            'total_reward': self.total_reward,
            'episode_count': self.episode_count
        }
        torch.save(state_dict, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str) -> None:
        """Загрузка агента."""
        state_dict = torch.load(path)
        self.state_encoder.load_state_dict(state_dict['state_encoder'])
        for core, core_state in zip(self.cores, state_dict['cores']):
            core.load_state_dict(core_state)
        self.memory.experiences = state_dict['memory']
        self.total_reward = state_dict['total_reward']
        self.episode_count = state_dict['episode_count']
        logger.info(f"Agent loaded from {path}")


class MultiAgentSystem:
    """
    Система множественных агентов для координации действий.
    """

    def __init__(self, num_agents: int, state_dim: int, 
                 num_actions: int, communication_dim: int = 16):
        """
        Инициализация мульти-агентной системы.

        Args:
            num_agents: Количество агентов
            state_dim: Размерность пространства состояний
            num_actions: Количество действий на агента
            communication_dim: Размерность пространства коммуникации
        """
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.num_actions = num_actions

        # Создание агентов
        self.agents = [
            KokaoAgent(state_dim, num_actions)
            for _ in range(num_agents)
        ]

        # Коммуникационный слой
        self.communication = torch.nn.Linear(
            num_agents * hidden_dim, 
            num_agents * communication_dim
        )

    def coordinate(self, states: List[np.ndarray]) -> List[int]:
        """
        Координация действий между агентами.

        Args:
            states: Состояния всех агентов

        Returns:
            Скоординированные действия
        """
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.select_action(state)
            actions.append(action)

        # Здесь может быть логика координации
        return actions


# Импорт nn в начале файла
import torch.nn as nn

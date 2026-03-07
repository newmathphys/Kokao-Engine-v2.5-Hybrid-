"""Модуль квантовых нейронных сетей на основе KokaoCore."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path
from dataclasses import dataclass

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class QuantumState:
    """Квантовое состояние (вектор состояния)."""
    amplitudes: torch.Tensor  # Комплексные амплитуды

    def __post_init__(self):
        # Нормализация состояния
        norm = torch.sqrt(torch.sum(torch.abs(self.amplitudes) ** 2))
        self.amplitudes = self.amplitudes / norm

    @property
    def probabilities(self) -> torch.Tensor:
        """Вероятности измерения базисных состояний."""
        return torch.abs(self.amplitudes) ** 2

    def measure(self) -> int:
        """
        Измерение квантового состояния.

        Returns:
            Индекс измеренного состояния
        """
        probs = self.probabilities.detach().numpy()
        probs = probs / probs.sum()  # Нормализация
        return np.random.choice(len(probs), p=probs)


class QuantumGate:
    """Базовый класс для квантовых ворот."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dimension = 2 ** num_qubits

    def apply(self, state: QuantumState) -> QuantumState:
        """Применение ворота к состоянию."""
        raise NotImplementedError

    def get_matrix(self) -> torch.Tensor:
        """Получение матрицы ворота."""
        raise NotImplementedError


class HadamardGate(QuantumGate):
    """Ворота Адамара."""

    def __init__(self, qubit: int = 0):
        super().__init__(1)
        self.qubit = qubit
        self._matrix = torch.tensor([
            [1, 1],
            [1, -1]
        ], dtype=torch.complex64) / np.sqrt(2)

    def get_matrix(self) -> torch.Tensor:
        return self._matrix

    def apply(self, state: QuantumState) -> QuantumState:
        # Упрощенная реализация для одного кубита
        if self.num_qubits == 1:
            new_amplitudes = self._matrix @ state.amplitudes
            return QuantumState(new_amplitudes)
        return state


class PhaseGate(QuantumGate):
    """Фазовые ворота."""

    def __init__(self, phase: float = 0.0):
        super().__init__(1)
        self.phase = phase
        self._matrix = torch.tensor([
            [1, 0],
            [0, torch.exp(1j * torch.tensor(phase))]
        ], dtype=torch.complex64)

    def get_matrix(self) -> torch.Tensor:
        return self._matrix

    def apply(self, state: QuantumState) -> QuantumState:
        if self.num_qubits == 1:
            new_amplitudes = self._matrix @ state.amplitudes
            return QuantumState(new_amplitudes)
        return state


class CNOTGate(QuantumGate):
    """Ворота CNOT (controlled-NOT)."""

    def __init__(self, control: int = 0, target: int = 1):
        super().__init__(2)
        self.control = control
        self.target = target
        self._matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.complex64)

    def get_matrix(self) -> torch.Tensor:
        return self._matrix

    def apply(self, state: QuantumState) -> QuantumState:
        if self.num_qubits == 2:
            new_amplitudes = self._matrix @ state.amplitudes
            return QuantumState(new_amplitudes)
        return state


class QuantumCircuit:
    """
    Квантовая схема (последовательность ворот).
    """

    def __init__(self, num_qubits: int):
        """
        Инициализация схемы.

        Args:
            num_qubits: Количество кубитов
        """
        self.num_qubits = num_qubits
        self.gates: List[QuantumGate] = []

    def add_gate(self, gate: QuantumGate) -> 'QuantumCircuit':
        """Добавление ворота в схему."""
        self.gates.append(gate)
        return self

    def execute(self, initial_state: Optional[QuantumState] = None) -> QuantumState:
        """
        Выполнение схемы.

        Args:
            initial_state: Начальное состояние (по умолчанию |00...0⟩)

        Returns:
            Конечное состояние
        """
        if initial_state is None:
            # Начальное состояние |00...0⟩
            amplitudes = torch.zeros(2 ** self.num_qubits, dtype=torch.complex64)
            amplitudes[0] = 1.0
            state = QuantumState(amplitudes)
        else:
            state = initial_state

        # Применение всех ворот
        for gate in self.gates:
            state = gate.apply(state)

        return state

    def measure(self, state: Optional[QuantumState] = None) -> int:
        """
        Измерение результата выполнения схемы.

        Args:
            state: Состояние для измерения (если None, выполняется схема)

        Returns:
            Результат измерения
        """
        if state is None:
            state = self.execute()
        return state.measure()

    def get_statevector(self) -> torch.Tensor:
        """Получение вектора состояния после выполнения схемы."""
        state = self.execute()
        return state.amplitudes


class KokaoQuantumNetwork:
    """
    Квантовая нейронная сеть на основе KokaoCore.
    Использует квантовые схемы для кодирования и KokaoCore для обработки.
    """

    def __init__(self, num_qubits: int = 4, input_dim: int = 4,
                 hidden_dim: int = 16, lr: float = 0.01):
        """
        Инициализация квантовой сети.

        Args:
            num_qubits: Количество кубитов
            input_dim: Размерность входа
            hidden_dim: Размерность скрытого слоя
            lr: Скорость обучения
        """
        self.num_qubits = num_qubits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Параметры для вариационных ворот
        self.variational_params = nn.Parameter(
            torch.randn(num_qubits * 3) * 0.1
        )

        # Квантовая схема для кодирования
        self.encoding_circuit = self._create_encoding_circuit()

        # KokaoCore для обработки квантовых признаков
        self.core = KokaoCore(CoreConfig(input_dim=hidden_dim))

        # Выходной слой
        self.output_layer = nn.Linear(hidden_dim, 1)

        # Оптимизатор
        self.optimizer = torch.optim.Adam(
            [self.variational_params] + list(self.output_layer.parameters()),
            lr=lr
        )

        # История обучения
        self.history = {'loss': []}

    def _create_encoding_circuit(self) -> QuantumCircuit:
        """Создание схемы кодирования."""
        circuit = QuantumCircuit(self.num_qubits)

        # Слой Адамара
        for i in range(self.num_qubits):
            circuit.add_gate(HadamardGate(i))

        return circuit

    def _create_variational_circuit(self, params: torch.Tensor) -> QuantumCircuit:
        """Создание вариационной схемы."""
        circuit = QuantumCircuit(self.num_qubits)

        # Вариационные вращения
        for i in range(self.num_qubits):
            circuit.add_gate(PhaseGate(params[i].item()))
            circuit.add_gate(HadamardGate(i))
            circuit.add_gate(PhaseGate(params[self.num_qubits + i].item()))

        # CNOT между соседними кубитами
        for i in range(self.num_qubits - 1):
            circuit.add_gate(CNOTGate(i, i + 1))

        return circuit

    def encode_input(self, x: torch.Tensor) -> QuantumState:
        """
        Кодирование входа в квантовое состояние.

        Args:
            x: Входные данные

        Returns:
            Квантовое состояние
        """
        # Нормализация входа
        x_normalized = x / (torch.norm(x) + 1e-10)

        # Создание начального состояния
        amplitudes = torch.zeros(2 ** self.num_qubits, dtype=torch.complex64)

        # Закодировать вход в амплитуды
        for i in range(min(len(x_normalized), self.num_qubits)):
            amplitudes[i] = x_normalized[i]

        # Применение схемы кодирования
        initial_state = QuantumState(amplitudes)
        encoded_state = self.encoding_circuit.execute(initial_state)

        return encoded_state

    def extract_features(self, state: QuantumState) -> torch.Tensor:
        """
        Извлечение признаков из квантового состояния.

        Args:
            state: Квантовое состояние

        Returns:
            Вектор признаков
        """
        # Вероятности измерения
        probs = state.probabilities

        # Реальные и мнимые части амплитуд
        real_parts = torch.real(state.amplitudes)
        imag_parts = torch.imag(state.amplitudes)

        # Конкатенация признаков
        features = torch.cat([probs, real_parts, imag_parts])

        # Проекция на нужную размерность
        if len(features) > self.hidden_dim:
            features = features[:self.hidden_dim]
        elif len(features) < self.hidden_dim:
            features = torch.cat([
                features,
                torch.zeros(self.hidden_dim - len(features))
            ])

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход сети.

        Args:
            x: Входные данные

        Returns:
            Выход сети
        """
        # Кодирование входа
        quantum_state = self.encode_input(x)

        # Применение вариационной схемы
        var_circuit = self._create_variational_circuit(self.variational_params)
        processed_state = var_circuit.execute(quantum_state)

        # Извлечение признаков
        features = self.extract_features(processed_state)

        # Обработка через KokaoCore
        core_signal = self.core.signal(features)

        # Выходной слой
        output = self.output_layer(torch.tensor([core_signal]))

        return output

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Один шаг обучения.

        Args:
            x: Входные данные
            y: Целевое значение

        Returns:
            Значение потерь
        """
        self.optimizer.zero_grad()

        output = self.forward(x)
        loss = nn.functional.mse_loss(output, y)

        loss.backward()
        self.optimizer.step()

        self.history['loss'].append(loss.item())
        return loss.item()

    def train(self, data: List[Tuple[torch.Tensor, float]],
              num_epochs: int = 100, verbose: bool = True) -> List[float]:
        """
        Обучение сети.

        Args:
            data: Данные для обучения
            num_epochs: Количество эпох
            verbose: Выводить ли прогресс

        Returns:
            История потерь
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for x, y in data:
                loss = self.train_step(x, torch.tensor([y]))
                epoch_loss += loss

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: Loss = {epoch_loss / len(data):.4f}")

        return self.history['loss']

    def predict(self, x: torch.Tensor) -> float:
        """Предсказание для одного входа."""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
        return output.item()

    def eval(self) -> None:
        """Перевод в режим оценки."""
        self.output_layer.eval()

    def save(self, path: str) -> None:
        """Сохранение модели."""
        state = {
            'variational_params': self.variational_params.detach().cpu().tolist(),
            'output_layer': self.output_layer.state_dict(),
            'core': self.core.state_dict(),
            'history': self.history
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Загрузка модели."""
        state = torch.load(path)
        self.variational_params.data = torch.tensor(state['variational_params'])
        self.output_layer.load_state_dict(state['output_layer'])
        self.core.load_state_dict(state['core'])
        self.history = state['history']


def create_bell_state() -> QuantumState:
    """
    Создание запутанного состояния Белла.

    Returns:
        Состояние Белла (|00⟩ + |11⟩) / √2
    """
    amplitudes = torch.zeros(4, dtype=torch.complex64)
    amplitudes[0] = 1 / np.sqrt(2)
    amplitudes[3] = 1 / np.sqrt(2)
    return QuantumState(amplitudes)


def create_ghz_state(num_qubits: int) -> QuantumState:
    """
    Создание состояния GHZ.

    Args:
        num_qubits: Количество кубитов

    Returns:
        Состояние GHZ (|00...0⟩ + |11...1⟩) / √2
    """
    dimension = 2 ** num_qubits
    amplitudes = torch.zeros(dimension, dtype=torch.complex64)
    amplitudes[0] = 1 / np.sqrt(2)
    amplitudes[-1] = 1 / np.sqrt(2)
    return QuantumState(amplitudes)

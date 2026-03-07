"""Модуль спайковых нейронных сетей (SNN) на основе KokaoCore."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import deque

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


@dataclass
class SpikeTrain:
    """
    Спайковый поезд (последовательность спайков во времени).
    """
    times: torch.Tensor  # Времена спайков
    duration: float  # Общая длительность

    def to_tensor(self, time_bins: int) -> torch.Tensor:
        """
        Конвертация в бинарный тензор.

        Args:
            time_bins: Количество временных бинов

        Returns:
            Бинарный тензор (time_bins,)
        """
        tensor = torch.zeros(time_bins)
        bin_size = self.duration / time_bins

        for t in self.times:
            bin_idx = int(t / bin_size)
            if 0 <= bin_idx < time_bins:
                tensor[bin_idx] = 1

        return tensor

    @property
    def firing_rate(self) -> float:
        """Частота спайков (спайков в секунду)."""
        if self.duration == 0:
            return 0.0
        return len(self.times) / self.duration


class LeakyIntegrateAndFire(nn.Module):
    """
    Модель нейрона с утечкой и порогом (LIF).
    """

    def __init__(self, threshold: float = 1.0, decay: float = 0.9,
                 reset_value: float = 0.0):
        """
        Инициализация LIF нейрона.

        Args:
            threshold: Порог срабатывания
            decay: Коэффициент затухания мембранного потенциала
            reset_value: Значение сброса после спайка
        """
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.reset_value = reset_value

        # Мембранный потенциал
        self.register_buffer('membrane_potential', torch.tensor(0.0))

        # История спайков
        self.spike_history: deque = deque(maxlen=1000)

    def forward(self, input_current: float) -> Tuple[float, bool]:
        """
        Один шаг симуляции нейрона.

        Args:
            input_current: Входной ток

        Returns:
            (новый мембранный потенциал, был ли спайк)
        """
        # Утечка
        self.membrane_potential *= self.decay

        # Интеграция входа
        self.membrane_potential += input_current

        # Проверка порога
        spiked = self.membrane_potential >= self.threshold

        if spiked:
            self.membrane_potential = torch.tensor(self.reset_value)
            self.spike_history.append(True)
        else:
            self.spike_history.append(False)

        return self.membrane_potential.item(), spiked

    def reset(self) -> None:
        """Сброс состояния нейрона."""
        self.membrane_potential = torch.tensor(0.0)
        self.spike_history.clear()

    def get_spike_train(self, dt: float = 0.001) -> SpikeTrain:
        """
        Получение спайкового поезда.

        Args:
            dt: Временной шаг

        Returns:
            SpikeTrain
        """
        spike_times = []
        for i, spiked in enumerate(self.spike_history):
            if spiked:
                spike_times.append(i * dt)

        return SpikeTrain(
            times=torch.tensor(spike_times),
            duration=len(self.spike_history) * dt
        )


class SpikeEncoding:
    """
    Методы кодирования входных данных в спайки.
    """

    @staticmethod
    def rate_encoding(values: np.ndarray, max_rate: float = 100.0,
                      duration: float = 1.0) -> List[SpikeTrain]:
        """
        Кодирование частотой спайков.

        Args:
            values: Входные значения
            max_rate: Максимальная частота спайков
            duration: Длительность симуляции

        Returns:
            Список спайковых поездов
        """
        spike_trains = []

        for value in values:
            # Нормализация значения
            normalized = (value - values.min()) / (values.max() - values.min() + 1e-10)
            rate = normalized * max_rate

            # Генерация спайков по Пуассону
            expected_spikes = rate * duration
            num_spikes = np.random.poisson(expected_spikes)

            spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
            spike_trains.append(SpikeTrain(
                times=torch.tensor(spike_times),
                duration=duration
            ))

        return spike_trains

    @staticmethod
    def temporal_encoding(values: np.ndarray, 
                          duration: float = 1.0) -> List[SpikeTrain]:
        """
        Временное кодирование (более сильные значения = раньше спайк).

        Args:
            values: Входные значения
            duration: Длительность

        Returns:
            Список спайковых поездов
        """
        spike_trains = []

        # Нормализация
        normalized = (values - values.min()) / (values.max() - values.min() + 1e-10)

        for value in normalized:
            # Время спайка обратно пропорционально значению
            spike_time = (1 - value) * duration

            spike_trains.append(SpikeTrain(
                times=torch.tensor([spike_time]),
                duration=duration
            ))

        return spike_trains

    @staticmethod
    def binning_encoding(spike_trains: List[SpikeTrain],
                         num_bins: int) -> torch.Tensor:
        """
        Биннинг спайковых поездов.

        Args:
            spike_trains: Список спайковых поездов
            num_bins: Количество бинов

        Returns:
            Тензор (num_neurons, num_bins)
        """
        encoded = []
        for train in spike_trains:
            encoded.append(train.to_tensor(num_bins))

        return torch.stack(encoded)


class KokaoSNN(nn.Module):
    """
    Спайковая нейронная сеть на основе KokaoCore.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 10, num_layers: int = 2,
                 threshold: float = 1.0, decay: float = 0.9,
                 simulation_steps: int = 100):
        """
        Инициализация SNN.

        Args:
            input_dim: Размерность входа
            hidden_dim: Размерность скрытого слоя
            output_dim: Размерность выхода (количество классов)
            num_layers: Количество скрытых слоев
            threshold: Порог нейронов
            decay: Затухание мембранного потенциала
            simulation_steps: Количество шагов симуляции
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.simulation_steps = simulation_steps

        # Слои LIF нейронов
        self.input_neurons = nn.ModuleList([
            LeakyIntegrateAndFire(threshold, decay) for _ in range(input_dim)
        ])

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleList([
                LeakyIntegrateAndFire(threshold, decay) for _ in range(hidden_dim)
            ])
            self.hidden_layers.append(layer)

        self.output_neurons = nn.ModuleList([
            LeakyIntegrateAndFire(threshold, decay) for _ in range(output_dim)
        ])

        # Синаптические веса
        self.input_to_hidden = nn.Parameter(
            torch.randn(input_dim, hidden_dim) * 0.1
        )
        
        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
            for _ in range(num_layers - 1)
        ])

        self.hidden_to_output = nn.Parameter(
            torch.randn(hidden_dim, output_dim) * 0.1
        )

        # KokaoCore для модуляции весов
        self.modulator = KokaoCore(CoreConfig(input_dim=hidden_dim))

    def forward(self, spike_trains: List[SpikeTrain]) -> torch.Tensor:
        """
        Прямой проход SNN.

        Args:
            spike_trains: Входные спайковые поезда

        Returns:
            Выходные спайковые частоты для каждого класса
        """
        # Биннинг входных спайков
        input_bins = SpikeEncoding.binning_encoding(
            spike_trains, self.simulation_steps
        )  # (input_dim, simulation_steps)

        # Хранение выходных спайков
        output_spikes = torch.zeros(self.output_dim, self.simulation_steps)

        # Симуляция по времени
        for t in range(self.simulation_steps):
            # Входной слой
            input_spikes = input_bins[:, t]  # (input_dim,)

            # Скрытый слой 1
            hidden_input = torch.matmul(input_spikes, self.input_to_hidden)
            hidden_spikes = self._process_layer(self.hidden_layers[0], hidden_input)

            # Промежуточные скрытые слои
            current_hidden = hidden_spikes
            for i, layer in enumerate(self.hidden_layers[1:]):
                hidden_input = torch.matmul(current_hidden, self.hidden_weights[i])
                current_hidden = self._process_layer(layer, hidden_input)

            # Выходной слой
            output_input = torch.matmul(current_hidden, self.hidden_to_output)

            # Модуляция через KokaoCore
            mod_signal = self.modulator.signal(current_hidden)
            output_input *= (1 + mod_signal * 0.1)

            output_spikes[:, t] = self._process_layer(self.output_neurons, output_input)

        # Подсчет частоты спайков
        output_rates = output_spikes.sum(dim=1) / self.simulation_steps

        return output_rates

    def _process_layer(self, neurons: nn.ModuleList,
                       inputs: torch.Tensor) -> torch.Tensor:
        """
        Обработка слоя нейронов.

        Args:
            neurons: Список нейронов
            inputs: Входные токи

        Returns:
            Выходы нейронов (спайки)
        """
        outputs = []
        for i, neuron in enumerate(neurons):
            _, spiked = neuron(inputs[i].item() if i < len(inputs) else 0)
            outputs.append(1.0 if spiked else 0.0)

        return torch.tensor(outputs)

    def reset(self) -> None:
        """Сброс всех нейронов."""
        for neuron in self.input_neurons:
            neuron.reset()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()

    def train_step(self, spike_trains: List[SpikeTrain], 
                   target_class: int, lr: float = 0.01) -> float:
        """
        Один шаг обучения с использованием STDP-like правила.

        Args:
            spike_trains: Входные спайковые поезда
            target_class: Целевой класс
            lr: Скорость обучения

        Returns:
            Потери
        """
        self.train()
        self.reset()

        # Прямой проход
        output_rates = self.forward(spike_trains)

        # Потери (кросс-энтропия)
        target = torch.zeros(self.output_dim)
        target[target_class] = 1.0
        loss = nn.functional.cross_entropy(
            output_rates.unsqueeze(0), 
            torch.tensor([target_class])
        )

        # Обратный проход
        loss.backward()

        # Градиентный шаг
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param -= lr * param.grad
                    param.grad.zero_()

        return loss.item()

    def predict(self, spike_trains: List[SpikeTrain]) -> int:
        """
        Предсказание класса.

        Args:
            spike_trains: Входные спайковые поезда

        Returns:
            Предсказанный класс
        """
        self.eval()
        self.reset()

        with torch.no_grad():
            output_rates = self.forward(spike_trains)
            predicted_class = output_rates.argmax().item()

        return int(predicted_class)

    def save(self, path: str) -> None:
        """Сохранение модели."""
        state = {
            'input_to_hidden': self.input_to_hidden.detach().cpu().tolist(),
            'hidden_weights': [w.detach().cpu().tolist() for w in self.hidden_weights],
            'hidden_to_output': self.hidden_to_output.detach().cpu().tolist(),
            'modulator': self.modulator.state_dict()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Загрузка модели."""
        state = torch.load(path)
        self.input_to_hidden.data = torch.tensor(state['input_to_hidden'])
        for w, w_state in zip(self.hidden_weights, state['hidden_weights']):
            w.data = torch.tensor(w_state)
        self.hidden_to_output.data = torch.tensor(state['hidden_to_output'])
        self.modulator.load_state_dict(state['modulator'])


class STDPPlasticity:
    """
    Спайко-зависимая синаптическая пластичность (STDP).
    """

    def __init__(self, tau_plus: float = 20.0, tau_minus: float = 20.0,
                 a_plus: float = 0.01, a_minus: float = 0.012):
        """
        Инициализация STDP.

        Args:
            tau_plus: Время затухания для потенциации
            tau_minus: Время затухания для депрессии
            a_plus: Сила потенциации
            a_minus: Сила депрессии
        """
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

        # Трекеры спайков
        self.pre_spike_times: List[float] = []
        self.post_spike_times: List[float] = []

    def update(self, pre_spike: bool, post_spike: bool, 
               weight: float, time: float,
               max_weight: float = 1.0, min_weight: float = 0.0) -> float:
        """
        Обновление веса синапса.

        Args:
            pre_spike: Был ли пресинаптический спайк
            post_spike: Был ли постсинаптический спайк
            weight: Текущий вес
            time: Текущее время
            max_weight: Максимальный вес
            min_weight: Минимальный вес

        Returns:
            Новый вес
        """
        delta_weight = 0.0

        if pre_spike:
            self.pre_spike_times.append(time)
            # Депрессия от постсинаптических спайков
            for t_post in self.post_spike_times:
                if t_post < time:
                    delta_t = time - t_post
                    delta_weight -= self.a_minus * np.exp(-delta_t / self.tau_minus)

        if post_spike:
            self.post_spike_times.append(time)
            # Потенциация от пресинаптических спайков
            for t_pre in self.pre_spike_times:
                if t_pre < time:
                    delta_t = time - t_pre
                    delta_weight += self.a_plus * np.exp(-delta_t / self.tau_plus)

        # Ограничение весов
        new_weight = np.clip(weight + delta_weight, min_weight, max_weight)

        # Очистка старых спайков
        self._cleanup_old_spikes(time)

        return new_weight

    def _cleanup_old_spikes(self, current_time: float, 
                            window: float = 100.0) -> None:
        """Очистка старых спайков."""
        self.pre_spike_times = [
            t for t in self.pre_spike_times 
            if current_time - t < window
        ]
        self.post_spike_times = [
            t for t in self.post_spike_times 
            if current_time - t < window
        ]

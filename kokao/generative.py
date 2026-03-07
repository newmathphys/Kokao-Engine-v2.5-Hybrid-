"""Модуль генеративных моделей (GAN/VAE) на основе KokaoCore."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from pathlib import Path

from .core import KokaoCore
from .core_base import CoreConfig
from .decoder import Decoder

logger = logging.getLogger(__name__)


class KokaoGAN:
    """
    Генеративно-состязательная сеть с использованием KokaoCore.
    
    Generator: создает поддельные данные
    Discriminator: KokaoCore, различающий реальные и поддельные данные
    """

    def __init__(self, latent_dim: int = 100, data_dim: int = 10,
                 hidden_dim: int = 128, lr: float = 0.001):
        """
        Инициализация GAN.

        Args:
            latent_dim: Размерность латентного пространства
            data_dim: Размерность данных
            hidden_dim: Размерность скрытых слоев
            lr: Скорость обучения
        """
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Генератор: MLP
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Tanh()  # Выход в диапазоне [-1, 1]
        )

        # Дискриминатор: KokaoCore
        self.discriminator = KokaoCore(CoreConfig(input_dim=data_dim))

        # Оптимизаторы
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(
            [self.discriminator.w_plus, self.discriminator.w_minus], 
            lr=lr
        )

        # История обучения
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'd_real_acc': [],
            'd_fake_acc': []
        }

    def generate(self, z: Optional[torch.Tensor] = None, 
                 batch_size: int = 1) -> torch.Tensor:
        """
        Генерация поддельных данных.

        Args:
            z: Латентный вектор (если None, генерируется случайно)
            batch_size: Размер батча

        Returns:
            Сгенерированные данные
        """
        if z is None:
            z = torch.randn(batch_size, self.latent_dim)

        self.generator.eval()
        with torch.no_grad():
            fake_data = self.generator(z)

        return fake_data

    def discriminate(self, x: torch.Tensor) -> float:
        """
        Различение реальных и поддельных данных.

        Args:
            x: Данные для проверки

        Returns:
            Сигнал дискриминатора (выше = более реальные)
        """
        if x.ndim == 1:
            return self.discriminator.signal(x)

        # Для батча - средний сигнал
        signals = [self.discriminator.signal(x[i]) for i in range(len(x))]
        return np.mean(signals)

    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Один шаг обучения GAN.

        Args:
            real_data: Реальные данные

        Returns:
            Потери генератора и дискриминатора
        """
        batch_size = len(real_data)

        # =====================
        # Обучение дискриминатора
        # =====================
        self.discriminator.train()

        # Реальные данные
        d_real_loss = 0.0
        for x in real_data:
            loss = self.discriminator.train(x, target=1.0, lr=self.lr)
            d_real_loss += loss

        # Поддельные данные
        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)

        d_fake_loss = 0.0
        with torch.no_grad():
            for x in fake_data:
                loss = self.discriminator.train(x, target=0.0, lr=self.lr)
                d_fake_loss += loss

        d_loss = (d_real_loss + d_fake_loss) / (2 * batch_size)

        # Точность дискриминатора
        d_real_acc = sum(1 for x in real_data if self.discriminator.signal(x) > 0.5) / len(real_data)
        d_fake_acc = sum(1 for x in fake_data if self.discriminator.signal(x) < 0.5) / len(fake_data)

        # =====================
        # Обучение генератора
        # =====================
        self.generator.train()
        self.optimizer_g.zero_grad()

        z = torch.randn(batch_size, self.latent_dim)
        fake_data = self.generator(z)

        # Генератор хочет обмануть дискриминатор
        g_loss = 0.0
        for x in fake_data:
            # Градиент через сигнал дискриминатора
            signal = self.discriminator.forward(x)
            g_loss -= signal.mean()  # Максимизируем сигнал

        g_loss.backward()
        self.optimizer_g.step()

        # Сохранение истории
        self.history['g_loss'].append(g_loss.item())
        self.history['d_loss'].append(d_loss)
        self.history['d_real_acc'].append(d_real_acc)
        self.history['d_fake_acc'].append(d_fake_acc)

        return {'g_loss': g_loss.item(), 'd_loss': d_loss}

    def train(self, data: torch.Tensor, num_epochs: int, 
              batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Обучение GAN.

        Args:
            data: Данные для обучения
            num_epochs: Количество эпох
            batch_size: Размер батча
            verbose: Выводить ли прогресс

        Returns:
            История обучения
        """
        num_batches = len(data) // batch_size

        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for i in range(num_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                losses = self.train_step(batch)
                epoch_g_loss += losses['g_loss']
                epoch_d_loss += losses['d_loss']

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                           f"G Loss: {epoch_g_loss / num_batches:.4f}, "
                           f"D Loss: {epoch_d_loss / num_batches:.4f}")

        return self.history

    def save(self, path: str) -> None:
        """Сохранение модели."""
        state = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'history': self.history
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Загрузка модели."""
        state = torch.load(path)
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.history = state['history']


class KokaoVAE:
    """
    Вариационный автоэнкодер с использованием KokaoCore.

    Encoder: кодирует данные в параметры распределения (mu, log_var)
    Decoder: KokaoCore, восстанавливающий данные из латентного представления
    """

    def __init__(self, data_dim: int = 10, latent_dim: int = 2,
                 hidden_dim: int = 64, beta: float = 1.0):
        """
        Инициализация VAE.

        Args:
            data_dim: Размерность данных
            latent_dim: Размерность латентного пространства
            hidden_dim: Размерность скрытых слоев
            beta: Коэффициент KL-дивергенции
        """
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Энкодер: MLP -> (mu, log_var)
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        # Декодер: KokaoCore для каждого измерения данных
        self.decoders = nn.ModuleList([
            KokaoCore(CoreConfig(input_dim=latent_dim))
            for _ in range(data_dim)
        ])

        # Оптимизатор
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + 
            list(self.fc_mu.parameters()) + 
            list(self.fc_log_var.parameters()),
            lr=0.001
        )

        # История
        self.history = {
            'reconstruction_loss': [],
            'kl_loss': [],
            'total_loss': []
        }

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Кодирование данных в латентное пространство.

        Args:
            x: Входные данные

        Returns:
            (mu, log_var) параметры распределения
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, 
                       log_var: torch.Tensor) -> torch.Tensor:
        """
        Репараметризация для backprop.

        Args:
            mu: Среднее
            log_var: Логарифм дисперсии

        Returns:
            Сэмпл из распределения
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Декодирование из латентного пространства.

        Args:
            z: Латентный вектор

        Returns:
            Восстановленные данные
        """
        reconstructed = []
        for decoder in self.decoders:
            signal = decoder.signal(z)
            reconstructed.append(signal)

        return torch.tensor(reconstructed)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход VAE.

        Args:
            x: Входные данные

        Returns:
            (reconstructed, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def compute_loss(self, x: torch.Tensor, reconstructed: torch.Tensor,
                     mu: torch.Tensor, log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вычисление функции потерь VAE.

        Args:
            x: Исходные данные
            reconstructed: Восстановленные данные
            mu: Среднее
            log_var: Логарифм дисперсии

        Returns:
            (total_loss, reconstruction_loss, kl_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, x)

        # KL-дивергенция
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        # Общая потеря
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_step(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Один шаг обучения VAE.

        Args:
            x: Входные данные

        Returns:
            Потери
        """
        self.optimizer.zero_grad()

        reconstructed, mu, log_var = self.forward(x)
        total_loss, recon_loss, kl_loss = self.compute_loss(x, reconstructed, mu, log_var)

        total_loss.backward()
        self.optimizer.step()

        # Обучение декодеров
        z = self.reparameterize(mu, log_var).detach()
        for i, decoder in enumerate(self.decoders):
            decoder.train(z, target=x[i].item() if x.ndim == 1 else x[:, i].mean().item(), lr=0.001)

        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }

    def train(self, data: torch.Tensor, num_epochs: int,
              batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Обучение VAE.

        Args:
            data: Данные для обучения
            num_epochs: Количество эпох
            batch_size: Размер батча
            verbose: Выводить ли прогресс

        Returns:
            История обучения
        """
        num_batches = max(1, len(data) // batch_size)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for i in range(num_batches):
                batch = data[i * batch_size:(i + 1) * batch_size]
                losses = self.train_step(batch)
                epoch_loss += losses['total_loss']

                # Сохранение истории
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    self.history['reconstruction_loss'].append(losses['reconstruction_loss'])
                    self.history['kl_loss'].append(losses['kl_loss'])
                    self.history['total_loss'].append(losses['total_loss'])

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                           f"Loss: {epoch_loss / num_batches:.4f}")

        return self.history

    def generate(self, num_samples: int = 1) -> torch.Tensor:
        """
        Генерация новых данных.

        Args:
            num_samples: Количество сэмплов

        Returns:
            Сгенерированные данные
        """
        z = torch.randn(num_samples, self.latent_dim)
        generated = []

        for i in range(num_samples):
            sample = self.decode(z[i])
            generated.append(sample)

        return torch.stack(generated)

    def save(self, path: str) -> None:
        """Сохранение модели."""
        state = {
            'encoder': self.encoder.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'fc_log_var': self.fc_log_var.state_dict(),
            'decoders': [d.state_dict() for d in self.decoders],
            'history': self.history
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Загрузка модели."""
        state = torch.load(path)
        self.encoder.load_state_dict(state['encoder'])
        self.fc_mu.load_state_dict(state['fc_mu'])
        self.fc_log_var.load_state_dict(state['fc_log_var'])
        for decoder, d_state in zip(self.decoders, state['decoders']):
            decoder.load_state_dict(d_state)
        self.history = state['history']

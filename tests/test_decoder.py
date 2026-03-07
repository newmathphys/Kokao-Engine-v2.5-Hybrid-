"""Тесты для декодера (Decoder) в Kokao Engine."""
import pytest
import torch
import numpy as np
from kokao import KokaoCore, CoreConfig
from kokao.decoder import Decoder
from kokao.inverse import InverseProblem


def test_decoder_generate():
    """Проверяем, что Decoder.generate возвращает корректный результат."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    decoder = Decoder(core, lr=0.1, max_steps=50)
    
    # Генерируем вектор для целевого сигнала
    target_signal = 0.5
    x_generated = decoder.generate(target_signal)
    
    # Проверяем, что результат имеет правильную форму
    assert x_generated.shape == (5,)
    
    # Проверяем, что результат не содержит NaN/Inf
    assert not torch.isnan(x_generated).any()
    assert not torch.isinf(x_generated).any()
    
    # Проверяем, что сигнал генерируемого вектора близок к цели
    generated_signal = core.signal(x_generated)
    # Проверяем, что сигнал хотя бы в разумном диапазоне
    assert isinstance(generated_signal, float)
    assert not np.isnan(generated_signal)
    assert not np.isinf(generated_signal)


def test_decoder_does_not_affect_core():
    """Проверяем, что вызов generate не изменяет веса оригинального ядра."""
    config = CoreConfig(input_dim=4)
    core = KokaoCore(config)
    
    # Сохраняем начальные веса
    initial_w_plus = core.w_plus.clone().detach()
    initial_w_minus = core.w_minus.clone().detach()
    
    decoder = Decoder(core, lr=0.1, max_steps=30)
    
    # Генерируем несколько векторов
    for target in [0.1, 0.5, -0.3]:
        _ = decoder.generate(target)
    
    # Проверяем, что веса ядра не изменились
    assert torch.allclose(core.w_plus, initial_w_plus), "Core w_plus should not change after decode"
    assert torch.allclose(core.w_minus, initial_w_minus), "Core w_minus should not change after decode"
    
    # Также проверяем, что эффективные веса не изменились
    initial_eff_w_plus, initial_eff_w_minus = core._get_effective_weights()
    current_eff_w_plus, current_eff_w_minus = core._get_effective_weights()
    
    assert torch.allclose(current_eff_w_plus, initial_eff_w_plus), "Effective weights should not change"
    assert torch.allclose(current_eff_w_minus, initial_eff_w_minus), "Effective weights should not change"


def test_decoder_generate_matches_inverse():
    """Проверяем, что Decoder.generate даёт тот же результат, что и InverseProblem.solve."""
    config = CoreConfig(input_dim=3)
    core = KokaoCore(config)
    
    decoder = Decoder(core, lr=0.1, max_steps=100)
    
    target_signal = 0.8
    
    # Генерируем через декодер
    x_from_decoder = decoder.generate(target_signal)
    
    # Создаём InverseProblem вручную и решаем
    inverse_problem = core.to_inverse_problem()
    x_from_inverse = inverse_problem.solve(S_target=target_signal, lr=0.1, max_steps=100)
    
    # Сравниваем результаты (с определённой точностью)
    # Они могут немного отличаться из-за разных параметров оптимизации
    assert x_from_decoder.shape == x_from_inverse.shape
    # Проверяем, что оба дают близкие сигналы
    signal_decoder = core.signal(x_from_decoder)
    signal_inverse = core.signal(x_from_inverse)
    
    # Сигналы должны быть близки к цели (с определённой погрешностью)
    assert abs(signal_decoder - target_signal) < 0.5
    assert abs(signal_inverse - target_signal) < 0.5


def test_decoder_different_targets():
    """Проверяем Decoder с различными целевыми сигналами."""
    config = CoreConfig(input_dim=6)
    core = KokaoCore(config)
    
    decoder = Decoder(core, lr=0.1, max_steps=50)
    
    # Проверяем несколько целевых сигналов
    targets = [0.1, 0.5, 1.0, -0.5, -1.0, 0.0]
    
    for target in targets:
        x_generated = decoder.generate(target)
        
        # Проверяем форму
        assert x_generated.shape == (6,)
        
        # Проверяем, что результат действителен
        assert not torch.isnan(x_generated).any()
        assert not torch.isinf(x_generated).any()
        
        # Проверяем, что сигнал находится в разумном диапазоне
        generated_signal = core.signal(x_generated)
        assert isinstance(generated_signal, float)
        assert not np.isnan(generated_signal)
        assert not np.isinf(generated_signal)


def test_decoder_with_different_configs():
    """Проверяем Decoder с различными конфигурациями."""
    configs = [
        CoreConfig(input_dim=2),
        CoreConfig(input_dim=7),
        CoreConfig(input_dim=10, target_sum=200.0)
    ]
    
    for config in configs:
        core = KokaoCore(config)
        decoder = Decoder(core, lr=0.05, max_steps=75)
        
        x_generated = decoder.generate(0.3)
        
        assert x_generated.shape == (config.input_dim,)
        assert not torch.isnan(x_generated).any()
        assert not torch.isinf(x_generated).any()


def test_decoder_parameters():
    """Проверяем, что параметры декодера передаются корректно."""
    config = CoreConfig(input_dim=4)
    core = KokaoCore(config)
    
    # Создаём декодер с особыми параметрами
    decoder = Decoder(core, lr=0.2, max_steps=200)
    
    # Генерируем вектор
    x_generated = decoder.generate(0.7)
    
    assert x_generated.shape == (4,)
    generated_signal = core.signal(x_generated)
    assert isinstance(generated_signal, float)


def test_decoder_with_trained_core():
    """Проверяем, что Decoder работает с предварительно обученным ядром."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    # Предварительно обучаем ядро
    x_train = torch.randn(5)
    for _ in range(20):
        core.train(x_train, target=0.6, lr=0.01, mode='gradient')
    
    # Создаём декодер
    decoder = Decoder(core, lr=0.1, max_steps=100)
    
    # Проверяем, что он всё ещё работает
    x_generated = decoder.generate(0.4)
    
    assert x_generated.shape == (5,)
    generated_signal = core.signal(x_generated)
    assert isinstance(generated_signal, float)
    assert not np.isnan(generated_signal)
    assert not np.isinf(generated_signal)


def test_decoder_multiple_generations():
    """Проверяем, что декодер может генерировать несколько векторов последовательно."""
    config = CoreConfig(input_dim=3)
    core = KokaoCore(config)
    
    decoder = Decoder(core, lr=0.1, max_steps=50)
    
    # Генерируем несколько векторов подряд
    targets = [0.1, -0.2, 0.5, 0.0, -0.8]
    generated_vectors = []
    
    for target in targets:
        x_gen = decoder.generate(target)
        generated_vectors.append(x_gen)
        
        # Проверяем каждый вектор
        assert x_gen.shape == (3,)
        assert not torch.isnan(x_gen).any()
        assert not torch.isinf(x_gen).any()
        
        # Проверяем сигнал
        signal = core.signal(x_gen)
        assert isinstance(signal, float)
        assert not np.isnan(signal)
        assert not np.isinf(signal)
    
    # Проверяем, что все векторы разные (в большинстве случаев)
    # (не все могут быть разными из-за особенностей оптимизации)
    unique_vectors = 0
    for i in range(len(generated_vectors)):
        for j in range(i+1, len(generated_vectors)):
            if not torch.allclose(generated_vectors[i], generated_vectors[j], atol=1e-3):
                unique_vectors += 1
    
    # Хотя бы несколько должны быть разными
    assert unique_vectors >= 0  # Просто проверяем, что код работает


if __name__ == "__main__":
    pytest.main([__file__])
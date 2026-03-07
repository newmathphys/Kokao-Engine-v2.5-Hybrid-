"""Тесты для модуля безопасности Kokao Engine."""
import pytest
import torch
import numpy as np
from kokao import KokaoCore, CoreConfig
from kokao.secure import SecureKokao, validate_tensor_input


def test_valid_tensor_passes():
    """Проверяем, что валидный тензор проходит проверку."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    valid_tensor = torch.randn(5)
    result = secure_core.signal(valid_tensor)
    
    # Проверяем, что результат - число
    assert isinstance(result, float)


def test_tensor_with_nan_raises_error():
    """Проверяем, что тензор с NaN вызывает ошибку."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0, 5.0])
    
    with pytest.raises(ValueError, match="contains NaN"):
        secure_core.signal(tensor_with_nan)


def test_tensor_with_inf_raises_error():
    """Проверяем, что тензор с Inf вызывает ошибку."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    tensor_with_inf = torch.tensor([1.0, 2.0, float('inf'), 4.0, 5.0])
    
    with pytest.raises(ValueError, match="contains Inf"):
        secure_core.signal(tensor_with_inf)


def test_tensor_wrong_dimension_raises_error():
    """Проверяем, что тензор неверной размерности вызывает ошибку."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    wrong_dim_tensor = torch.randn(3)  # Должно быть 5
    
    with pytest.raises(ValueError, match="Expected input dimension"):
        secure_core.signal(wrong_dim_tensor)


def test_non_tensor_input_raises_error():
    """Проверяем, что не-тензорный ввод вызывает ошибку."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    non_tensor_input = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    with pytest.raises(ValueError, match="Expected torch.Tensor"):
        secure_core.signal(non_tensor_input)


def test_secure_kokao_proxies_methods_correctly():
    """Проверяем, что SecureKokao правильно проксирует методы исходного ядра."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    # Проверяем, что атрибуты доступны
    assert secure_core.config.input_dim == 5
    assert secure_core.w.shape[0] == 5
    
    # Проверяем, что методы работают
    x = torch.randn(5)
    signal_result = secure_core.signal(x)
    forward_result = secure_core.forward(x)
    
    assert isinstance(signal_result, float)
    assert isinstance(forward_result, torch.Tensor)
    assert forward_result.shape == torch.Size([])  # Скаляр для одиночного ввода


def test_secure_train_method():
    """Проверяем метод train в безопасной обертке."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    x = torch.randn(5)
    target = 0.5
    
    loss = secure_core.train(x, target)
    
    assert isinstance(loss, float)
    assert loss >= 0


def test_secure_forward_method():
    """Проверяем метод forward в безопасной обертке."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    x = torch.randn(5)
    
    result = secure_core.forward(x)
    
    assert isinstance(result, torch.Tensor)


def test_secure_forget_method():
    """Проверяем метод forget в безопасной обертке (не требует тензоров)."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    # Метод forget не принимает тензоры, поэтому не нуждается в проверке
    secure_core.forget(rate=0.1)
    
    # Просто проверяем, что метод вызывается без ошибок
    assert True


def test_batch_tensor_validation():
    """Проверяем валидацию батчевых тензоров."""
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    secure_core = SecureKokao(core)
    
    # Валидный батч
    valid_batch = torch.randn(3, 5)
    
    # forward может обрабатывать батчи, но signal - нет
    batch_forward_result = secure_core.forward(valid_batch)
    assert batch_forward_result.shape == torch.Size([3])  # Выход для каждого элемента в батче


if __name__ == "__main__":
    pytest.main([__file__])
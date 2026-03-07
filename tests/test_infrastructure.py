"""
Уровень 0: Базовые проверки инфраструктуры.

Проверка импорта всех модулей, версий, конфигурации.
"""
import pytest
import sys
import torch
import kokao
from kokao import CoreConfig, set_debug, DEBUG


# =============================================================================
# ТЕСТЫ ИМПОРТА
# =============================================================================

class TestImports:
    """Проверка импорта всех модулей."""

    def test_import_kokao(self):
        """Базовый импорт kokao."""
        import kokao
        assert hasattr(kokao, '__version__')

    def test_import_core(self):
        """Импорт ядра."""
        from kokao import KokaoCore, CoreConfig
        assert KokaoCore is not None
        assert CoreConfig is not None

    def test_import_inverse(self):
        """Импорт обратной задачи."""
        from kokao import InverseProblem, Decoder
        assert InverseProblem is not None
        assert Decoder is not None

    def test_import_etalon(self):
        """Импорт эталонных систем."""
        from kokao import IntuitiveEtalonSystem, NormalIntuitiveEtalonSystem
        assert IntuitiveEtalonSystem is not None
        assert NormalIntuitiveEtalonSystem is not None

    def test_import_goal_system(self):
        """Импорт системы целей."""
        from kokao import SelfPlanningSystem
        assert SelfPlanningSystem is not None

    def test_import_math_exact(self):
        """Импорт точных математических методов."""
        from kokao import MathExactCore, MathExactConfig, InversionMethod
        assert MathExactCore is not None
        assert MathExactConfig is not None
        assert InversionMethod is not None

    def test_import_security(self):
        """Импорт модулей безопасности."""
        from kokao import SecureKokao, validate_tensor_input
        assert SecureKokao is not None

    def test_import_integrations(self):
        """Импорт интеграций."""
        from kokao import RAGModule, XAIAnalyzer
        assert RAGModule is not None
        assert XAIAnalyzer is not None

    def test_import_all_public_api(self):
        """Проверка что все элементы __all__ импортируются."""
        for name in kokao.__all__:
            assert hasattr(kokao, name), f"Отсутствует в API: {name}"


# =============================================================================
# ТЕСТЫ ВЕРСИЙ И ЗАВИСИМОСТЕЙ
# =============================================================================

class TestVersions:
    """Проверка версий Python и PyTorch."""

    def test_python_version(self):
        """Проверка версии Python (требуется >= 3.9)."""
        assert sys.version_info >= (3, 9), "Требуется Python 3.9+"

    def test_pytorch_version(self):
        """Проверка версии PyTorch (требуется >= 2.1)."""
        from packaging import version
        assert version.parse(torch.__version__) >= version.parse("2.1"), \
            "Требуется PyTorch 2.1+"

    def test_kokao_version(self):
        """Проверка версии Kokao Engine."""
        # Версия должна начинаться с "3." для v3.0
        assert kokao.__version__.startswith("3."), f"Версия должна быть 3.x, текущая: {kokao.__version__}"

    def test_cuda_available(self):
        """Проверка доступности CUDA (информационный тест)."""
        cuda_available = torch.cuda.is_available()
        # Тест всегда проходит, просто логируем
        assert True  # CUDA может быть недоступен, это нормально


# =============================================================================
# ТЕСТЫ КОНФИГУРАЦИИ
# =============================================================================

class TestCoreConfig:
    """Проверка CoreConfig с валидными/невалидными данными."""

    def test_core_config_valid(self):
        """Валидная конфигурация."""
        config = CoreConfig(input_dim=10)
        assert config.input_dim == 10
        assert config.device == "cpu"
        assert config.dtype == "float32"
        assert config.target_sum == 100.0

    def test_core_config_custom(self):
        """Конфигурация с пользовательскими параметрами."""
        config = CoreConfig(
            input_dim=50,
            device="cpu",
            dtype="float64",
            target_sum=50.0,
            seed=42
        )
        assert config.input_dim == 50
        assert config.dtype == "float64"
        assert config.target_sum == 50.0
        assert config.seed == 42

    def test_core_config_invalid_input_dim(self):
        """Невалидная размерность входа (<= 0)."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CoreConfig(input_dim=0)

    def test_core_config_invalid_device(self):
        """Невалидное устройство."""
        with pytest.raises(Exception):
            CoreConfig(input_dim=10, device="invalid_device")

    def test_core_config_invalid_dtype(self):
        """Невалидный тип данных."""
        with pytest.raises(Exception):
            CoreConfig(input_dim=10, dtype="invalid_dtype")

    def test_core_config_invalid_target_sum(self):
        """Невалидная целевая сумма (<= 0)."""
        with pytest.raises(Exception):
            CoreConfig(input_dim=10, target_sum=-100.0)


# =============================================================================
# ТЕСТЫ ГЛОБАЛЬНОГО ФЛАГА DEBUG
# =============================================================================

class TestDebugFlag:
    """Проверка работы глобального флага DEBUG и set_debug."""

    def test_debug_initial_value(self):
        """Начальное значение DEBUG = False."""
        # Импортируем заново для получения актуального значения
        import importlib
        import kokao
        importlib.reload(kokao)
        assert kokao.DEBUG == False

    def test_set_debug_true(self):
        """Установка DEBUG = True."""
        import importlib
        import kokao
        importlib.reload(kokao)
        
        kokao.set_debug(True)
        # Проверяем через модуль
        assert kokao.DEBUG == True
        # Возвращаем обратно
        kokao.set_debug(False)

    def test_set_debug_false(self):
        """Установка DEBUG = False."""
        import importlib
        import kokao
        importlib.reload(kokao)
        
        kokao.set_debug(True)  # Сначала включаем
        kokao.set_debug(False)
        assert kokao.DEBUG == False

    def test_set_debug_toggle(self):
        """Переключение DEBUG."""
        import importlib
        import kokao
        importlib.reload(kokao)
        
        initial = kokao.DEBUG
        kokao.set_debug(not initial)
        assert kokao.DEBUG == (not initial)
        # Возвращаем обратно
        kokao.set_debug(initial)


# =============================================================================
# ТЕСТЫ ЗАВИСИМОСТЕЙ
# =============================================================================

class TestDependencies:
    """Проверка наличия обязательных зависимостей."""

    def test_torch_installed(self):
        """PyTorch установлен."""
        import torch
        assert torch is not None

    def test_pytorch_has_cuda(self):
        """PyTorch поддерживает CUDA (информационный тест)."""
        assert hasattr(torch, 'cuda')

    def test_pytorch_has_autograd(self):
        """PyTorch поддерживает autograd."""
        x = torch.tensor([1.0], requires_grad=True)
        y = x ** 2
        y.backward()
        assert x.grad is not None

    def test_pydantic_installed(self):
        """Pydantic установлен."""
        from pydantic import BaseModel
        assert BaseModel is not None

    def test_typing_extensions(self):
        """typing_extensions доступен."""
        from typing_extensions import Literal
        assert Literal is not None


# =============================================================================
# ТЕСТЫ УТИЛИТ
# =============================================================================

class TestUtilities:
    """Проверка вспомогательных функций."""

    def test_tensor_creation(self):
        """Создание тензоров работает."""
        x = torch.randn(10)
        assert x.shape == (10,)
        assert x.dtype == torch.float32

    def test_tensor_device(self):
        """Устройства тензоров."""
        x_cpu = torch.tensor([1.0], device="cpu")
        assert x_cpu.device.type == "cpu"

    def test_tensor_dtype(self):
        """Типы данных тензоров."""
        x_float32 = torch.tensor([1.0], dtype=torch.float32)
        x_float64 = torch.tensor([1.0], dtype=torch.float64)
        assert x_float32.dtype == torch.float32
        assert x_float64.dtype == torch.float64

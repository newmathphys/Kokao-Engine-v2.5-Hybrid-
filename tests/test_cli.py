"""Тесты для CLI интерфейса Kokao Engine."""
import pytest
import tempfile
import json
import torch
from pathlib import Path
from typer.testing import CliRunner
from kokao.cli import app


def test_help_command():
    """Проверяем, что команда --help работает."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_train_command():
    """Тестируем команду train с тестовым файлом."""
    # Создаем временный файл с тестовыми данными
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"x": [0.1, 0.2, 0.3, 0.4, 0.5]}, f)
        data_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "train",
            "--data", data_path,
            "--target", "0.8",
            "--output", output_path,
            "--dim", "5",
            "--epochs", "1"
        ])
        
        # Проверяем, что команда завершилась успешно
        assert result.exit_code == 0
        assert "Модель сохранена" in result.output
        
        # Проверяем, что файл модели создан
        assert Path(output_path).exists()
        
    finally:
        # Удаляем временные файлы
        Path(data_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_invert_command():
    """Тестируем команду invert с тестовой моделью."""
    # Сначала создаем тестовую модель
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Создаем простую тестовую модель
        import torch
        from kokao import KokaoCore, CoreConfig
        
        config = CoreConfig(input_dim=5)
        core = KokaoCore(config)
        model_path = f.name
        core.save(model_path)
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "invert",
            "--target", "0.5",
            "--model", model_path,
            "--output", output_path,
            "--dim", "5"
        ])
        
        # Проверяем, что команда завершилась успешно
        assert result.exit_code == 0
        assert "Сгенерирован вектор" in result.output
        assert "Вектор сохранен" in result.output
        
        # Проверяем, что файл вектора создан
        assert Path(output_path).exists()
        
    finally:
        # Удаляем временные файлы
        Path(model_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


def test_signal_command():
    """Тестируем команду signal с вектором."""
    # Создаем тестовую модель
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        from kokao import KokaoCore, CoreConfig
        
        config = CoreConfig(input_dim=3)
        core = KokaoCore(config)
        model_path = f.name
        core.save(model_path)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "signal",
            "--vector", "[0.1, 0.2, 0.3]",
            "--model", model_path,
            "--dim", "3"
        ])

        # Проверяем, что команда завершилась успешно
        assert result.exit_code == 0
        assert "Сигнал для вектора" in result.output
        # Проверяем наличие сигнала (число с плавающей точкой)
        assert "0.99" in result.output or "1." in result.output or "0." in result.output

    finally:
        # Удаляем временный файл
        Path(model_path).unlink(missing_ok=True)


def test_info_command():
    """Тестируем команду info."""
    # Создаем тестовую модель
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        from kokao import KokaoCore, CoreConfig
        
        config = CoreConfig(input_dim=4)
        core = KokaoCore(config)
        model_path = f.name
        core.save(model_path)
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "info",
            "--model", model_path
        ])
        
        # Проверяем, что команда завершилась успешно
        assert result.exit_code == 0
        assert "Модель загружена" in result.output
        assert "Размерность:" in result.output
        
    finally:
        # Удаляем временный файл
        Path(model_path).unlink(missing_ok=True)


def test_invalid_json_file():
    """Тестируем реакцию на невалидный JSON файл."""
    # Создаем файл с невалидным JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{invalid json")
        data_path = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        output_path = f.name
    
    try:
        runner = CliRunner()
        result = runner.invoke(app, [
            "train",
            "--data", data_path,
            "--target", "0.8",
            "--output", output_path,
            "--dim", "5"
        ])
        
        # Команда должна завершиться с ошибкой
        assert result.exit_code != 0
        
    finally:
        # Удаляем временные файлы
        Path(data_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
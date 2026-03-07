"""Интерфейс командной строки для Kokao Engine."""
import typer
import json
import torch
from pathlib import Path
from typing import Optional
import numpy as np

from . import KokaoCore, CoreConfig, Decoder

app = typer.Typer()


@app.command()
def train(
    data_path: str = typer.Option(..., "--data", help="Путь к файлу с обучающими данными (JSON)"),
    target: float = typer.Option(..., "--target", help="Целевое значение сигнала"),
    output_path: str = typer.Option("model.json", "--output", help="Путь для сохранения обученной модели"),
    epochs: int = typer.Option(1, "--epochs", help="Количество эпох обучения"),
    lr: float = typer.Option(0.01, "--lr", help="Скорость обучения"),
    mode: str = typer.Option("gradient", "--mode", help="Режим обучения (gradient/kosyakov)"),
    input_dim: int = typer.Option(10, "--dim", help="Размерность входного вектора")
):
    """
    Обучение модели KokaoCore на указанных данных.
    """
    typer.echo(f"Загрузка данных из {data_path}")
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Если данные - это список чисел, преобразуем в тензор
        if isinstance(data, list):
            x_values = torch.tensor(data, dtype=torch.float32)
            if x_values.numel() != input_dim:
                typer.echo(f"Ошибка: размерность данных ({x_values.numel()}) не совпадает с --dim ({input_dim})")
                raise typer.Exit(code=1)
            x = x_values
        else:
            # Если данные - это словарь с ключом 'x'
            x_list = data.get('x', [])
            if len(x_list) != input_dim:
                typer.echo(f"Ошибка: размерность данных ({len(x_list)}) не совпадает с --dim ({input_dim})")
                raise typer.Exit(code=1)
            x = torch.tensor(x_list, dtype=torch.float32)
    except FileNotFoundError:
        typer.echo(f"Файл {data_path} не найден")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        typer.echo(f"Файл {data_path} не содержит корректный JSON")
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"Ошибка при загрузке данных: {str(e)}")
        raise typer.Exit(code=1)
    
    # Создаем конфиг и ядро
    config = CoreConfig(input_dim=input_dim)
    core = KokaoCore(config)
    
    typer.echo(f"Обучение ядра на векторе {x.tolist()} с целью {target}")
    
    # Обучаем
    for epoch in range(epochs):
        loss = core.train(x, target, lr=lr, mode=mode)
        typer.echo(f"Эпоха {epoch + 1}/{epochs}, потеря: {loss:.6f}")
    
    # Сохраняем модель
    core.save(output_path)
    typer.echo(f"Модель сохранена в {output_path}")


@app.command()
def invert(
    target: float = typer.Option(..., "--target", help="Целевой сигнал для генерации вектора"),
    model_path: str = typer.Option("model.json", "--model", help="Путь к сохраненной модели"),
    output_path: str = typer.Option("generated_vector.json", "--output", help="Путь для сохранения сгенерированного вектора"),
    input_dim: int = typer.Option(10, "--dim", help="Размерность вектора")
):
    """
    Генерация вектора с заданным сигналом (обратная задача).
    """
    # Загружаем модель
    try:
        core = KokaoCore.load(model_path)
        typer.echo(f"Модель загружена из {model_path}")
    except FileNotFoundError:
        typer.echo(f"Модель {model_path} не найдена, создаем новую")
        config = CoreConfig(input_dim=input_dim)
        core = KokaoCore(config)
    
    # Создаем декодер и генерируем вектор
    decoder = Decoder(core)
    generated_vector = decoder.generate(target)
    
    typer.echo(f"Сгенерирован вектор: {generated_vector.tolist()}")
    typer.echo(f"Сигнал сгенерированного вектора: {core.signal(generated_vector):.6f}")
    
    # Сохраняем вектор
    with open(output_path, 'w') as f:
        json.dump({
            "vector": generated_vector.tolist(),
            "signal": core.signal(generated_vector),
            "target": target
        }, f, indent=2)
    
    typer.echo(f"Вектор сохранен в {output_path}")


@app.command()
def signal(
    vector_str: str = typer.Option(..., "--vector", help="Входной вектор (в формате JSON или через запятую)"),
    model_path: str = typer.Option("model.json", "--model", help="Путь к сохраненной модели"),
    input_dim: int = typer.Option(10, "--dim", help="Размерность вектора")
):
    """
    Вычисление сигнала для заданного вектора.
    """
    # Загружаем модель
    try:
        core = KokaoCore.load(model_path)
        typer.echo(f"Модель загружена из {model_path}")
    except FileNotFoundError:
        typer.echo(f"Модель {model_path} не найдена, создаем новую")
        config = CoreConfig(input_dim=input_dim)
        core = KokaoCore(config)
    
    # Парсим вектор
    try:
        # Пробуем как JSON
        vector_data = json.loads(vector_str)
        if isinstance(vector_data, list):
            vector = torch.tensor(vector_data, dtype=torch.float32)
        else:
            raise ValueError("Вектор должен быть массивом чисел")
    except json.JSONDecodeError:
        # Если не JSON, пробуем CSV
        try:
            vector_list = [float(x.strip()) for x in vector_str.split(',')]
            vector = torch.tensor(vector_list, dtype=torch.float32)
        except ValueError:
            typer.echo("Ошибка: вектор должен быть в формате JSON или CSV")
            raise typer.Exit(code=1)
    
    # Проверяем размерность
    if vector.shape[0] != input_dim:
        typer.echo(f"Ошибка: размерность вектора ({vector.shape[0]}) не совпадает с --dim ({input_dim})")
        raise typer.Exit(code=1)
    
    # Вычисляем сигнал
    signal_value = core.signal(vector)
    typer.echo(f"Сигнал для вектора {vector.tolist()}: {signal_value:.6f}")


@app.command()
def info(
    model_path: str = typer.Option("model.json", "--model", help="Путь к сохраненной модели")
):
    """
    Вывод информации о модели.
    """
    try:
        core = KokaoCore.load(model_path)
        typer.echo(f"Модель загружена из {model_path}")
        typer.echo(f"Размерность: {core.config.input_dim}")
        typer.echo(f"Устройство: {core.device}")
        typer.echo(f"Версия: {core.version}")
        typer.echo(f"Сумма w_plus: {core.w_plus.sum():.6f}")
        typer.echo(f"Сумма w_minus: {core.w_minus.sum():.6f}")
    except FileNotFoundError:
        typer.echo(f"Модель {model_path} не найдена")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
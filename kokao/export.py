"""Модуль экспорта моделей в различные форматы (ONNX, TensorRT, etc.)."""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import logging
from pathlib import Path
import json

from .core import KokaoCore
from .core_base import CoreConfig

logger = logging.getLogger(__name__)


class KokaoExportWrapper(nn.Module):
    """
    Обертка для экспорта KokaoCore.
    """

    def __init__(self, core: KokaoCore):
        """
        Инициализация обертки.

        Args:
            core: Ядро для экспорта
        """
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход."""
        # Обработка 1D и 2D входов
        if x.ndim == 1:
            return self.core.forward(x).unsqueeze(0)
        else:
            # Для батча обрабатываем каждый элемент
            outputs = []
            for i in range(x.shape[0]):
                out = self.core.forward(x[i])
                outputs.append(out)
            return torch.stack(outputs)


class ModelExporter:
    """
    Экспортер моделей KokaoCore в различные форматы.
    """

    def __init__(self, core: KokaoCore):
        """
        Инициализация экспортера.

        Args:
            core: Модель для экспорта
        """
        self.core = core
        self.wrapper = KokaoExportWrapper(core)
        self.export_history: List[Dict[str, Any]] = []

    def export_to_onnx(self, output_path: str, 
                       input_shape: Tuple[int, ...] = (1, 10),
                       opset_version: int = 14,
                       dynamic_axes: Optional[Dict[str, Any]] = None,
                       verbose: bool = False) -> str:
        """
        Экспорт в формат ONNX.

        Args:
            output_path: Путь для сохранения
            input_shape: Форма входа
            opset_version: Версия ONNX opset
            dynamic_axes: Динамические оси
            verbose: Подробный вывод

        Returns:
            Путь к сохраненному файлу
        """
        self.wrapper.eval()

        # Пример входа
        dummy_input = torch.randn(*input_shape)

        # Настройка dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Экспорт
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )

        self.export_history.append({
            'format': 'onnx',
            'path': output_path,
            'input_shape': input_shape,
            'opset_version': opset_version
        })

        logger.info(f"Exported to ONNX: {output_path}")
        return output_path

    def export_to_torchscript(self, output_path: str,
                               method: str = 'trace') -> str:
        """
        Экспорт в TorchScript.

        Args:
            output_path: Путь для сохранения
            method: Метод ('trace' или 'script')

        Returns:
            Путь к сохраненному файлу
        """
        self.wrapper.eval()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if method == 'trace':
            # Трейсинг
            dummy_input = torch.randn(1, self.core.config.input_dim)
            scripted = torch.jit.trace(self.wrapper, dummy_input)
        else:
            # Скриптование
            scripted = torch.jit.script(self.wrapper)

        scripted.save(output_path)

        self.export_history.append({
            'format': 'torchscript',
            'path': output_path,
            'method': method
        })

        logger.info(f"Exported to TorchScript: {output_path}")
        return output_path

    def export_to_tensorrt(self, output_path: str,
                           input_shape: Tuple[int, ...] = (1, 10),
                           precision: str = 'fp32',
                           max_batch_size: int = 32,
                           workspace_size: int = 1 << 30) -> Optional[str]:
        """
        Экспорт в TensorRT (требует torch2trt).

        Args:
            output_path: Путь для сохранения
            input_shape: Форма входа
            precision: Точность ('fp32', 'fp16', 'int8')
            max_batch_size: Максимальный размер батча
            workspace_size: Размер рабочего пространства

        Returns:
            Путь к сохраненному файлу или None если не удалось
        """
        try:
            from torch2trt import torch2trt
        except ImportError:
            logger.warning("torch2trt not installed. Install with: pip install torch2trt")
            return None

        self.wrapper.eval()

        # Пример входа
        dummy_input = torch.randn(*input_shape).to(self.core.device)

        # Конвертация
        trt_model = torch2trt(
            self.wrapper,
            [dummy_input],
            fp16_mode=(precision == 'fp16'),
            int8_mode=(precision == 'int8'),
            max_batch_size=max_batch_size,
            max_workspace_size=workspace_size
        )

        # Сохранение
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(trt_model.state_dict(), output_path)

        self.export_history.append({
            'format': 'tensorrt',
            'path': output_path,
            'precision': precision
        })

        logger.info(f"Exported to TensorRT: {output_path}")
        return output_path

    def export_to_coreml(self, output_path: str,
                         input_name: str = 'input',
                         output_name: str = 'output') -> Optional[str]:
        """
        Экспорт в CoreML (для macOS/iOS).

        Args:
            output_path: Путь для сохранения
            input_name: Имя входа
            output_name: Имя выхода

        Returns:
            Путь к сохраненному файлу или None если не удалось
        """
        try:
            import coremltools as ct
        except ImportError:
            logger.warning("coremltools not installed. Install with: pip install coremltools")
            return None

        self.wrapper.eval()

        # Пример входа
        dummy_input = torch.randn(1, self.core.config.input_dim)

        # Трейсинг
        traced = torch.jit.trace(self.wrapper, dummy_input)

        # Конвертация в CoreML
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name=input_name, shape=dummy_input.shape)],
            outputs=[ct.TensorType(name=output_name)]
        )

        # Сохранение
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(output_path)

        self.export_history.append({
            'format': 'coreml',
            'path': output_path
        })

        logger.info(f"Exported to CoreML: {output_path}")
        return output_path

    def export_to_tflite(self, output_path: str,
                         input_shape: Tuple[int, ...] = (1, 10)) -> Optional[str]:
        """
        Экспорт в TensorFlow Lite.

        Args:
            output_path: Путь для сохранения
            input_shape: Форма входа

        Returns:
            Путь к сохраненному файлу или None если не удалось
        """
        try:
            import tensorflow as tf
        except ImportError:
            logger.warning("tensorflow not installed. Install with: pip install tensorflow")
            return None

        self.wrapper.eval()

        # Сначала экспортируем в ONNX
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
            onnx_path = tmp.name

        self.export_to_onnx(onnx_path, input_shape)

        # Конвертация ONNX -> TensorFlow -> TFLite
        try:
            from onnx_tf.backend import prepare
            import onnx
        except ImportError:
            logger.warning("onnx-tensorflow not installed")
            Path(onnx_path).unlink()
            return None

        # Загрузка ONNX
        onnx_model = onnx.load(onnx_path)

        # Конвертация в TensorFlow
        tf_rep = prepare(onnx_model)

        # Сохранение в формате SavedModel
        tf_path = Path(output_path).parent / 'tf_model'
        tf_rep.export_graph(str(tf_path))

        # Конвертация в TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_path))
        tflite_model = converter.convert()

        # Сохранение TFLite
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Очистка
        Path(onnx_path).unlink()
        import shutil
        shutil.rmtree(tf_path)

        self.export_history.append({
            'format': 'tflite',
            'path': output_path
        })

        logger.info(f"Exported to TFLite: {output_path}")
        return output_path

    def get_export_metadata(self) -> Dict[str, Any]:
        """
        Получение метаданных экспорта.

        Returns:
            Метаданные
        """
        return {
            'model_info': {
                'input_dim': self.core.config.input_dim,
                'device': str(self.core.device),
                'version': self.core.version
            },
            'export_history': self.export_history,
            'supported_formats': [
                'onnx', 'torchscript', 'tensorrt', 'coreml', 'tflite'
            ]
        }

    def verify_export(self, exported_path: str, 
                      format: str,
                      test_input: Optional[torch.Tensor] = None,
                      tolerance: float = 1e-4) -> Dict[str, Any]:
        """
        Проверка корректности экспорта.

        Args:
            exported_path: Путь к экспортированному файлу
            format: Формат файла
            test_input: Тестовый вход
            tolerance: Допуск для сравнения

        Returns:
            Результаты проверки
        """
        if test_input is None:
            test_input = torch.randn(1, self.core.config.input_dim)

        # Оригинальный вывод
        self.wrapper.eval()
        with torch.no_grad():
            original_output = self.wrapper(test_input)

        results = {
            'format': format,
            'path': exported_path,
            'original_output': original_output.tolist(),
            'exported_output': None,
            'error': None,
            'passed': False
        }

        try:
            if format == 'onnx':
                import onnxruntime as ort
                session = ort.InferenceSession(exported_path)
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name

                exported_output = session.run(
                    [output_name],
                    {input_name: test_input.numpy()}
                )[0]

            elif format == 'torchscript':
                scripted = torch.jit.load(exported_path)
                scripted.eval()
                with torch.no_grad():
                    exported_output = scripted(test_input).numpy()

            else:
                logger.warning(f"Verification not supported for format: {format}")
                results['error'] = 'Format not supported for verification'
                return results

            results['exported_output'] = exported_output.tolist()

            # Сравнение
            error = np.mean(np.abs(original_output.numpy() - exported_output))
            results['error'] = error
            results['passed'] = error < tolerance

        except Exception as e:
            results['error'] = str(e)

        return results


def export_model(core: KokaoCore, output_dir: str,
                 formats: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Быстрый экспорт модели в несколько форматов.

    Args:
        core: Модель для экспорта
        output_dir: Директория для экспорта
        formats: Список форматов (по умолчанию ['onnx', 'torchscript'])

    Returns:
        Словарь {формат: путь}
    """
    if formats is None:
        formats = ['onnx', 'torchscript']

    exporter = ModelExporter(core)
    exported_paths = {}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output_path = Path(output_dir) / f"model.{fmt}"

        if fmt == 'onnx':
            path = exporter.export_to_onnx(str(output_path))
        elif fmt == 'torchscript':
            path = exporter.export_to_torchscript(str(output_path))
        elif fmt == 'tensorrt':
            path = exporter.export_to_tensorrt(str(output_path))
        elif fmt == 'coreml':
            path = exporter.export_to_coreml(str(output_path))
        elif fmt == 'tflite':
            path = exporter.export_to_tflite(str(output_path))
        else:
            logger.warning(f"Unknown format: {fmt}")
            continue

        exported_paths[fmt] = path

    return exported_paths


def load_exported_model(path: str, format: str,
                        input_dim: int) -> Any:
    """
    Загрузка экспортированной модели.

    Args:
        path: Путь к файлу
        format: Формат файла
        input_dim: Размерность входа

    Returns:
        Загруженная модель
    """
    if format == 'onnx':
        import onnxruntime as ort
        return ort.InferenceSession(path)

    elif format == 'torchscript':
        return torch.jit.load(path)

    elif format == 'tensorrt':
        from torch2trt import TRTModule
        trt_model = TRTModule()
        trt_model.load_state_dict(torch.load(path))
        return trt_model

    else:
        raise ValueError(f"Cannot load format: {format}")

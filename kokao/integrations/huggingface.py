"""Интеграция Kokao Engine с HuggingFace Hub."""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import torch

try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from ..core import KokaoCore
from ..core_base import CoreConfig

logger = logging.getLogger(__name__)


class HFModelManager:
    """Менеджер для загрузки и выгрузки моделей KokaoCore в/из HuggingFace Hub."""
    
    def __init__(self):
        """Инициализация менеджера моделей HuggingFace."""
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub не установлен. Установите с помощью: pip install huggingface_hub"
            )
        
        self.api = HfApi()
    
    def push_model(self, core: KokaoCore, repo_id: str, filename: str = "kokao_model.json", private: bool = False) -> str:
        """
        Загрузка модели в репозиторий HuggingFace Hub.
        
        Args:
            core: Экземпляр KokaoCore для сохранения
            repo_id: ID репозитория в формате "username/model_name"
            filename: Имя файла модели
            private: Приватный ли репозиторий
            
        Returns:
            URL загруженной модели
        """
        # Создаем временный файл для сохранения модели
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            # Сохраняем состояние ядра
            state = {
                "w_plus": core.w_plus.cpu().tolist(),
                "w_minus": core.w_minus.cpu().tolist(),
                "config": core.config.model_dump(),
                "version": core.version,
                "architecture": "KokaoCore-v2.0.0",
                "description": "KokaoCore model trained with two-channel architecture (S = S+/S-)"
            }
            
            json.dump(state, tmp, indent=2)
            temp_path = tmp.name
        
        try:
            # Загружаем файл в репозиторий
            url = self.api.upload_file(
                path_or_fileobj=temp_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                create_pr=False
            )
            
            logger.info(f"Model uploaded to {url}")
            return url
            
        finally:
            # Удаляем временный файл
            Path(temp_path).unlink(missing_ok=True)
    
    def pull_model(self, repo_id: str, filename: str = "kokao_model.json", device: Optional[str] = None) -> KokaoCore:
        """
        Загрузка модели из репозитория HuggingFace Hub.
        
        Args:
            repo_id: ID репозитория в формате "username/model_name"
            filename: Имя файла модели
            device: Устройство для загрузки модели (cpu/cuda)
            
        Returns:
            Экземпляр KokaoCore
        """
        # Скачиваем файл из репозитория
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )
        
        # Загружаем состояние из файла
        with open(downloaded_file, 'r') as f:
            state = json.load(f)
        
        # Воссоздаем конфигурацию
        config_data = state["config"]
        config = CoreConfig(**config_data)
        
        # Если указано устройство, обновляем конфиг
        if device:
            config.device = device
        
        # Создаем ядро с данной конфигурацией
        core = KokaoCore(config)
        
        # Загружаем веса
        w_plus_data = torch.tensor(state["w_plus"], device=config.device, dtype=core.dtype)
        w_minus_data = torch.tensor(state["w_minus"], device=config.device, dtype=core.dtype)
        core.w_plus = torch.nn.Parameter(w_plus_data, requires_grad=True)
        core.w_minus = torch.nn.Parameter(w_minus_data, requires_grad=True)

        # Загружаем дополнительные параметры
        core.version = state.get("version", 0)
        core.is_quantized = state.get("is_quantized", False)
        
        # Пересоздаем оптимизатор
        core.optimizer = torch.optim.Adam([core.w_plus, core.w_minus], lr=0.01)

        logger.info(f"Model loaded from {repo_id}/{filename}")
        return core
    
    def create_repo(self, repo_id: str, private: bool = False, repo_type: str = "model") -> str:
        """
        Создание нового репозитория на HuggingFace Hub.
        
        Args:
            repo_id: ID репозитория
            private: Приватный ли репозиторий
            repo_type: Тип репозитория
            
        Returns:
            URL созданного репозитория
        """
        url = self.api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type=repo_type
        )
        
        logger.info(f"Repository created at {url}")
        return url
    
    def model_info(self, repo_id: str, filename: str = "kokao_model.json") -> Dict[str, Any]:
        """
        Получение информации о модели в репозитории.
        
        Args:
            repo_id: ID репозитория
            filename: Имя файла модели
            
        Returns:
            Словарь с информацией о модели
        """
        # Скачиваем файл с информацией
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model"
        )
        
        # Читаем и возвращаем содержимое
        with open(downloaded_file, 'r') as f:
            state = json.load(f)
        
        return state
    
    def list_models_by_author(self, author: str) -> list:
        """
        Получение списка моделей конкретного автора.
        
        Args:
            author: Имя автора на HuggingFace
            
        Returns:
            Список репозиториев моделей
        """
        models = self.api.list_models(author=author, cardData=True)
        kokao_models = []
        
        for model in models:
            # Проверяем, содержит ли модель KokaoCore
            if hasattr(model, 'cardData') and model.cardData:
                tags = model.cardData.get('tags', [])
                if 'kokao' in [tag.lower() for tag in tags] or 'kokaoengine' in [tag.lower() for tag in tags]:
                    kokao_models.append({
                        'id': model.modelId,
                        'tags': tags,
                        'description': model.cardData.get('model_description', ''),
                        'url': f"https://huggingface.co/{model.modelId}"
                    })
        
        return kokao_models
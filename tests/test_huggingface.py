"""Тесты для интеграции с HuggingFace Hub."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from kokao import KokaoCore, CoreConfig
from kokao.integrations.huggingface import HFModelManager


def test_hf_manager_initialization():
    """Проверяем инициализацию HFModelManager."""
    # Проверяем, что при отсутствии huggingface_hub выбрасывается ошибка
    with patch('kokao.integrations.huggingface.HF_AVAILABLE', False):
        with pytest.raises(ImportError, match="huggingface_hub не установлен"):
            HFModelManager()


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
@patch('kokao.integrations.huggingface.HfApi')
def test_push_model(mock_api_class):
    """Проверяем загрузку модели."""
    # Подготовим mock API
    mock_api = Mock()
    mock_api_class.return_value = mock_api
    mock_api.upload_file.return_value = "https://huggingface.co/test/repo/resolve/main/model.json"
    
    manager = HFModelManager()
    
    # Создаем тестовую модель
    config = CoreConfig(input_dim=5)
    core = KokaoCore(config)
    
    # Выполняем загрузку
    repo_id = "test/repo"
    result_url = manager.push_model(core, repo_id)
    
    # Проверяем, что upload_file был вызван
    assert mock_api.upload_file.called
    assert result_url == "https://huggingface.co/test/repo/resolve/main/model.json"


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
@patch('kokao.integrations.huggingface.hf_hub_download')
def test_pull_model(mock_hf_download):
    """Проверяем скачивание модели."""
    # Создаем временный файл с тестовыми данными модели
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        model_data = {
            "w_plus": [0.1, 0.2, 0.3, 0.4, 0.5],
            "w_minus": [0.05, 0.15, 0.25, 0.35, 0.45],
            "config": {
                "input_dim": 5,
                "device": "cpu",
                "dtype": "float32",
                "target_sum": 100.0,
                "max_history": 100,
                "use_log_domain": False
            },
            "version": 1
        }
        json.dump(model_data, f)
        temp_path = f.name
    
    # Мокаем hf_hub_download, чтобы он возвращал наш временный файл
    mock_hf_download.return_value = temp_path
    
    manager = HFModelManager()
    
    # Загружаем модель
    repo_id = "test/repo"
    core = manager.pull_model(repo_id)
    
    # Проверяем, что модель создана с правильной конфигурацией
    assert core.config.input_dim == 5
    assert core.w_plus.shape[0] == 5
    assert core.w_minus.shape[0] == 5
    
    # Удаляем временный файл
    Path(temp_path).unlink()


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
@patch('kokao.integrations.huggingface.HfApi')
def test_create_repo(mock_api_class):
    """Проверяем создание репозитория."""
    mock_api = Mock()
    mock_api_class.return_value = mock_api
    mock_api.create_repo.return_value = "https://huggingface.co/test/new-repo"
    
    manager = HFModelManager()
    
    repo_id = "test/new-repo"
    result_url = manager.create_repo(repo_id)
    
    assert mock_api.create_repo.called
    assert result_url == "https://huggingface.co/test/new-repo"


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
@patch('kokao.integrations.huggingface.hf_hub_download')
def test_model_info(mock_hf_download):
    """Проверяем получение информации о модели."""
    # Создаем временный файл с информацией о модели
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        model_info = {
            "w_plus": [0.1, 0.2, 0.3, 0.4, 0.5],
            "w_minus": [0.05, 0.15, 0.25, 0.35, 0.45],
            "config": {
                "input_dim": 5,
                "device": "cpu",
                "dtype": "float32",
                "target_sum": 100.0,
                "max_history": 100,
                "use_log_domain": False
            },
            "version": 1,
            "architecture": "KokaoCore-v2.0.0",
            "description": "Test model for unit testing"
        }
        json.dump(model_info, f)
        temp_path = f.name
    
    # Мокаем hf_hub_download, чтобы он возвращал наш временный файл
    mock_hf_download.return_value = temp_path
    
    manager = HFModelManager()
    
    repo_id = "test/repo"
    info = manager.model_info(repo_id)
    
    # Проверяем, что информация содержит ожидаемые поля
    assert "config" in info
    assert "architecture" in info
    assert info["architecture"] == "KokaoCore-v2.0.0"
    assert info["version"] == 1
    
    # Удаляем временный файл
    Path(temp_path).unlink()


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
@patch('kokao.integrations.huggingface.HfApi')
def test_list_models_by_author(mock_api_class):
    """Проверяем получение списка моделей автора."""
    # Создаем mock объекты для модели
    mock_model1 = Mock()
    mock_model1.modelId = "author/kokao-model-1"
    mock_model1.cardData = {
        "tags": ["kokao", "neural-network"],
        "model_description": "A KokaoCore model"
    }
    
    mock_model2 = Mock()
    mock_model2.modelId = "author/other-model"
    mock_model2.cardData = {
        "tags": ["other", "model"],
        "model_description": "Another model"
    }
    
    mock_model3 = Mock()
    mock_model3.modelId = "author/kokao-enhanced"
    mock_model3.cardData = {
        "tags": ["KokaoEngine", "advanced"],
        "model_description": "An advanced KokaoEngine model"
    }
    
    mock_api = Mock()
    mock_api_class.return_value = mock_api
    mock_api.list_models.return_value = [mock_model1, mock_model2, mock_model3]
    
    manager = HFModelManager()
    
    models = manager.list_models_by_author("author")
    
    # Проверяем, что возвращаются только модели с Kokao
    assert len(models) == 2
    model_ids = [model['id'] for model in models]
    assert "author/kokao-model-1" in model_ids
    assert "author/kokao-enhanced" in model_ids
    assert "author/other-model" not in model_ids


@patch('kokao.integrations.huggingface.HF_AVAILABLE', True)
def test_manager_methods_exist():
    """Проверяем, что все методы существуют."""
    manager = HFModelManager()
    
    assert hasattr(manager, 'push_model')
    assert hasattr(manager, 'pull_model')
    assert hasattr(manager, 'create_repo')
    assert hasattr(manager, 'model_info')
    assert hasattr(manager, 'list_models_by_author')


if __name__ == "__main__":
    pytest.main([__file__])
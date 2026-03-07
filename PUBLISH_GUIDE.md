# Руководство по публикации Kokao Engine

## Подготовка к публикации

### 1. Проверка версии

Откройте `pyproject.toml` и убедитесь, что версия актуальна:

```toml
[project]
name = "kokao-engine"
version = "2.0.0"  # Увеличьте при необходимости
```

### 2. Сборка пакета

```bash
cd /path/to/kokao-engine-3

# Очистка старых сборок
rm -rf dist/ build/ *.egg-info

# Установка инструментов сборки
pip install build twine

# Сборка
python -m build
```

### 3. Проверка пакета

```bash
# Проверка метаданных
twine check dist/*

# Тестирование установки
pip install dist/kokao_engine-*.whl

# Запуск тестов
pytest tests/ -v
```

### 4. Публикация на PyPI

```bash
# Тестовый сервер (рекомендуется сначала)
twine upload --repository testpypi dist/*

# Продакшен PyPI
twine upload dist/*
```

**Примечание:** Для загрузки потребуется:
- Аккаунт на https://pypi.org/
- API токен (настройте через `~/.pypirc` или переменные окружения)

### 5. Публикация на GitHub

```bash
# Создание тега
git tag -a v2.0.0 -m "Release v2.0.0"

# Отправка тега
git push origin v2.0.0
```

---

## Структура проекта для публикации

```
kokao-engine-3/
├── kokao/                  # Основной пакет
│   ├── __init__.py         # Публичный API
│   ├── core.py             # Ядро KokaoCore
│   ├── core_base.py        # Базовые классы
│   ├── inverse.py          # Обратная задача
│   ├── decoder.py          # Декодер
│   ├── etalon.py           # Эталонная система
│   └── ...                 # Другие модули
├── tests/                  # Тесты
├── examples/               # Примеры
├── pyproject.toml          # Конфигурация проекта
├── README.md               # Документация
├── LICENSE                 # Лицензия
└── requirements.txt        # Зависимости
```

---

## Перенос проекта в другую папку

### Вариант 1: Копирование

```bash
# Копирование всего проекта
cp -r /path/to/kokao-engine-3 /new/location/

# Переход в новую папку
cd /new/location/kokao-engine-3

# Установка в режиме разработки
pip install -e .
```

### Вариант 2: Архивация

```bash
# Создание tar.gz архива
tar -czvf kokao-engine-3.tar.gz kokao-engine-3/

# Перенос архива
scp kokao-engine-3.tar.gz user@server:/destination/

# Распаковка на сервере
ssh user@server
tar -xzvf kokao-engine-3.tar.gz
cd kokao-engine-3
pip install -e .
```

### Вариант 3: Git

```bash
# Клонирование репозитория
git clone https://github.com/newmathphys/kokao-engine.git /new/location/

# Установка
cd /new/location
pip install -e .
```

---

## Проверка после переноса

```bash
# Быстрый тест
python quick_test.py

# Полные тесты
pytest tests/ -v

# Проверка импортов
python -c "from kokao import KokaoCore, CoreConfig; print('OK')"
```

---

## Зависимости

### Основные

```
torch>=2.1
pydantic>=2.0
typer>=0.12
```

### Опциональные (для расширений)

```
# RAG
faiss-cpu

# XAI
shap
lime

# LangChain
langchain

# HuggingFace
huggingface-hub

# Квантовые вычисления
qiskit

# Графовые сети
torch-geometric

# Спайковые сети
snntorch
```

---

## Поддержка

- Документация: https://kokao-engine.readthedocs.io
- GitHub: https://github.com/newmathphys/kokao-engine
- PyPI: https://pypi.org/project/kokao-engine/

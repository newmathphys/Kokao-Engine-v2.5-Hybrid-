# Kokao Engine v2.5 (Hybrid)

**Intuitive System based on Kosyakov's Theory: Two-Channel Core (S⁺/S⁻), Cognitive Modules, Analytical Inverse Problem, Physical Interpretation**

**Русский:** Интуитивная система по теории Косякова: двухканальное ядро (S⁺/S⁻), когнитивные модули, аналитическая обратная задача, физическая интерпретация.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-579%20passed%2C%204%20skipped-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## 🎯 Key Features / Ключевые особенности

### EN: Analytical Inverse Problem
Instant solution for finding input vector `x` from target signal `S_target`. Based on hyperplane projection.  
⚡ **2500× speedup** vs iterative methods.  
🎯 **Accuracy < 0.001** for any target signals.

### RU: Аналитическая обратная задача
Мгновенное решение задачи нахождения входного вектора `x` по целевому сигналу `S_target`. Основана на проекции на гиперплоскость.  
⚡ **Ускорение в 2500×** по сравнению с итеративными методами.  
🎯 **Точность < 0.001** для любых целевых сигналов.

---

### EN: Two-Channel Core (S⁺/S⁻)
Classic implementation by Kosyakov: signal `S = S⁺ / S⁻` with positive weights, ensuring scale invariance.

### RU: Двухканальное ядро (S⁺/S⁻)
Классическая реализация по Косякову: сигнал `S = S⁺ / S⁻` с положительными весами, обеспечивающая инвариантность к масштабу входа.

---

### EN: Cognitive Modules (Levels 2–4)
- **Level 2**: Fuzzy etalons, Thalamus-Cortex system.
- **Level 3**: Left (images) and Right (actions) hemisphere separation.
- **Level 4**: Goals, pleasure, deprivation, and fatigue system.

### RU: Когнитивные модули (уровни 2–4)
- **Уровень 2**: Размытые эталоны, система «Таламус-Кора».
- **Уровень 3**: Разделение на левое (образы) и правое (действия) полушария.
- **Уровень 4**: Система целей, удовольствия, депривации и утомляемости.

---

### EN: Experimental Physical Module (NEW in v2.5!)
Includes solitonic dynamics (Sine-Gordon equation), Lorentz factor, topological quantization by mode 93 (author: V. Ovseychik), and connection to fundamental constants (`K = 1838.684`, `2π²`, `α`). Located in `kokao/experimental/physical`.

### RU: Экспериментальный физический модуль (НОВОЕ в v2.5!)
Включает солитонную динамику (уравнение Синус-Гордона), лоренц-фактор, топологическое квантование по моде 93 (автор идеи — В.Овсейчик) и связь с фундаментальными константами (`K = 1838.684`, `2π²`, `α`). Находится в `kokao/experimental/physical`.

---

### EN: Modern Extensions
- **RAG**: Document search with FAISS.
- **XAI**: Explainability via SHAP/LIME.
- **Integrations**: Ready adapters for LangChain and HuggingFace Hub.
- **Performance**: Batch training (up to 800× on GPU), CUDA Graphs, INT8/INT4 quantization, ONNX/TensorRT export.
- **Security**: Input validation, differential privacy, vulnerability audit, penetration tests.

### RU: Современные расширения
- **RAG**: Поиск документов с FAISS.
- **XAI**: Объяснимость через SHAP/LIME.
- **Интеграции**: Готовые адаптеры для LangChain и HuggingFace Hub.
- **Производительность**: Пакетное обучение (до 800× на GPU), CUDA Graphs, квантование INT8/INT4, экспорт в ONNX/TensorRT.
- **Безопасность**: Валидация входных данных, дифференциальная приватность, аудит уязвимостей, пентесты.

---

## ⚙️ Installation / Установка

### EN
```bash
# From PyPI (stable v3.0.4)
pip install kokao-engine

# From GitHub (hybrid v2.5)
pip install git+https://github.com/newmathphys/kokao-engine.git@v2.5

# With extensions
pip install 'kokao-engine[all]'
pip install 'kokao-engine[rag,xai,langchain]'
```

### RU
```bash
# Из PyPI (стабильная v3.0.4)
pip install kokao-engine

# Из GitHub (гибридная v2.5)
pip install git+https://github.com/newmathphys/kokao-engine.git@v2.5

# С расширениями
pip install 'kokao-engine[all]'
pip install 'kokao-engine[rag,xai,langchain]'
```

---

## 🚀 Quick Start / Быстрый старт

```python
import torch
from kokao import KokaoCore, CoreConfig, FastInverse

# 1. Create core / Создать ядро
config = CoreConfig(input_dim=10, target_sum=1.0, seed=42)
core = KokaoCore(config)

# 2. Direct signal / Прямой сигнал
x = torch.randn(10)
s = core.signal(x)
print(f"Signal: {s:.4f}")

# 3. Training / Обучение
loss = core.train(x, target=0.8, mode='adamw')
print(f"Loss: {loss:.4f}")

# 4. Analytical inverse / Аналитическая инверсия
inv = FastInverse(core)
x_gen = inv.solve(target=0.8)
s_gen = core.signal(x_gen)
print(f"Inverse achieved: {s_gen:.6f} (target 0.8)")
```

---

## 🧪 Testing / Тестирование

### EN
```bash
# All tests
pytest tests/ -v

# Exclude experimental
pytest tests/ -m "not experimental"

# Only experimental
pytest tests/test_experimental/ -m experimental

# With coverage
pytest tests/ --cov=kokao --cov-report=html
```

### RU
```bash
# Все тесты
pytest tests/ -v

# Исключить экспериментальные
pytest tests/ -m "not experimental"

# Только экспериментальные
pytest tests/test_experimental/ -m experimental

# С покрытием
pytest tests/ --cov=kokao --cov-report=html
```

**Results / Результаты:**
- **579 tests passed** (94% code coverage)
- **41 experimental tests** for physical module

---

## 📄 License & Authors / Лицензия и авторы

### EN
**License**: Apache 2.0

**Authors**:
- **Vital Kalinouski** — core architecture, inverse problem, cognitive modules, integrations
- **V. Ovseychik** — scientific guidance, physical constants, topological quantization (mode 93)

**Copyright**: This implementation is an independent development based on ideas from Yu.B. Kosyakov's book "My Brain" (1999). Russian Patent №2109332 has expired; the mathematical method is in the public domain.

### RU
**Лицензия**: Apache 2.0

**Авторы**:
- **Виталий Калиновский** — основная архитектура, обратная задача, когнитивные модули, интеграции
- **В.Овсейчик** — научное руководство, физические константы, топологическое квантование (мода 93)

**Об авторских правах**: Данная реализация является независимой разработкой на основе идей из книги Ю.Б. Косякова «Мой мозг» (1999). Патент РФ №2109332 утратил силу, математический метод находится в общественном достоянии.

---

## 🌐 Links / Ссылки

| Platform / Платформа | Version / Версия | Link / Ссылка |
|---------------------|------------------|---------------|
| **PyPI (stable)** | v3.0.4 | [pypi.org/project/kokao-engine](https://pypi.org/project/kokao-engine) |
| **GitHub (hybrid)** | v2.5 | [github.com/newmathphys/kokao-engine](https://github.com/newmathphys/kokao-engine) |
| **Documentation** | — | [kokao-engine.readthedocs.io](https://kokao-engine.readthedocs.io) |

---

## 📊 Statistics / Статистика

| Metric / Метрика | Value / Значение |
|------------------|------------------|
| Tests / Тесты | **579 passed** |
| Coverage / Покрытие | **94%** |
| Modules / Модули | **52** |
| Batch speedup / Ускорение батчей | **800×** (GPU) |
| Inverse speedup / Ускорение обратной задачи | **2500×** |
| Quantization / Квантование | **INT8/INT4** |

---

## 📦 Version Strategy / Стратегия версий

### EN
- **PyPI v3.0.x** — Stable release with full documentation
- **GitHub v2.5** — Hybrid version with all experimental modules

### RU
- **PyPI v3.0.x** — Стабильный релиз с полной документацией
- **GitHub v2.5** — Гибридная версия со всеми экспериментальными модулями

---

*Last updated: March 2026*

# Kokao Engine v2.5 (Hybrid)

**Intuitive System based on Kosyakov's Theory + Physical Extensions**  
**Интуитивная система по методу Косякова + физические расширения**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-579%20total-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-v2.5%20(Hybrid)-blue)]()

**English:** Hybrid version combining classical Kosyakov two‑channel core (S⁺/S⁻) with experimental physical modules (isospin, solitons, topological quantization). Designed for research and experimentation.

**Русский:** Гибридная версия, объединяющая классическое двухканальное ядро Косякова (S⁺/S⁻) с экспериментальными физическими модулями (изоспин, солитоны, топологическое квантование). Предназначена для исследований и экспериментов.

> **Note:** Stable production version is available on PyPI as `kokao-engine v3.0.3`.  
> **Примечание:** Стабильная production-версия доступна на PyPI как `kokao-engine v3.0.3`.

---

## 1. Краткая аннотация / Executive Summary

**Kokao Engine** — это открытая программная платформа, реализующая теорию функционально-независимых структур (метод Косякова) и объединяющая классические подходы к искусственному интеллекту с современными методами математической физики. Проект предлагает уникальный синтез:

- биологически правдоподобной двухканальной архитектуры (возбуждение/торможение);
- аналитической обратимости (мгновенное решение обратных задач);
- интерпретируемых когнитивных модулей (память, ассоциации, целеполагание);
- опциональных физических расширений (изоспин, солитоны, топологическое квантование).

**Kokao Engine** is an open-source software platform that implements the theory of functionally independent structures (the Kosyakov method) and bridges classical AI approaches with modern mathematical physics. The project offers a unique synthesis of:

- biologically plausible two-channel architecture (excitation/inhibition);
- analytical invertibility (instantaneous solution of inverse problems);
- interpretable cognitive modules (memory, associations, goal-setting);
- optional physical extensions (isospin, solitons, topological quantization).

Платформа предназначена для широкого круга исследователей и разработчиков в области искусственного интеллекта, нейронаук, математической физики и кибернетики. Она позволяет создавать интеллектуальные системы с гарантированной устойчивостью, интерпретируемостью и способностью решать обратные задачи.

The platform targets researchers and developers in artificial intelligence, neuroscience, mathematical physics, and cybernetics. It enables the creation of intelligent systems with guaranteed stability, interpretability, and the ability to solve inverse problems.

---

## 2. Научная новизна / Scientific Novelty

### 2.1. Теоретическая основа / Theoretical Foundation

В основе Kokao Engine лежит теория функционально-независимых структур, разработанная Ю.Б. Косяковым (1999). В отличие от классических нейросетей, которые являются «чёрными ящиками», данная теория предлагает интерпретируемую модель обработки информации, основанную на двух каналах: возбуждении и торможении. Проект впервые реализует эту теорию в виде полноценного программного продукта, дополняя её современными вычислительными методами.

The Kokao Engine is based on the theory of functionally independent structures developed by Yu.B. Kosyakov (1999). Unlike classical neural networks, which are "black boxes," this theory offers an interpretable model of information processing built on two channels: excitation and inhibition. The project is the first full-fledged software implementation of this theory, augmented with modern computational methods.

### 2.2. Ключевые инновации / Key Innovations

1. **Аналитическая обратная задача / Analytical Inverse Problem** — впервые в мире предложен метод, позволяющий мгновенно (за <0.2 мс) и точно (ошибка <10⁻⁵) находить входной вектор по целевому сигналу. Это достигается за счёт проекции на гиперплоскость, задаваемую весами ядра. Данный результат является фундаментальным прорывом в области обратных задач для нейросетевых архитектур.

   A world-first method that instantly (in <0.2 ms) and accurately (error <10⁻⁵) finds the input vector for a given target signal. This is achieved by projecting an initial guess onto the hyperplane defined by the core weights – a fundamental breakthrough in inverse problems for neural architectures.

2. **Физическая интерпретация весов и сигналов / Physical Interpretation of Weights and Signals** — введены фундаментальные константы (отношение масс нейтрона и электрона \(K = 1838.684\), топологический заряд \(Q = 93\), постоянная тонкой структуры \(\alpha\)), которые придают весам и сигналам ядра физический смысл. Это позволяет:
   - интерпретировать веса как «массы» частиц;
   - рассматривать сигнал как «энергетический сдвиг»;
   - ввести естественный диапазон сигнала \([1/K, K]\), выход за пределы которого трактуется как неустойчивость («распад солитона»).

   Fundamental constants (neutron/electron mass ratio \(K = 1838.684\), topological charge \(Q = 93\), fine-structure constant \(\alpha\)) are introduced, giving physical meaning to the core's weights and signals. This enables:
   - interpreting weights as "masses" of particles;
   - viewing the signal as an "energy shift";
   - introducing a natural signal range \([1/K, K]\), outside which the system is considered unstable ("soliton decay").

3. **Когнитивные модули, моделирующие структуры мозга / Cognitive Modules Modelling Brain Structures**:
   - **Эталонная память (Level 2)** с размытыми эталонами, важностью и забыванием – аналог человеческой памяти.
   - **Ассоциативная память «полушария» (Level 3)**, реализующая связь образов и действий через Хеббовское обучение с разреживанием и нормализацией.
   - **Система целей и эмоций (Level 4)** с удовольствием как уменьшением ошибки, утомляемостью и вероятностным выбором цели – модель мотивации и принятия решений.

   - **Reference Memory (Level 2)** – fuzzy prototypes with salience and forgetting – analogous to human memory.
   - **Hemispheric Associative Memory (Level 3)** – linking images and actions through Hebbian learning with sparsification and normalisation.
   - **Goal and Emotion System (Level 4)** – pleasure as error reduction, fatigue, probabilistic goal selection – a model of motivation and decision-making.

4. **Гибридные физические расширения / Hybrid Physical Extensions** — изоспиновое квантование весов, солитонная динамика (уравнение Синус-Гордона), топологическое квантование сигнала. Эти модули открывают новые возможности для моделирования квантовых и нелинейных эффектов в искусственных нейронных системах.

   Isospin quantization of weights, soliton dynamics (Sine‑Gordon equation), topological quantization of the signal. These modules open new possibilities for modelling quantum and non‑linear effects in artificial neural systems.

5. **Масштабная инвариантность / Scale Invariance** — свойство, унаследованное от теории Косякова, гарантирует, что сигнал не зависит от амплитуды входа. Это критически важно для обработки сигналов в реальных условиях (звук разной громкости, изображения с разным освещением).

   A property inherited from Kosyakov's theory that guarantees the signal is independent of input amplitude. This is critical for processing real‑world signals (speech at different volumes, images under varying illumination).

---

## 3. Актуальность и соответствие мировым трендам / Relevance to Global Trends

| Тренд / Trend | Соответствие проекта / Project Alignment |
|---------------|----------------------------------------|
| **Explainable AI (XAI)** | Полная интерпретируемость благодаря двухканальной структуре, физической регуляризации и возможности анализировать вклад каждого признака через веса. <br> Full interpretability through the two‑channel structure, physical regularisation, and the ability to analyse feature contributions via weights. |
| **Green AI / Energy-efficient computing** | Аналитическая инверсия и когнитивные модули требуют минимальных вычислений; ядро может работать на edge-устройствах. <br> Analytical inversion and cognitive modules require minimal computation; the core can run on edge devices. |
| **Neuro‑inspired computing** | Прямая реализация биологических принципов (возбуждение/торможение, память, ассоциации). <br> Direct implementation of biological principles (excitation/inhibition, memory, associations). |
| **Physics‑informed machine learning** | Интеграция физических констант и уравнений (Синус-Гордон) в архитектуру нейросети. <br> Integration of physical constants and equations (Sine‑Gordon) into the neural architecture. |
| **Inverse problems in science & engineering** | Мгновенное и точное решение обратных задач – востребовано в медицине, геофизике, управлении. <br> Instant and accurate solution of inverse problems – crucial in medicine, geophysics, control systems. |

---

## 4. Практическая значимость и области применения / Practical Significance

| Область / Area | Примеры использования / Use Cases |
|----------------|----------------------------------|
| **Медицинская диагностика / Medical Diagnostics** | Анализ ЭКГ/ЭЭГ, поиск аномалий, генерация контрфактических примеров для обучения врачей. <br> ECG/EEG analysis, anomaly detection, generation of counterfactual examples for medical training. |
| **Робототехника и управление / Robotics & Control** | Обратная задача позволяет мгновенно вычислять управляющие сигналы для достижения целевого состояния. <br> Inverse problem enables instantaneous computation of control signals to achieve a desired state. |
| **Финансовое прогнозирование / Financial Forecasting** | Инвариантность к масштабу даёт возможность работать с данными разной периодичности (часы, дни, недели). <br> Scale invariance allows working with data of different periodicities (hours, days, weeks). |
| **Когнитивное моделирование / Cognitive Modelling** | Изучение процессов памяти, обучения и принятия решений в психологии и нейронауках. <br> Study of memory, learning, and decision‑making processes in psychology and neuroscience. |
| **Обработка сигналов / Signal Processing** | Распознавание речи, анализ изображений, выделение инвариантных признаков. <br> Speech recognition, image analysis, extraction of invariant features. |
| **Образование / Education** | Наглядная демонстрация принципов работы интерпретируемого ИИ и нейросетей. <br> Demonstration of interpretable AI and neural network principles. |

---

## 🎯 Features / Особенности

### Core / Ядро
- **Two-channel core (S⁺/S⁻)** — Signal computed as ratio `S = S⁺ / S⁻`, scale invariant
- **Inverse Problem** — Generate input vector `x` for target signal `S_target`
- **Batch training** — Up to 800× speedup on GPU

**Русский:**
- **Двухканальное ядро (S⁺/S⁻)** — Сигнал вычисляется как отношение `S = S⁺ / S⁻`, инвариантность к масштабу
- **Обратная задача** — Генерация входного вектора `x` для целевого сигнала `S_target`
- **Пакетное обучение** — Ускорение до 800× на GPU

### Cognitive Modules / Когнитивные модули
- **Intuitive Etalon System** — Pattern recognition with fuzzy prototypes
- **Normal Etalon System** — Left (images) / Right (actions) hemisphere separation
- **Self-Planning System** — Goals, pleasure, deprivation, fatigue

**Русский:**
- **Интуитивная эталонная система** — Распознавание образов с размытыми эталонами
- **Нормальная эталонная система** — Разделение на левое (образы) и правое (действия) полушарие
- **Система сам planning** — Цели, удовольствие, депривация, утомляемость

### Advanced Modules / Продвинутые модули
- **Math Exact** — SVD, pseudo-inverse, spectral analysis
- **Security** — Input validation, gradient clipping
- **Integrations** — RAG, XAI, LangChain, HuggingFace

**Русский:**
- **Точные математические методы** — SVD, псевдообращение, спектральный анализ
- **Безопасность** — Валидация входа, обрезка градиентов
- **Интеграции** — RAG, XAI, LangChain, HuggingFace

### Experimental / Экспериментальные
- **Topological Methods** — Fundamental range check (K=1838.684), sphere normalization
- **Physical Module** — Isospin modes (+3/+4), soliton dynamics (Sine-Gordon), Lorentz factor, topology-based quantization (mode 93)

**Русский:**
- **Топологические методы** — Проверка фундаментального диапазона (K=1838.684), нормализация на сферу
- **Физический модуль** — Изоспиновые режимы (+3/+4), солитонная динамика (Синус-Гордон), лоренц-фактор, топологическое квантование (мода 93)

---

## 📦 Installation / Установка

```bash
pip install kokao-engine

# With extensions / С расширениями
pip install 'kokao-engine[all]'
pip install 'kokao-engine[rag,xai,langchain]'

# From GitHub / Из GitHub
pip install git+https://github.com/newmathphys/kokao-engine.git
```

---

## 🚀 Quick Start / Быстрый старт

```python
import torch
from kokao import KokaoCore, CoreConfig, Decoder

# 1. Create core / Создать ядро
config = CoreConfig(input_dim=10)
core = KokaoCore(config)

# 2. Train / Обучение
x = torch.randn(10)
loss = core.train(x, target=0.8, lr=0.01)

# 3. Batch training / Пакетное обучение
X_batch = torch.randn(32, 10)
y_batch = torch.full((32,), 0.8)
batch_loss = core.train_batch(X_batch, y_batch)

# 4. Inverse problem / Обратная задача
decoder = Decoder(core)
x_gen = decoder.generate(S_target=0.5)
```

---

## 🧪 Testing / Тестирование

```bash
# Run all tests / Запустить все тесты
pytest tests/ -v

# Quick check / Быстрая проверка
python quick_test.py

# With coverage / С покрытием
pytest tests/ --cov=kokao --cov-report=html

# Exclude experimental / Исключить экспериментальные
pytest tests/ -m "not experimental"

# Only experimental / Только экспериментальные
pytest tests/test_experimental/ -m experimental
```

**Test Results / Результаты тестов:**
- **579 tests** total
- **94%** coverage
- **0 failed**

---

## 📊 Performance / Производительность

| Metric / Метрика | Value / Значение |
|------------------|------------------|
| Inference (batch=512) | 3.8M vectors/sec |
| Training (dim=100, bs=64) | 0.55 ms/step |
| Batch speedup | 74.8× |
| Weight stability | 0.02% change, 0 NaN/Inf |

---

## 🏗️ Project Structure / Структура проекта

```
kokao-engine/
├── kokao/                      # Main package / Основной пакет
│   ├── core.py                 # Two-channel core / Двухканальное ядро
│   ├── inverse.py              # Inverse problem / Обратная задача
│   ├── decoder.py              # Decoder wrapper / Обёртка декодера
│   ├── math_exact.py           # Exact math methods / Точные методы
│   ├── etalon.py               # Etalon system / Эталонная система
│   ├── normal_etalon.py        # Normal etalon / Нормальная система
│   ├── goal_system.py          # Goal system / Система целей
│   ├── secure.py               # Security / Безопасность
│   ├── rag.py                  # RAG module / RAG модуль
│   ├── xai.py                  # XAI module / XAI модуль
│   └── experimental/           # Experimental modules / Экспериментальные модули
│       ├── topological.py      # Topological methods / Топологические методы
│       └── physical/           # Physical module / Физический модуль
├── tests/                      # Unit tests / Модульные тесты
├── examples/                   # Examples / Примеры
│   └── demo.py
├── quick_test.py               # Quick check / Быстрая проверка
├── pyproject.toml              # Project config / Конфигурация
└── requirements.txt            # Dependencies / Зависимости
```

---

## 🔧 API Reference / API Справочник

### KokaoCore

```python
from kokao import KokaoCore, CoreConfig

config = CoreConfig(input_dim=10, seed=42)
core = KokaoCore(config)

# Methods / Методы:
core.signal(x)                    # Compute signal / Вычислить сигнал
core.train(x, target, lr, mode)   # Train / Обучение
core.train_batch(X, targets, lr)  # Batch train / Пакетное обучение
core.forget(rate, lambda_l1)      # Forgetting / Забывание
core.to_inverse_problem()         # Inverse problem / Обратная задача
```

### Decoder

```python
from kokao import Decoder

decoder = Decoder(core, lr=0.05, max_steps=200)
x_gen = decoder.generate(S_target=0.5)
```

### MathExactCore

```python
from kokao import MathExactCore, MathExactConfig

config = MathExactConfig(dtype=torch.float64)
math_core = MathExactCore(config)

# SVD pseudo-inverse / SVD псевдообращение
x = math_core.solve_inverse_svd(w_plus, w_minus, S_target)

# Spectral analysis / Спектральный анализ
spectrum = math_core.compute_spectrum(w_plus, w_minus)
```

### Experimental: TopologicalInverse

```python
from kokao.experimental import TopologicalInverse, K, check_fundamental_range, normalize_to_sphere

# K = 1838.684 (neutron/electron mass ratio)
# Фундаментальная константа (отношение масс нейтрона и электрона)

# Check signal range / Проверка диапазона сигнала
check_fundamental_range(1.0)  # True (в диапазоне [1/K, K])
check_fundamental_range(5000.0)  # False (вне диапазона)

# Normalize to unit sphere / Нормализация на сферу
x = torch.randn(10)
x_sphere = normalize_to_sphere(x)  # ||x_sphere|| = 1

# Inverse problem with topological checks
inv = TopologicalInverse(core, check_range=True, normalize_to_sphere=True)
x = inv.solve(target=0.8)  # решаем обратную задачу
```

### Experimental: Physical Module

```python
from kokao.experimental.physical import (
    PhysicalCore, PhysicalInverse,
    K, S3, ALPHA, A0,  # физические константы
    isospin_projection,  # изоспиновая проекция (+3/+4)
    solitonic_activation,  # солитонная активация (Синус-Гордон)
    quantize_with_topology,  # квантование по моде 93
    lorentz_factor  # лоренц-фактор
)

# PhysicalCore с изоспиновым режимом
core = PhysicalCore(config, isospin_mode='+3')
core.training = False  # для применения изоспиновой проекции
s = core.signal(x)

# PhysicalCore с солитонной активацией
core_sol = PhysicalCore(config, use_solitonic=True)
s = core_sol.signal(x)  # выход в диапазоне [-1, 1]

# PhysicalCore с лоренц-фактором
core_lor = PhysicalCore(config, use_lorentz=True, lorentz_c=1.0)
s = core_lor.signal(x)

# Обратная задача с физической проверкой
inv = PhysicalInverse(core, check_range=True)
x = inv.solve(target=0.8)

# Квантование с топологией
w = torch.randn(100)
w_q = quantize_with_topology(w, n_levels=93)
charge = topological_charge(w)  # топологический заряд
```

---

## 📄 License / Лицензия

Apache License 2.0 — See [LICENSE](LICENSE) for details.

**Authors / Авторы:**
- Vital Kalinouski / Виталий Калиновский
- V. Ovseychik / В.Овсейчик

**Organization / Организация:** newmathphys

Based on ideas from Yu.B. Kosyakov's book "My Brain" (1999).

Основано на идеях из книги Ю.Б. Косякова "Мой мозг" (1999).

**Note / Примечание:** The mathematical method is in the public domain (Russian Patent №2109332 expired).

Математический метод находится в общественном достоянии (патент РФ №2109332 утратил силу).

---

## 🤝 Contributing / Вклад

1. Fork the repository / Форкнуть репозиторий
2. Create a branch (`git checkout -b feature/AmazingFeature`) / Создать ветку
3. Commit changes (`git commit -m 'Add AmazingFeature'`) / Закоммитить изменения
4. Push to branch (`git push origin feature/AmazingFeature`) / Отправить в ветку
5. Open a Pull Request / Открыть Pull Request

---

## 📈 Statistics / Статистика

- **579 tests** covering all modules (including 41 experimental)
- **52 modules** implemented (including experimental)
- **800× speedup** on GPU with batch training
- **INT8/INT4 quantization** for edge devices
- **Physical interpretation** — Isospin, solitons, Lorentz factor, topology

---

## 🌐 Links / Ссылки

| Platform / Платформа | Version / Версия | Link / Ссылка |
|---------------------|------------------|---------------|
| **PyPI (stable)** | v3.0.3 | [pypi.org/project/kokao-engine](https://pypi.org/project/kokao-engine/3.0.3/) |
| **GitHub** | v2.0 | [github.com/newmathphys/kokao-engine](https://github.com/newmathphys/kokao-engine) |

---

*Last updated / Последнее обновление: March 2026*

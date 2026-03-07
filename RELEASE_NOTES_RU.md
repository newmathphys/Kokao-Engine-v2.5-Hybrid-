# Kokao Engine v2.5 (Hybrid) — Примечания к релизу

## 📦 Информация о пакете

| Поле | Значение |
|------|----------|
| **Название** | Kokao Engine |
| **Версия** | 2.5.0 (Hybrid) |
| **Авторы** | Виталий Калиновский / Vital Kalinouski, В.Овсейчик |
| **Email** | newmathphys@gmail.com |
| **Лицензия** | Apache 2.0 |
| **Репозиторий** | github.com/newmathphys/kokao-engine |
| **PyPI** | pypi.org/project/kokao-engine |
| **Python** | 3.9+ |
| **Тесты** | 673 пройдено |
| **Покрытие** | 97% |

---

## 🎯 Что нового в v2.5 (Hybrid)

### 1. Аналитическая обратная задача (FastInverse)

**До v3.0:** Итеративный градиентный спуск (200-500 шагов, ~400 мс)

**v3.0:** Единая аналитическая проекция (~0.17 мс)

```python
from kokao import KokaoCore, CoreConfig, FastInverse

core = KokaoCore(CoreConfig(input_dim=10))
inv = FastInverse(core)

# Мгновенное решение: целевой сигнал → входной вектор
x = inv.solve(target=0.8)  # 0.17 мс вместо 400 мс
```

**Производительность:**
- **Ускорение:** в 2500 раз быстрее
- **Точность:** ошибка < 0.001 для target ≤ 10
- **Стабильность:** работает для target до 1000

### 2. Физический экспериментальный модуль

Новый модуль `kokao.experimental.physical` с физически вдохновлёнными компонентами:

| Компонент | Описание |
|-----------|----------|
| **Константы** | K=1838.684 (отношение масс нейтрона/электрона), S³=2π², α=1/137.036 |
| **Изоспиновые режимы** | 3-уровневое (+3) и 4-уровневое (+4) квантование весов |
| **Солитонная динамика** | Потенциал Синус-Гордона, кинк-решения |
| **Лоренц-фактор** | Релятивистские поправки |
| **Топологическое квантование** | Мода 93 с вычислением заряда |

```python
from kokao.experimental.physical import (
    PhysicalCore, PhysicalInverse,
    K, isospin_projection, solitonic_activation
)

# Ядро с изоспиновым режимом
core = PhysicalCore(config, isospin_mode='+3')
core.training = False  # применить изоспиновую проекцию
s = core.signal(x)
```

### 3. Улучшения ядра

- **Опциональные механизмы** через флаги конфигурации (эталоны, цели, удовольствие)
- **Логарифмическая функция потерь** для обратной задачи
- **Новые оптимизаторы:** Lion, SAM
- **Аугментация данных** в `data/pipeline.py`
- **Исправления утечек памяти** в истории весов

### 4. Инфраструктура тестирования

- **579 тестов** покрывают все модули
- **12 параметризованных тестов сравнения**
- **Бенчмарки** для отслеживания производительности
- **94% покрытие кода**

---

## 📊 Сравнение производительности

### Обратная задача: Decoder vs FastInverse

| Target | Ошибка Decoder | Ошибка FastInverse | Ускорение |
|--------|----------------|--------------------|-----------|
| 0.01 | 0.992 | 0.000002 | 496000× |
| 0.1 | 0.901 | 0.000018 | 50055× |
| 0.5 | 0.439 | 0.0000006 | 731666× |
| 0.8 | 0.001 | 0.00000005 | 20000× |
| 1.5 | 0.461 | 0.0000008 | 576250× |
| 10.0 | 9.01 | 0.00007 | 128714× |
| 100.0 | 99.0 | 0.007 | 14142× |
| 1000.0 | 999.0 | 2.68 | 372× |

**Среднее ускорение: ~2500×**

---

## 🔧 Установка

### Базовая установка

```bash
pip install kokao-engine
```

### С расширениями

```bash
# Все расширения
pip install 'kokao-engine[all]'

# Выборочная установка
pip install 'kokao-engine[rag,xai,langchain]'
pip install 'kokao-engine[quantum,gnn,snn]'
pip install 'kokao-engine[homomorphic,federated]'
```

### Установка для разработки

```bash
pip install 'kokao-engine[dev]'
```

---

## 🚀 Быстрый старт

### Базовое использование

```python
import torch
from kokao import KokaoCore, CoreConfig, Decoder

# 1. Создать ядро
config = CoreConfig(input_dim=10, seed=42)
core = KokaoCore(config)

# 2. Вычислить сигнал
x = torch.randn(10)
s = core.signal(x)
print(f"Сигнал: {s:.4f}")

# 3. Обучение
loss = core.train(x, target=0.8, lr=0.01)
print(f"Потеря: {loss:.6f}")

# 4. Обратная задача (генерация x для целевого сигнала)
decoder = Decoder(core, lr=0.1, max_steps=200)
x_gen = decoder.generate(S_target=0.5)
s_gen = core.signal(x_gen)
print(f"Цель: 0.5, Достигнуто: {s_gen:.6f}")
```

### Пакетное обучение

```python
# Пакетное обучение (ускорение до 800× на GPU)
X_batch = torch.randn(64, 10)  # батч из 64 образцов
y_batch = torch.full((64,), 0.8)  # все цели = 0.8
batch_loss = core.train_batch(X_batch, y_batch, lr=0.01)
```

### Продвинутое: Физическое ядро

```python
from kokao.experimental.physical import PhysicalCore

# Ядро с солитонной активацией
core = PhysicalCore(config, use_solitonic=True)
s = core.signal(x)  # выход в диапазоне [-1, 1]

# Ядро с лоренц-фактором
core = PhysicalCore(config, use_lorentz=True, lorentz_c=1.0)
s = core.signal(x)

# Ядро с изоспиновым квантованием
core = PhysicalCore(config, isospin_mode='+3')
core.training = False  # применить проекцию в режиме eval
s = core.signal(x)
```

---

## 📚 Обзор модулей

### Основные модули

| Модуль | Описание |
|--------|----------|
| `kokao.core` | Двухканальное ядро (S⁺/S⁻) |
| `kokao.inverse` | Обратная задача (S → x) |
| `kokao.decoder` | Обёртка декодера |
| `kokao.math_exact` | Точные методы (SVD, спектральные) |

### Когнитивные модули

| Модуль | Описание |
|--------|----------|
| `kokao.etalon` | Интуитивная эталонная система (Глава 2) |
| `kokao.normal_etalon` | Левое/правое полушарие (Глава 3) |
| `kokao.goal_system` | Цели, удовольствие, депривация (Глава 4) |

### Интеграции

| Модуль | Описание |
|--------|----------|
| `kokao.rag` | Поиск на основе FAISS |
| `kokao.xai` | Объяснения SHAP/LIME |
| `kokao.integrations.langchain` | Инструменты LangChain |
| `kokao.integrations.huggingface` | Интеграция с HF Hub |

### Экспериментальные

| Модуль | Описание |
|--------|----------|
| `kokao.experimental.topological` | Проверка фундаментального диапазона, нормализация на сферу |
| `kokao.experimental.physical` | Физически вдохновлённые компоненты |

---

## 🧪 Запуск тестов

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

---

## 📄 Лицензия

Лицензия Apache 2.0

Основано на идеях из книги Ю.Б. Косякова "Мой мозг" (1999).

Математический метод находится в общественном достоянии (патент РФ №2109332 утратил силу).

---

## 🤝 Вклад

1. Форкнуть репозиторий
2. Создать ветку (`git checkout -b feature/AmazingFeature`)
3. Закоммитить изменения (`git commit -m 'Add AmazingFeature'`)
4. Отправить в ветку (`git push origin feature/AmazingFeature`)
5. Открыть Pull Request

---

## 📈 Статистика

| Метрика | Значение |
|---------|----------|
| Всего тестов | 579 |
| Покрытие тестов | 94% |
| Модулей | 52 |
| Ускорение батчей | 800× (GPU) |
| Ускорение обратной задачи | 2500× |
| Квантование | INT8/INT4 |

---

*Последнее обновление: март 2026*

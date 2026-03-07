# Архитектура Kokao Engine v2.0.0

## Обзор

Kokao Engine v2.5 (Hybrid)— это реализация двухканального интуитивного ядра по теории Косякова, где сигнал определяется как отношение положительного и отрицательного каналов: `S = S⁺ / S⁻`.

## 📐 Математическая модель

### Основная формула

```
S⁺ = Σ xᵢ wᵢ⁺  (возбуждающий канал)
S⁻ = Σ xᵢ wᵢ⁻  (тормозящий канал)
S = S⁺ / S⁻    (интуитивный сигнал)
```

### Формула 8 (Правило обучения Косякова)

```
Δw = (Δ₀ · x) / Σx²
```

где:
- `Δ₀ = S - target` — ошибка предсказания
- `x` — входной вектор
- `Σx²` — сумма квадратов элементов входа

### Инвариантность к масштабу

```
S(k·x) = S(x) для любого k ≠ 0
```

## 🏗️ Структура проекта

```
kokao-engine-3/
├── kokao/                      # Основной пакет
│   ├── __init__.py             # Публичный интерфейс
│   ├── core_base.py            # Базовые классы и конфигурация
│   ├── core.py                 # Основное ядро KokaoCore
│   ├── inverse.py              # Обратная задача (S → x)
│   ├── decoder.py              # Обертка Decoder
│   ├── config.py               # Конфигурация (совместимость)
│   ├── base.py                 # Базовые классы (совместимость)
│   ├── secure.py               # Модуль безопасности
│   ├── cli.py                  # CLI интерфейс
│   ├── rag.py                  # RAG модуль (FAISS)
│   ├── xai.py                  # XAI модуль (SHAP/LIME)
│   └── integrations/
│       ├── __init__.py
│       ├── langchain.py        # LangChain инструменты
│       └── huggingface.py      # HuggingFace Hub интеграция
├── tests/                      # Модульные тесты
├── examples/
│   └── demo.py                 # Демонстрационный пример
├── test_*.py                   # Интеграционные тесты
├── run_all_tests.py            # Скрипт запуска всех тестов
├── pyproject.toml              # Конфигурация проекта
├── requirements.txt            # Зависимости
├── README.md                   # Основная документация
└── ARCHITECTURE.md             # Этот файл
```

## 🔧 Основные компоненты

### CoreConfig

Класс конфигурации ядра:

```python
class CoreConfig(BaseModel):
    input_dim: int = 10           # Размерность входа
    device: str = "cpu"           # Устройство вычислений
    dtype: str = "float32"        # Тип данных
    target_sum: float = 100.0     # Целевая сумма весов
    max_history: int = 100        # Размер истории весов
    use_log_domain: bool = False  # Логарифмический домен
```

### KokaoCore

Основной класс ядра, наследуется от `torch.nn.Module`:

```python
class KokaoCore(torch.nn.Module, KokaoCoreBase):
    """Двухканальное интуитивное ядро."""
    
    # Основные методы:
    def signal(x) -> float                    # Вычисление сигнала
    def train(x, target, lr, mode) -> float   # Обучение
    def train_adam(x, target) -> float        # Adam обучение
    def train_batch(X, targets, lr) -> float  # Батчевое обучение
    def forget(rate, lambda_l1)               # Забывание
    def to_inverse_problem() -> InverseProblem # Обратная задача
    def quantize_int8() -> KokaoCore          # INT8 квантование
    def quantize_int4() -> KokaoCore          # INT4 квантование
```

### Внутренняя структура KokaoCore

```
KokaoCore
├── w_plus (Parameter)        # Внутренние параметры канала S⁺
├── w_minus (Parameter)       # Внутренние параметры канала S⁻
├── optimizer (Adam)          # Оптимизатор Adam
├── config (CoreConfig)       # Конфигурация
├── history (deque)           # История весов
├── version (int)             # Версия модели
├── is_quantized (bool)       # Флаг квантования
└── quantized_model           # Квантованная модель
```

### Эффективные веса

Веса всегда положительные благодаря функции `softplus`:

```python
def _get_effective_weights() -> Tuple[Tensor, Tensor]:
    """Возвращает гарантированно положительные веса."""
    eff_w_plus = softplus(w_plus)   # log(1 + exp(w_plus))
    eff_w_minus = softplus(w_minus) # log(1 + exp(w_minus))
    return eff_w_plus, eff_w_minus
```

### Нормализация

Сумма эффективных весов каждого канала равна `target_sum`:

```python
def _normalize() -> None:
    """Нормализация: Σ eff_w_plus = Σ eff_w_minus = target_sum"""
    # 1. Вычисляем текущие суммы эффективных весов
    total_p = eff_w_plus.sum()
    total_m = eff_w_minus.sum()
    
    # 2. Масштабируем к целевой сумме
    target_eff_plus = eff_w_plus * (target / total_p)
    target_eff_minus = eff_w_minus * (target / total_m)
    
    # 3. Обновляем внутренние параметры
    w_plus = log(expm1(target_eff_plus) + ε)
    w_minus = log(expm1(target_eff_minus) + ε)
```

## 📊 Алгоритмы обучения

### 1. Градиентный метод

```python
def train_gradient(x, target, lr):
    s = forward(x)
    loss = (s - target) ** 2
    loss.backward()
    
    with torch.no_grad():
        w_plus -= lr * w_plus.grad
        w_minus -= lr * w_minus.grad
        _normalize()
    
    return loss.item()
```

### 2. Метод Косякова

```python
def train_kosyakov(x, target, lr):
    s = forward(x)
    delta_0 = s - target
    sum_x_sq = torch.dot(x, x)
    
    delta_w = (delta_0 / sum_x_sq) * x
    
    with torch.no_grad():
        w_plus -= delta_w * lr * 0.5
        w_minus += delta_w * lr * 0.5
        _normalize()
    
    return loss.item()
```

### 3. Adam оптимизатор

```python
def train_adam(x, target):
    optimizer.zero_grad()
    s = forward(x)
    loss = (s - target) ** 2
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        _normalize()
    
    return loss.item()
```

### 4. Батчевое обучение

```python
def train_batch(X, targets, lr):
    # X: (Batch, input_dim)
    # targets: (Batch,)
    S = forward(X)  # (Batch,)
    loss = ((S - targets) ** 2).mean()
    loss.backward()
    
    with torch.no_grad():
        w_plus -= lr * w_plus.grad
        w_minus -= lr * w_minus.grad
        _normalize()
    
    return loss.item()
```

## 🔄 Обратная задача (S → x)

Класс `InverseProblem` решает уравнение `S(x) = S_target`:

```python
class InverseProblem:
    def __init__(self, w_plus, w_minus):
        self.w_plus = w_plus.clone()
        self.w_minus = w_minus.clone()
    
    def solve(S_target, x_init, lr, max_steps, clamp_range):
        x = x_init if x_init else torch.randn(input_dim)
        x.requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            s = compute_S(x)
            loss = (s - S_target) ** 2
            loss.backward()
            optimizer.step()
            x.data.clamp_(*clamp_range)
        
        return x.detach()
```

### Decoder

Обертка для удобного использования:

```python
class Decoder:
    def __init__(self, core, lr=0.1, max_steps=100):
        self.core = core
        self.lr = lr
        self.max_steps = max_steps
    
    def generate(S_target):
        inverse = self.core.to_inverse_problem()
        return inverse.solve(S_target, lr=self.lr, max_steps=self.max_steps)
```

## 🛡️ Модуль безопасности

`SecureKokao` — обертка для проверки входных данных:

```python
class SecureKokao:
    """Проверяет все входные тензоры перед передачей в ядро."""
    
    @validate_tensor_input
    def signal(self, x):
        # Проверяет: тип, NaN, Inf, размерность
        return self._core.signal(x)
```

### Декоратор валидации

```python
def validate_tensor_input(func):
    def wrapper(self, x, *args, **kwargs):
        assert isinstance(x, torch.Tensor)
        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()
        assert x.shape[-1] == self.config.input_dim
        return func(self, x, *args, **kwargs)
    return wrapper
```

## 🧩 Интеграции

### RAG (Retrieval-Augmented Generation)

Модуль для поиска документов по эмбеддингам:

```python
class RAGModule:
    def __init__(self, core, embedding_dim):
        self.core = core
        self.index = faiss.IndexFlatL2(embedding_dim)
    
    def add_document(doc_id, embedding, metadata):
        self.index.add(embedding)
        self.id_map[doc_id] = metadata
    
    def search(query_embedding, k):
        distances, indices = self.index.search(query_embedding, k)
        return [(doc_ids[i], distances[i]) for i in range(k)]
    
    def search_by_signal_similarity(query_embedding, k):
        # Ранжирование по разнице сигналов KokaoCore
        ...
```

### XAI (Explainable AI)

Модуль объяснения решений:

```python
class XAIAnalyzer:
    def __init__(self, core):
        self.core = core
        self._predict_fn = lambda x: self._predict_batch(x)
    
    def shap_explain(x, method):
        explainer = shap.Explainer(self._predict_fn, ...)
        return explainer(x).values
    
    def lime_explain(x, num_samples, kernel_width):
        explainer = lime.lime_tabular.LimeTabularExplainer(...)
        return explainer.explain_instance(x, predict_fn, ...)
    
    def analyze_feature_importance(x):
        return {
            'shap_values': self.shap_explain(x),
            'lime_coeffs': self.lime_explain(x),
            'effective_weights_plus': ...,
            'effective_weights_minus': ...,
            'approx_feature_contributions': ...
        }
```

### LangChain

Инструменты для LangChain агентов:

```python
class KokaoSignalTool(BaseTool):
    name = "kokao_signal_calculator"
    description = "Вычисляет сигнал KokaoCore"
    
    def _run(vector_json):
        vector = torch.tensor(json.loads(vector_json))
        signal = self.core.signal(vector)
        return f"Signal: {signal:.6f}"

class KokaoInversionTool(BaseTool):
    name = "kokao_signal_inverter"
    description = "Генерирует вектор по сигналу"
    
    def _run(target_signal):
        generated = self.decoder.generate(float(target_signal))
        return f"Vector: {generated.tolist()}"
```

### HuggingFace Hub

Публикация и загрузка моделей:

```python
class HFModelManager:
    def push_model(core, repo_id, filename):
        state = {
            "w_plus": core.w_plus.tolist(),
            "w_minus": core.w_minus.tolist(),
            "config": core.config.dict(),
            "version": core.version
        }
        upload_file(state, repo_id, filename)
    
    def pull_model(repo_id, filename):
        state = download_file(repo_id, filename)
        core = KokaoCore(CoreConfig(**state["config"]))
        core.w_plus = torch.tensor(state["w_plus"])
        core.w_minus = torch.tensor(state["w_minus"])
        return core
```

## ⚡ Квантование

### INT8 квантование

```python
def quantize_int8(self):
    self.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        self, {torch.nn.Parameter}, dtype=torch.qint8
    )
    quantized_core = KokaoCore(self.config)
    quantized_core.is_quantized = True
    quantized_core.quantized_model = quantized_model
    return quantized_core
```

### INT4 квантование

```python
def quantize_int4(self):
    self.eval()
    qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    quantized_core = KokaoCore(self.config)
    torch.ao.quantization.prepare(quantized_core, inplace=True)
    torch.ao.quantization.convert(quantized_core, inplace=True)
    quantized_core.is_quantized = True
    return quantized_core
```

## 🔒 Потокобезопасность

Все методы ядра используют `threading.RLock`:

```python
def train(self, x, target, lr, mode):
    with self._lock:  # RLock для рекурсивной блокировки
        # ... код обучения
```

## 📈 История версий

### v2.0.0

- ✅ Двухканальная архитектура S = S⁺/S⁻
- ✅ Гарантированно положительные веса (softplus)
- ✅ Нормализация весов к target_sum
- ✅ Батчевое обучение
- ✅ Adam оптимизатор
- ✅ Обратная задача (S → x)
- ✅ RAG модуль (FAISS)
- ✅ XAI модуль (SHAP/LIME)
- ✅ LangChain интеграция
- ✅ HuggingFace Hub интеграция
- ✅ CLI инструменты
- ✅ Квантование INT8/INT4
- ✅ Модуль безопасности

## 🎯 Планы развития

| Модуль | Приоритет | Описание |
|--------|-----------|----------|
| GNN | 🟢 Низкий | Graph Neural Networks (GCN, GAT) |
| MLflow Registry | 🟡 Средний | Управление жизненным циклом моделей |
| ONNX/TensorRT | 🟡 Средний | Экспорт для production |
| DVC | 🟡 Средний | Версионирование данных |
| FlashAttention | 🟡 Средний | Ускорение трансформеров |
| DeepSpeed/Horovod | 🟡 Средний | Распределённое обучение |
| CUDA Graphs | 🟡 Средний | Оптимизация повторяющихся вызовов |
| DP-SGD | 🟡 Средний | Дифференциальная приватность |

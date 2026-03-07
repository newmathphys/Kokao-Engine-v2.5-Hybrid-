# Changelog

All notable changes to Kokao Engine will be documented in this file.

---

## [3.0.0] - 2026-03-04

### Added

#### Analytical Inverse Problem / Аналитическая обратная задача
- **FastInverse** - мгновенная обратная задача (~0.17 мс вместо ~400 мс)
- **Аналитическая проекция** на гиперплоскость v·x = 0
- **Ускорение в 2500×** при ошибке < 0.001 для target ≤ 10

#### Physical Module / Физический модуль
- **kokao.experimental.physical** - пакет для физических интерпретаций
- **PhysicalCore** - расширенное ядро с физическими эффектами
- **PhysicalInverse** - обратная задача с физической проверкой диапазона
- **Физические константы** - K (1838.684), S³ (2π²), ALPHA (1/137.036), A0 (боровский радиус)
- **Изоспиновые режимы** - проекция весов на 3 или 4 уровня (+3/+4)
- **Солитонная динамика** - потенциал Синус-Гордона, кинк-решение, нелинейная активация
- **Квантование по моде 93** - топологическое квантование с вычислением заряда
- **Лоренц-фактор** - релятивистские поправки к сигналу

#### Topological Module / Топологический модуль
- **kokao.experimental.topological** - топологические методы
- **K = 1838.684** - фундаментальная константа (отношение масс нейтрона/электрона)
- **check_fundamental_range** - проверка диапазона [1/K, K]
- **normalize_to_sphere** - нормализация на единичную сферу
- **TopologicalInverse** - обратная задача с топологическими проверками

#### Benchmarks / Бенчмарки
- **benchmarks/benchmark_methods.py** - сравнение производительности методов
- **tests/test_experimental/test_compare_methods.py** - 12 параметризованных тестов сравнения

#### Testing / Тестирование
- **579 тестов** покрывают все модули
- **94% покрытие** кода
- **41 экспериментальный тест** с маркером `@pytest.mark.experimental`

### Changed

#### Performance / Производительность
- **Физический метод быстрее в 2500–3000 раз** стандартного Decoder
- Стандартный Decoder: 300–500 мс, физический: 0.16–0.3 мс
- **Ошибка физического метода:** < 0.001 для target ≤ 10

#### API / API
- Добавлен экспорт экспериментальных модулей в `kokao/__init__.py`
- Прямой импорт: `from kokao import PhysicalCore, PhysicalInverse, K`

#### Documentation / Документация
- Полная документация на английском и русском
- RELEASE_NOTES.md и RELEASE_NOTES_RU.md
- README_GITHUB.md (v2.0) и README_PYPI.md (v3.0)

### Fixed

#### PhysicalCore / Физическое ядро
- Исправлена инициализация атрибутов до вызова super().__init__()
- Изоспиновая проекция применяется только в режиме инференса (training=False)
- Исправлена форма лоренц-фактора для 1D входа

#### Tests / Тесты
- Исправлены допуски в test_parameterized.py для отрицательных target
- Исправлена проверка версии в test_infrastructure.py
- Исправлены тесты verbose режимов в test_inverse_accuracy.py

### Version Info / Информация о версии

| Platform | Version |
|----------|---------|
| **PyPI** | 3.0.0 |
| **GitHub** | 2.0.4 |

---

## [2.0.4] - 2026-03-04

### Added

#### Benchmarks / Бенчмарки
- **benchmarks/benchmark_methods.py** - Сравнение производительности методов обратной задачи
- **tests/test_experimental/test_compare_methods.py** - Параметризованные тесты для 12 целевых сигналов

#### Experimental API Export / Экспорт экспериментального API
- Добавлен экспорт экспериментальных модулей в `kokao/__init__.py`
- Прямой импорт: `from kokao import PhysicalCore, PhysicalInverse, K`

### Changed

#### Performance / Производительность
- **Физический метод быстрее в 2500–3000 раз** стандартного Decoder
- Стандартный Decoder: 300–500 мс, физический: 0.16–0.3 мс

#### Accuracy / Точность
- Физический метод: ошибка < 0.001 для target ≤ 10
- Физический метод: ошибка < 0.1 для target ≤ 100
- Стандартный Decoder: ошибка ~9–99 для больших target

### Fixed

#### PhysicalCore / Физическое ядро
- Исправлена инициализация атрибутов до вызова super().__init__()
- Изоспиновая проекция применяется только в режиме инференса (training=False)
- Исправлена форма лоренц-фактора для 1D входа

---

## [2.0.3] - 2026-03-04

### Added

#### Physical Module / Физический модуль
- **kokao.experimental.physical** - Новый пакет для физических интерпретаций
- **PhysicalCore** - Расширенное ядро с физическими эффектами
- **PhysicalInverse** - Обратная задача с физической проверкой диапазона
- **Физические константы** - K (1838.684), S³ (2π²), ALPHA (1/137.036), A0 (боровский радиус)
- **Изоспиновые режимы** - Проекция весов на 3 или 4 уровня (+3/+4)
- **Солитонная динамика** - Потенциал Синус-Гордона, кинк-решение, нелинейная активация
- **Квантование по моде 93** - Топологическое квантование с вычислением заряда
- **Лоренц-фактор** - Релятивистские поправки к сигналу

#### Tests / Тесты
- **25 tests** для физического модуля с маркером `@pytest.mark.experimental`

### Changed

#### Documentation / Документация
- Добавлена документация физического модуля в README.md
- Обновлена структура проекта в README.md
- Обновлена статистика: 571 тест, 52 модуля

---

## [2.0.2] - 2026-03-04

### Added

#### Experimental Modules / Экспериментальные модули
- **kokao.experimental** - New package for experimental features
- **TopologicalInverse** - Inverse problem with topological checks
- **K** - Fundamental constant (1838.684, neutron/electron mass ratio)
- **check_fundamental_range** - Signal range validation [1/K, K]
- **normalize_to_sphere** - Unit sphere normalization
- **Experimental tests** - 4 tests with `@pytest.mark.experimental` marker

### Changed

#### Testing / Тестирование
- Added `experimental` pytest marker for experimental tests
- Experimental tests can be excluded with `-m "not experimental"`

#### Documentation / Документация
- Added experimental module documentation to README.md
- Updated project structure in README.md

---

## [2.0] - 2026-03-04

### Added

#### Core / Ядро
- **Two-channel KokaoCore** - Signal as ratio S = S⁺/S⁻ (scale invariant)
- **InverseProblem** - Generate input vector x for target signal S_target
- **Decoder** - Wrapper for inverse problem with convenient defaults
- **MathExactCore** - Exact mathematical methods (SVD, pseudo-inverse, spectral analysis)

#### Cognitive Modules / Когнитивные модули
- **IntuitiveEtalonSystem** - Pattern recognition with fuzzy prototypes (Chapter 2)
- **NormalIntuitiveEtalonSystem** - Left/right hemisphere separation (Chapter 3)
- **SelfPlanningSystem** - Goals, pleasure, deprivation, fatigue (Chapter 4)

#### Security / Безопасность
- **SecureKokao** - Input validation wrapper
- **validate_tensor_input** - Decorator for NaN/Inf/dimension checks

#### Integrations / Интеграции
- **RAGModule** - FAISS-based document retrieval
- **XAIAnalyzer** - SHAP/LIME explanations
- **LangChainKokaoAdapter** - LangChain tools
- **HFModelManager** - HuggingFace Hub integration

### Changed

#### Improvements / Улучшения
- **InverseProblem**: Adaptive LR (ReduceLROnPlateau), L-BFGS optimizer, multi-restart (5 attempts)
- **L1 + L2 regularization**: Enhanced regularization (0.05 + 0.01)
- **Smart initialization**: Weight-based initialization for inverse problem
- **max_steps**: Increased 200→500 (base), 500→1000 (extreme)
- **num_restarts**: Increased 1→5

#### Documentation / Документация
- Bilingual documentation (EN/RU)
- Restructured README for GitHub
- Updated ARCHITECTURE.md
- Updated PUBLISH_GUIDE.md

### Fixed

#### Critical / Критические
- **forward()**: Replaced sign() with tanh() approximation for differentiability
- **_normalize()**: Removed abs() that was hiding errors, added negative weight checks
- **train_batch()**: Added device check, automatic data movement
- **amp.py**: Updated to torch.amp API (from deprecated torch.cuda.amp)
- **metrics.py**: Fixed Prometheus port conflicts (auto-select free port)

#### Tests / Тесты
- **542 tests** total (94.1% coverage)
- **163 parameterized tests** added
- **27 tests** for math_exact module

### Performance / Производительность

| Metric | Value |
|--------|-------|
| Inference (batch=512) | 3.8M vectors/sec |
| Training (dim=100, bs=64) | 0.55 ms/step |
| Batch speedup | 74.8x |
| Weight stability | 0.02% change, 0 NaN/Inf |

### Known Limitations / Известные ограничения

#### Inverse Problem / Обратная задача
- **Success rate**: 20-33% for target error <0.01
- **Cause**: Fundamental architecture limitation (S⁺/S⁻ with positive weights via softplus)
- **Negative signals**: Unachievable (requires S⁻ < 0, impossible with softplus)
- **Loss landscape**: Non-convex, gradient descent gets stuck in local minima

#### Recommendations for v3.0 / Рекомендации для v3.0
1. Change architecture to support negative weights (remove softplus)
2. Use global optimization (genetic algorithms, simulated annealing)
3. Pre-trained model for initialization
4. Ensemble of multiple optimization methods

---

## [1.0] - 2023-XX-XX

### Added
- Initial Kokao Engine release
- Basic core implementation
- Initial tests

---

*For more details, see ARCHITECTURE.md and README.md*

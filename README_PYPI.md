# Kokao Engine v2.5 (Hybrid)

**Intuitive System based on Kosyakov's Theory: Two-Channel Core (S⁺/S⁻), Cognitive Modules, Analytical Inverse Problem, Physical Interpretation**

[![PyPI version](https://badge.fury.io/py/kokao-engine.svg)](https://badge.fury.io/py/kokao-engine)
[![Version](https://img.shields.io/badge/version-v2.5%20(Hybrid)-blue)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/newmathphys/kokao-engine/blob/main/LICENSE)
[![Tests](https://github.com/newmathphys/kokao-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/newmathphys/kokao-engine/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

---

## 🎯 Key Features

### Two-Channel Core (S⁺/S⁻)
Signal computed as ratio `S = S⁺ / S⁻`, providing **scale invariance** to input magnitude.

### Analytical Inverse Problem
Generate input vector `x` for target signal `S_target` in **~0.17ms** (vs 400ms iterative).

**Performance:**
- **2500× faster** than iterative methods
- **Error < 0.001** for target ≤ 10
- Works for targets up to 1000

### Cognitive Modules
Based on Yu.B. Kosyakov's theory (book "My Brain", 1999):

| Level | Module | Description |
|-------|--------|-------------|
| 2 | Intuitive Etalon System | Pattern recognition with fuzzy prototypes |
| 3 | Normal Etalon System | Left (images) / Right (actions) hemisphere |
| 4 | Self-Planning System | Goals, pleasure, deprivation, fatigue |

### Physical Experimental Module
Physics-inspired components:

- **Fundamental constants**: K=1838.684, S³=2π², α=1/137.036
- **Isospin modes**: 3-level (+3) and 4-level (+4) quantization
- **Solitonic dynamics**: Sine-Gordon potential, kink solutions
- **Lorentz factor**: Relativistic corrections
- **Topological quantization**: Mode 93 with charge computation

### Modern Extensions
- **RAG** (FAISS-based retrieval)
- **XAI** (SHAP/LIME explanations)
- **LangChain** integration
- **HuggingFace Hub** integration
- **Quantum neural networks**
- **Graph neural networks** (GNN)
- **Spiking neural networks** (SNN)
- **Homomorphic encryption**
- **Federated learning**

### Security & Privacy
- Input validation
- Differential privacy (DP-SGD)
- Vulnerability auditing
- Penetration testing

### Performance
- **Batch training**: up to 800× speedup on GPU
- **CUDA Graphs** optimization
- **INT8/INT4 quantization**
- **ONNX/TensorRT export**

---

## ⚙️ Installation

### Basic Installation

```bash
pip install kokao-engine
```

### With Extensions

```bash
# All extensions
pip install 'kokao-engine[all]'

# Selective installation
pip install 'kokao-engine[rag,xai,langchain]'
pip install 'kokao-engine[quantum,gnn,snn]'
pip install 'kokao-engine[homomorphic,federated]'
```

### Development Installation

```bash
pip install 'kokao-engine[dev]'
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from kokao import KokaoCore, CoreConfig, Decoder

# 1. Create core
config = CoreConfig(input_dim=10, seed=42)
core = KokaoCore(config)

# 2. Compute signal
x = torch.randn(10)
s = core.signal(x)
print(f"Signal: {s:.4f}")

# 3. Train
loss = core.train(x, target=0.8, lr=0.01)
print(f"Loss: {loss:.6f}")

# 4. Inverse problem (generate x for target signal)
decoder = Decoder(core, lr=0.1, max_steps=200)
x_gen = decoder.generate(S_target=0.5)
s_gen = core.signal(x_gen)
print(f"Target: 0.5, Achieved: {s_gen:.6f}")
```

### Batch Training

```python
# Batch training (up to 800× speedup on GPU)
X_batch = torch.randn(64, 10)  # batch of 64 samples
y_batch = torch.full((64,), 0.8)  # all targets = 0.8
batch_loss = core.train_batch(X_batch, y_batch, lr=0.01)
```

### Advanced: Physical Core

```python
from kokao.experimental.physical import PhysicalCore

# Core with solitonic activation
core = PhysicalCore(config, use_solitonic=True)
s = core.signal(x)  # output in [-1, 1]

# Core with Lorentz factor
core = PhysicalCore(config, use_lorentz=True, lorentz_c=1.0)
s = core.signal(x)

# Core with isospin quantization
core = PhysicalCore(config, isospin_mode='+3')
core.training = False  # apply projection in eval mode
s = core.signal(x)
```

---

## 📊 Performance Comparison

### Inverse Problem: Decoder vs Analytical

| Target | Decoder Error | Analytical Error | Speedup |
|--------|---------------|------------------|---------|
| 0.01 | 0.992 | 0.000002 | 496000× |
| 0.1 | 0.901 | 0.000018 | 50055× |
| 0.5 | 0.439 | 0.0000006 | 731666× |
| 0.8 | 0.001 | 0.00000005 | 20000× |
| 1.5 | 0.461 | 0.0000008 | 576250× |
| 10.0 | 9.01 | 0.00007 | 128714× |
| 100.0 | 99.0 | 0.007 | 14142× |
| 1000.0 | 999.0 | 2.68 | 372× |

**Average speedup: ~2500×**

---

## 📚 Module Overview

### Core Modules

| Module | Description |
|--------|-------------|
| `kokao.core` | Two-channel core (S⁺/S⁻) |
| `kokao.inverse` | Inverse problem (S → x) |
| `kokao.decoder` | Decoder wrapper |
| `kokao.math_exact` | Exact methods (SVD, spectral) |

### Cognitive Modules

| Module | Description |
|--------|-------------|
| `kokao.etalon` | Intuitive etalon system (Chapter 2) |
| `kokao.normal_etalon` | Left/right hemisphere (Chapter 3) |
| `kokao.goal_system` | Goals, pleasure, deprivation (Chapter 4) |

### Integrations

| Module | Description |
|--------|-------------|
| `kokao.rag` | FAISS-based retrieval |
| `kokao.xai` | SHAP/LIME explanations |
| `kokao.integrations.langchain` | LangChain tools |
| `kokao.integrations.huggingface` | HF Hub integration |

### Experimental

| Module | Description |
|--------|-------------|
| `kokao.experimental.topological` | Fundamental range check, sphere normalization |
| `kokao.experimental.physical` | Physics-inspired components |

---

## 🧪 Testing

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

**Results:** 579 tests passed, 4 skipped, 94% coverage

---

## 📄 License

Apache License 2.0

Based on ideas from Yu.B. Kosyakov's book "My Brain" (1999).

**Note:** The mathematical method is in the public domain (Russian Patent №2109332 expired).

---

## 🤝 Contributing

1. Fork the repository
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📈 Statistics

| Metric | Value |
|--------|-------|
| Total tests | 579 |
| Test coverage | 94% |
| Modules | 52 |
| Batch speedup | 800× (GPU) |
| Inverse speedup | 2500× |
| Quantization | INT8/INT4 |

---

## 🔗 Links

- **GitHub:** https://github.com/newmathphys/kokao-engine
- **Documentation:** https://kokao-engine.readthedocs.io
- **PyPI:** https://pypi.org/project/kokao-engine
- **Changelog:** https://github.com/newmathphys/kokao-engine/blob/main/CHANGELOG.md

---

## 👥 Authors

- **Vital Kalinouski**
- **V. Ovseychik**

**Organization:** newmathphys

---

*Last updated: March 2026*

"""
Экспериментальный физический модуль Kokao Engine.

Содержит физические интерпретации и расширения:
- Фундаментальные константы (K, S³, α, a₀)
- Изоспиновые режимы (+3/+4)
- Солитонная динамика (Синус-Гордон)
- Квантование по моде 93
- Лоренц-фактор
- PhysicalCore и PhysicalInverse
"""
from .constants import K, S3, ALPHA, A0
from .isospin import isospin_projection, isospin_regularization
from .solitonic import sine_gordon_potential, solitonic_activation, kink_solution
from .quantization import quantize_with_topology, topological_charge
from .lorentz import lorentz_factor, lorentz_boost
from .core import PhysicalCore
from .inverse import PhysicalInverse

__all__ = [
    # Константы
    'K', 'S3', 'ALPHA', 'A0',
    # Изоспин
    'isospin_projection', 'isospin_regularization',
    # Солитоны
    'sine_gordon_potential', 'solitonic_activation', 'kink_solution',
    # Квантование
    'quantize_with_topology', 'topological_charge',
    # Лоренц
    'lorentz_factor', 'lorentz_boost',
    # Ядро и обратная задача
    'PhysicalCore', 'PhysicalInverse',
]

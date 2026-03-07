"""
Kokao Engine v2.5 (Hybrid) - Intuitive System based on Kosyakov's Theory.

Two-channel core (S⁺/S⁻), cognitive modules, analytical inverse problem, physical interpretation.
Двухканальное ядро (S⁺/S⁻), когнитивные модули, аналитическая обратная задача, физическая интерпретация.

GitHub: v2.5 (Hybrid with experimental modules)
PyPI: v2.5 (Hybrid release)
"""

__version__ = "2.5.0"
__author__ = "Vital Kalinouski / Виталий Калиновский, V. Ovseychik / В. Овсейчик"
__email__ = "newmathphys@gmail.com"

# =============================================================================
# GLOBAL DEBUG FLAG
# =============================================================================
DEBUG = False


def set_debug(enable: bool):
    """
    Включить/выключить отладочный вывод для всех модулей.

    Args:
        enable: True для включения отладки
    """
    global DEBUG
    DEBUG = enable


# =============================================================================
# CORE MODULES
# =============================================================================
from .core import KokaoCore, KokaoCoreInference, to_inference
from .core_base import CoreConfig, KokaoCoreBase
from .inverse import InverseProblem
from .decoder import Decoder
from .config import CoreConfig as ConfigAlias
from .base import KokaoCoreBase as BaseAlias

# =============================================================================
# ETALON SYSTEMS (Chapters 2, 3, 4)
# =============================================================================
from .etalon import IntuitiveEtalonSystem
from .normal_etalon import NormalIntuitiveEtalonSystem
from .goal_system import SelfPlanningSystem

# =============================================================================
# SECURITY & SAFETY
# =============================================================================
from .secure import SecureKokao, validate_tensor_input

# =============================================================================
# INTEGRATIONS
# =============================================================================
from .integrations import langchain, huggingface
from .integrations.langchain import (
    KokaoSignalTool,
    KokaoInversionTool,
    KokaoTrainTool,
    LangChainKokaoAdapter
)
from .integrations.huggingface import HFModelManager

# =============================================================================
# AI/ML MODULES
# =============================================================================
from .rag import RAGModule
from .xai import XAIAnalyzer

# =============================================================================
# TIME SERIES
# =============================================================================
from .timeseries import (
    TimeSeriesPredictor,
    TimeSeriesDataset,
    create_seasonal_features
)

# =============================================================================
# AGENTIC AI
# =============================================================================
from .agentic import (
    KokaoAgent,
    MultiAgentSystem,
    AgentState,
    AgentMemory
)

# =============================================================================
# FEDERATED LEARNING
# =============================================================================
from .federated import (
    FederatedClient,
    FederatedServer,
    FederatedLearning,
    ClientConfig
)

# =============================================================================
# GENERATIVE MODELS
# =============================================================================
from .generative import KokaoGAN, KokaoVAE

# =============================================================================
# QUANTUM NEURAL NETWORKS
# =============================================================================
from .quantum import (
    KokaoQuantumNetwork,
    QuantumCircuit,
    QuantumState,
    HadamardGate,
    PhaseGate,
    CNOTGate,
    create_bell_state,
    create_ghz_state
)

# =============================================================================
# GRAPH NEURAL NETWORKS
# =============================================================================
from .gnn import (
    KokaoGNN,
    Graph,
    GraphConvolution,
    GraphAttentionLayer,
    GraphDataset,
    create_random_graph,
    karate_club_graph
)

# =============================================================================
# SPIKING NEURAL NETWORKS
# =============================================================================
from .snn import (
    KokaoSNN,
    LeakyIntegrateAndFire,
    SpikeTrain,
    SpikeEncoding,
    STDPPlasticity
)

# =============================================================================
# PRIVACY & SECURITY
# =============================================================================
from .privacy import (
    DPSGD,
    GaussianMechanism,
    MomentsAccountant,
    PrivacyBudget,
    add_dp_noise_to_weights,
    compute_privacy_budget
)

from .homomorphic import (
    HomomorphicKokao,
    PaillierCipher,
    EncryptedTensor,
    SecureMultiPartyComputation,
    create_secure_model,
    benchmark_encryption
)

# =============================================================================
# VULNERABILITY & PENETRATION TESTING
# =============================================================================
from .vulnerability_audit import (
    VulnerabilityAuditor,
    VulnerabilityReport,
    audit_model
)

from .penetration_testing import (
    PenetrationTester,
    AdversarialAttack,
    FGSMAttack,
    PGDAttack,
    AttackType,
    AttackResult,
    run_quick_penetration_test
)

from .threat import (
    ThreatDetector,
    ThreatIntelligence,
    Threat,
    ThreatLevel,
    ThreatType,
    ThreatReport,
    create_threat_detector
)

# =============================================================================
# UTILITIES
# =============================================================================
from .quantization import (
    QuantizationConfig,
    QuantizedKokaoCore,
    quantize_model,
    benchmark_quantization
)

from .export import (
    ModelExporter,
    export_model,
    load_exported_model,
    KokaoExportWrapper
)

from .mlflow_logging import (
    MLflowLogger,
    ExperimentTracker,
    create_mlflow_logger,
    create_experiment_tracker
)

from .distributed import (
    DistributedKokaoTrainer,
    DistributedConfig,
    DataParallelTrainer,
    RayDistributedTrainer,
    train_distributed,
    create_distributed_trainer
)

from .cuda_graphs import (
    CUDAGraphWrapper,
    TrainingCUDAGraph,
    BatchedCUDAGraph,
    CUDAGraphManager,
    enable_cuda_graphs,
    benchmark_cuda_graphs
)

# =============================================================================
# KOKAO HUB
# =============================================================================
from .kokao_hub import (
    KokaoHub,
    ModelInfo,
    ModelZoo,
    create_hub,
    quick_register
)

# =============================================================================
# MATHEMATICAL EXACT METHODS
# =============================================================================
from .math_exact import (
    MathExactCore,
    MathExactConfig,
    InversionMethod,
    create_math_exact_core,
    solve_inverse_exact
)

# =============================================================================
# EXPERIMENTAL MODULES
# =============================================================================
from . import experimental
from .experimental.topological import (
    K as TOPOLOGICAL_K,
    check_fundamental_range,
    normalize_to_sphere,
    TopologicalInverse
)
from .experimental.physical.constants import K, S3, ALPHA, A0
from .experimental.physical import (
    PhysicalCore,
    PhysicalInverse,
    isospin_projection,
    solitonic_activation,
    quantize_with_topology,
    lorentz_factor
)


# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Version
    '__version__',
    '__author__',
    'DEBUG',
    'set_debug',

    # Core
    'KokaoCore',
    'KokaoCoreInference',
    'to_inference',
    'CoreConfig',
    'KokaoCoreBase',
    'InverseProblem',
    'Decoder',
    'ConfigAlias',
    'BaseAlias',

    # Etalon Systems
    'IntuitiveEtalonSystem',
    'NormalIntuitiveEtalonSystem',
    'SelfPlanningSystem',

    # Security
    'SecureKokao',
    'validate_tensor_input',

    # Integrations
    'langchain',
    'huggingface',
    'KokaoSignalTool',
    'KokaoInversionTool',
    'KokaoTrainTool',
    'LangChainKokaoAdapter',
    'HFModelManager',

    # AI/ML
    'RAGModule',
    'XAIAnalyzer',

    # Time Series
    'TimeSeriesPredictor',
    'TimeSeriesDataset',
    'create_seasonal_features',

    # Agentic
    'KokaoAgent',
    'MultiAgentSystem',
    'AgentState',
    'AgentMemory',

    # Federated
    'FederatedClient',
    'FederatedServer',
    'FederatedLearning',
    'ClientConfig',

    # Generative
    'KokaoGAN',
    'KokaoVAE',

    # Quantum
    'KokaoQuantumNetwork',
    'QuantumCircuit',
    'QuantumState',
    'HadamardGate',
    'PhaseGate',
    'CNOTGate',
    'create_bell_state',
    'create_ghz_state',

    # Graph
    'KokaoGNN',
    'Graph',
    'GraphConvolution',
    'GraphAttentionLayer',
    'GraphDataset',
    'create_random_graph',
    'karate_club_graph',

    # Spiking
    'KokaoSNN',
    'LeakyIntegrateAndFire',
    'SpikeTrain',
    'SpikeEncoding',
    'STDPPlasticity',

    # Privacy
    'DPSGD',
    'GaussianMechanism',
    'MomentsAccountant',
    'PrivacyBudget',
    'add_dp_noise_to_weights',
    'compute_privacy_budget',

    # Homomorphic
    'HomomorphicKokao',
    'PaillierCipher',
    'EncryptedTensor',
    'SecureMultiPartyComputation',
    'create_secure_model',
    'benchmark_encryption',

    # Vulnerability
    'VulnerabilityAuditor',
    'VulnerabilityReport',
    'audit_model',

    # Penetration Testing
    'PenetrationTester',
    'AdversarialAttack',
    'FGSMAttack',
    'PGDAttack',
    'AttackType',
    'AttackResult',
    'run_quick_penetration_test',

    # Threat
    'ThreatDetector',
    'ThreatIntelligence',
    'Threat',
    'ThreatLevel',
    'ThreatType',
    'ThreatReport',
    'create_threat_detector',

    # Utilities
    'QuantizationConfig',
    'QuantizedKokaoCore',
    'quantize_model',
    'benchmark_quantization',

    # Export
    'ModelExporter',
    'export_model',
    'load_exported_model',
    'KokaoExportWrapper',

    # Logging
    'MLflowLogger',
    'ExperimentTracker',
    'create_mlflow_logger',
    'create_experiment_tracker',

    # Distributed
    'DistributedKokaoTrainer',
    'DistributedConfig',
    'DataParallelTrainer',
    'RayDistributedTrainer',
    'train_distributed',
    'create_distributed_trainer',

    # CUDA Graphs
    'CUDAGraphWrapper',
    'TrainingCUDAGraph',
    'BatchedCUDAGraph',
    'CUDAGraphManager',
    'enable_cuda_graphs',
    'benchmark_cuda_graphs',

    # Hub
    'KokaoHub',
    'ModelInfo',
    'ModelZoo',
    'create_hub',
    'quick_register',

    # Mathematical Exact Methods
    'MathExactCore',
    'MathExactConfig',
    'InversionMethod',
    'create_math_exact_core',
    'solve_inverse_exact',

    # Experimental
    'experimental',
    'TOPOLOGICAL_K',
    'check_fundamental_range',
    'normalize_to_sphere',
    'TopologicalInverse',
    'K',  # Physical K
    'S3',
    'ALPHA',
    'A0',
    'PhysicalCore',
    'PhysicalInverse',
    'isospin_projection',
    'solitonic_activation',
    'quantize_with_topology',
    'lorentz_factor',
]

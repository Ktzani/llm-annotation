"""
LLM Annotation System - Sistema de Anotação Automática com Múltiplas LLMs

Este pacote implementa um sistema completo para anotação automática de datasets
usando múltiplas LLMs com análise de consenso.

Instalação:
    poetry install

Uso básico:
    >>> from llm_annotation_system import LLMAnnotator, ConsensusAnalyzer
    >>> 
    >>> annotator = LLMAnnotator(models, categories, api_keys)
    >>> df = annotator.annotate_dataset(texts)
    >>> df = annotator.calculate_consensus(df)

Autor: Gabriel Catizani
Data: Novembro 2025
Versão: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Gabriel Catizani"

from .llm_annotator import LLMAnnotator
from .consensus_analyzer import ConsensusAnalyzer
from .visualizer import ConsensusVisualizer
from ..llm_config.config import (
    LLM_CONFIGS,
    EXPERIMENT_CONFIG,
    BASE_ANNOTATION_PROMPT,
    FEW_SHOT_PROMPT,
    COT_PROMPT,
)

__all__ = [
    "LLMAnnotator",
    "ConsensusAnalyzer",
    "ConsensusVisualizer",
    "LLM_CONFIGS",
    "EXPERIMENT_CONFIG",
    "BASE_ANNOTATION_PROMPT",
    "FEW_SHOT_PROMPT",
    "COT_PROMPT",
]

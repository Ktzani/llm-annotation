# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO BÁSICO
# =============================================================================
from src.config.prompts import BASE_ANNOTATION_PROMPT

"""
Estratégias para resolução de conflitos entre anotações.
"""
CACHE_DIR = "..\..\data\.cache"

# Modelos open-source padrão
DEFAULT_MODELS = [
    "deepseek-r1-8b",
    "qwen2.5-7b",
    "gemma2-9b",
    "mistral-7b",
    "llama3-8b",
]

# =============================================================================
# CONFIGURAÇÃO GLOBAL DE DATASETS
# =============================================================================

# Use 'split': 'train' como padrão (podemos combinar splits se necessário)
# Ajuste 'sample_size' para começar com amostra pequena
# Ajuste 'combine_splits' para combinar múltiplos splits quando necessário. Ex: ["train", "test"]
DATASET_CONFIG = {
    "split": "train",
    "combine_splits": None,
    "sample_size": 100
}


EXPERIMENT_CONFIG = {
    "dataset_config": DATASET_CONFIG,
    
    # Modelos padrão
    "default_models": DEFAULT_MODELS,
    
    # Número de repetições para a mesma LLM
    "num_repetitions_per_llm": 3,
    
    # Threshold de consenso para aceitar anotação automaticamente
    "consensus_threshold": 0.8,  # 80% de acordo
    
    # Estratégia para casos sem consenso
    "no_consensus_strategy": "flag_for_review",
    
    # Métricas de distância a calcular
    "distance_metrics": ["hamming", "jaccard", "cohen_kappa"],
    
    # Salvar resultados intermediários
    "save_intermediate": True,
    
    # Usar cache de respostas
    "cache": {
        "enabled": True,
        "dir": CACHE_DIR
    },
    
    "prompt_template": BASE_ANNOTATION_PROMPT
}

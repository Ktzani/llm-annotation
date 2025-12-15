# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO BÁSICO
# =============================================================================
from src.config.prompts import BASE_ANNOTATION_PROMPT, FEW_SHOT_PROMPT, COT_PROMPT, SIMPLER_PROMPT

CACHE_DIR = "..\..\data\.cache"
RESULTS_DIR = ""

DEFAULT_MODELS = [
    "deepseek-r1-8b",
    "qwen3-8b",
    "gemma3-4b",
    "mistral-7b",
    "llama3.1-8b", 
]

# =============================================================================
# CONFIGURAÇÃO GLOBAL DE DATASETS
# =============================================================================
# Use 'split': 'train' como padrão (podemos combinar splits se necessário)
# Ajuste 'sample_size' para começar com amostra pequena (se desejar). Ex: 1000 ou None para usar todo o split.
# Ajuste 'combine_splits' para combinar múltiplos splits quando necessário. Ex: ["train", "test"]
DATASET_CONFIG = {
    "split": "train",
    "combine_splits": ["train", "test"],
    "sample_size": 500,
    "random_state": 42
}


EXPERIMENT_CONFIG = {
    "dataset_config": DATASET_CONFIG,
    "default_models": DEFAULT_MODELS,
    "prompt_template": BASE_ANNOTATION_PROMPT,
    "num_repetitions_per_llm": 1,
    
    "consensus": {
        "threshold": 0.8,
        "strategy": "majority_vote",
        "no_consensus_strategy": "flag_for_review",
    },
        
    # Salvar resultados intermediários
    "save_intermediate": True,
    
    "cache": {
        "enabled": True,
        "dir": CACHE_DIR
    },

}

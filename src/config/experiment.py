# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO - OPEN SOURCE
# =============================================================================

"""
Estratégias para resolução de conflitos entre anotações.
"""

# Modelos open-source padrão
DEFAULT_MODELS = [
    "llama3-8b",      # Meta Llama 3 8B - Geral
    "mistral-7b",     # Mistral 7B - Rápido
    "qwen2-7b",       # Qwen 2 7B - Excelente PT-BR
]

CONFLICT_RESOLUTION_STRATEGIES = {
    "majority_vote": {
        "description": "Escolhe a classe com mais votos",
        "min_votes_required": 3,
    },
    "weighted_vote": {
        "description": "Voto ponderado baseado na confiança dos modelos",
        "weights": {
            # Open-source models
            "llama3-70b": 1.2,
            "llama3-8b": 1.0,
            "mistral-7b": 1.0,
            "qwen2-7b": 1.1,
            "mixtral-8x7b": 1.2,
        }
    },
    "unanimous_only": {
        "description": "Aceita apenas casos com 100% de consenso",
        "threshold": 1.0,
    },
    "remove_outliers": {
        "description": "Remove classificações muito diferentes",
        "outlier_threshold": 0.3,
    }
}

EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "cohen_kappa",
    "inter_annotator_agreement",
    "krippendorff_alpha",
]

EXPERIMENT_CONFIG = {
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
    "use_cache": True,
}

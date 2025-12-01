CONFLICT_RESOLUTION_STRATEGIES = {
    "majority_vote": {
        "description": "Escolhe a classe com mais votos",
        "min_votes_required": 3,
    },
    "weighted_vote": {
        "description": "Voto ponderado baseado na confiança dos modelos",
        "weights": {
            "deepseek-r1-8b": 1.20,
            "qwen3-8b": 1.25,
            "gemma3-4b": 0.95,
            "mistral-7b": 1.10,
            "llama3.1-8b": 1.30
        }
    },
    "unanimous_only": {
        "description": "Aceita apenas casos com 100% de consenso",
        "threshold": 1.0,
    },
    "remove_outliers": {
        "description": "Remove classificações muito diferentes",
        "outlier_threshold": 0.3,
    },
    "flag_for_review": {
        "description": "Marca para revisão manual quando não há consenso",
    },
}
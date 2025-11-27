CONFLICT_RESOLUTION_STRATEGIES = {
    "majority_vote": {
        "description": "Escolhe a classe com mais votos",
        "min_votes_required": 3,
    },
    "weighted_vote": {
        "description": "Voto ponderado baseado na confiança dos modelos",
        "weights": {
            "llama3-70b": 1.2,
            "llama3-8b": 1.0,
            "mistral-7b": 1.0,
            "qwen2-7b": 1.1,
            "mixtral-8x7b": 1.2,
        }
    },
    "unanimous_only": {
        "description": "Aceita apenas casos com 100% de consenso",
        "threshold": 0.7,
    },
    "remove_outliers": {
        "description": "Remove classificações muito diferentes",
        "outlier_threshold": 0.3,
    }
}
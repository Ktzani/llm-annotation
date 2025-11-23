"""
Estratégias para resolução de conflitos entre anotações.
"""

CONFLICT_RESOLUTION_STRATEGIES = {
    "majority_vote": {
        "description": "Escolhe a classe com mais votos",
        "min_votes_required": 3,
    },
    "weighted_vote": {
        "description": "Voto ponderado baseado na confiança dos modelos",
        "weights": {
            "gpt-4-turbo": 1.2,
            "claude-3-opus": 1.2,
            "gpt-3.5-turbo": 1.0,
            "claude-3-sonnet": 1.0,
            "gemini-pro": 1.0,
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

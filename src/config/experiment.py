# =============================================================================
# CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================

EXPERIMENT_CONFIG = {
    # Número de repetições para a mesma LLM (OBS-2)
    "num_repetitions_per_llm": 3,
    
    # Threshold de consenso para aceitar anotação automaticamente
    "consensus_threshold": 0.8,  # 80% de acordo
    
    # Estratégia para casos sem consenso
    "no_consensus_strategy": "flag_for_review",  # Opções: "flag_for_review", "majority_vote", "remove", "random"
    
    # Métricas de distância a calcular
    "distance_metrics": ["hamming", "jaccard", "cohen_kappa"],
    
    # Salvar resultados intermediários
    "save_intermediate": True,
    
    # Usar cache de respostas (economiza API calls)
    "use_cache": True,
}
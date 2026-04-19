"""
Fine-tuning controlado de RoBERTa (GT vs Consenso LLM)

Este script realiza fine-tuning de modelos RoBERTa comparando resultados
entre ground truth e consenso de anotações LLM.
"""

import sys
from loguru import logger


# Configuração do logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from src.fine_tune_system.pipeline import FineTuningPipeline, FineTuningConfig

def main():
    """Função principal"""
    dataset_name = "dblp"
    model_name = "roberta-base"
    run_type = "single"  # "single" ou "cross-validation"
    
    # Configuração
    config = FineTuningConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        num_epochs=20,
        learning_rate=5e-5,
        train_batch_size=32,
        eval_batch_size=64,
        max_length=256
    )
    
    # Executar pipeline
    pipeline = FineTuningPipeline(config)
    results = pipeline.run(run_type=run_type, max_parallel_folds=1)  # max_parallel_folds=1 para evitar paralelismo em teste local
    
    return results


if __name__ == "__main__":
    main()
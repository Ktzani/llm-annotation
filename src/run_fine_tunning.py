"""
Fine-tuning controlado de RoBERTa (GT vs Consenso LLM)

Este script realiza fine-tuning de modelos RoBERTa comparando resultados
entre ground truth e consenso de anotações LLM. As configurações (incluindo o
bloco `instance_selection`) são carregadas de um JSON de experimento.
"""

import sys
from pathlib import Path
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
    job = "local_job"
    config_path = Path("src/api/experiments") / "fine_tuning" / f"{job}.json"
    if not config_path.exists():
        logger.error(f"Configuração de fine-tuning não encontrada: {config_path}")
        return

    config = FineTuningConfig(experiment_config=str(config_path))

    pipeline = FineTuningPipeline(config)
    results = pipeline.run(
        run_type=config.run_type,
        max_parallel_folds=config.max_parallel_folds,
    )

    return results


if __name__ == "__main__":
    main()

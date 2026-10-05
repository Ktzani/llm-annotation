"""
Entry point: ESTIMAÇÃO da validação humana (margem de erro por grupo).

Fluxo de cada rodada:
    1. `run_human_validation.py` gera a rodada; as planilhas vão aos avaliadores
    2. As planilhas preenchidas voltam para `validacao_humana/rodada_XX/` com o mesmo nome
    3. Este script recalcula θ̂, IC e MoE por grupo com todas as rodadas e indica
       quais grupos ainda precisam de rodada (MoE > ε)

As configurações são definidas estaticamente abaixo.
"""
import sys
from loguru import logger

# Garante UTF-8 no console do Windows (evita UnicodeEncodeError com emojis/acentos).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

from src.systems.human_validation_system.estimation.pipeline import (
    HumanValidationEstimationConfig,
    HumanValidationEstimationPipeline,
)


def main() -> None:
    # Configuração estática
    experiments = {
        "books": "2026-04-09_13-21-37",
        "dblp": "2026-04-09_14-05-21",
    }

    for dataset_name, specific_date in experiments.items():
        config = HumanValidationEstimationConfig(dataset_name=dataset_name, specific_date=specific_date)
        HumanValidationEstimationPipeline(config).run()


if __name__ == "__main__":
    main()

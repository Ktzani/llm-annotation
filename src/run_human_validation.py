"""
Entry point: preparação da VALIDAÇÃO HUMANA (amostragem incremental por rodadas).

Para cada dataset, gera em `<results>/<dataset>/<date>/validacao_humana/` o guia
do avaliador e, para a rodada pedida, as planilhas cegas (uma por avaliador) e
o gabarito interno.

Como pedir uma nova rodada:
    1. Mantenha `round_number = None` e rode o script: gera a próxima rodada
       (a 1ª tem INITIAL_ROUND_SIZE por grupo; as seguintes, `increment_size`).
    2. Para regenerar arquivos de uma rodada já entregue, use `round_number = k`
       (mesmos documentos e mesma ordem; planilhas existentes não são sobrescritas).

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

from src.systems.human_validation_system.pipeline import (
    HumanValidationConfig,
    HumanValidationPipeline,
)


def main() -> None:
    # Configuração estática
    experiments = {
        "books": "2026-04-09_13-21-37",
        "dblp": "2026-04-09_14-05-21",
    }
    round_number = None  # None = próxima rodada; k = regenera a rodada k
    increment_size = 20  # documentos por grupo nas rodadas após a 1ª

    for dataset_name, specific_date in experiments.items():
        config = HumanValidationConfig(
            dataset_name=dataset_name,
            specific_date=specific_date,
            increment_size=increment_size,
        )
        HumanValidationPipeline(config).run(round_number=round_number)


if __name__ == "__main__":
    main()

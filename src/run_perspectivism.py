"""
Entry point: geração do dataset de PERSPECTIVISMO.

Gera o dataset de perspectivismo (uma linha por anotação de LLM, em vez do voto
majoritário agregado) a partir do `dataset_consenso.csv` de um
experimento, salvando em `<results>/<dataset>/<date>/perspectivismo/`.

Materializar o dataset aqui permite que o fine-tuning (com
`training_mode="perspectivism"`) o reutilize sem refazer a operação.

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

from src.llm_annotation_system.perspectivism.pipeline import (
    PerspectivismConfig,
    PerspectivismPipeline,
)


def main() -> None:
    # Configuração estática
    dataset_name = "movie_review"
    specific_date = "2026-04-09_13-17-23"
    force = False  # True regenera mesmo se o dataset já existir

    config = PerspectivismConfig(
        dataset_name=dataset_name,
        specific_date=specific_date,
        force=force,
    )

    df = PerspectivismPipeline(config).run()

    logger.success("Dataset de perspectivismo pronto.")
    logger.info(
        f"Linhas: {len(df)} | Textos únicos: {df['text_id'].nunique()} | "
        f"LLMs: {sorted(df['llm'].unique())}"
    )


if __name__ == "__main__":
    main()

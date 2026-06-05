"""
Entry point: filtragem por Seleção de Instâncias (biO-IS).

Aplica a técnica biO-IS (waashk/bio-is) para remover instâncias redundantes e
ruidosas do dataset anotado pelas LLMs (coluna de consenso `resolved_annotation`),
gerando um conjunto filtrado para o fine-tuning.

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

from src.is_system.pipeline import InstanceSelectionConfig, InstanceSelectionPipeline


def main() -> None:
    # Configuração estática
    dataset_name = "movie_review"
    specific_date = "2026-04-09_13-17-23"
    method = "bio-is"
    params = {"beta": 0.25, "theta": 0.5}

    config = InstanceSelectionConfig(
        dataset_name=dataset_name,
        specific_date=specific_date,
        method=method,
        params=params,
    )

    result = InstanceSelectionPipeline(config).run()

    s = result.stats
    logger.success("Filtragem concluída.")
    logger.info(
        f"Mantidas: {s['kept_instances']} | Removidas: {s['removed_instances']} "
        f"(redundantes: {s['removed_redundant']}, ruidosas: {s['removed_noise']}) | "
        f"Redução: {s['reduction_rate']:.2%}"
    )


if __name__ == "__main__":
    main()

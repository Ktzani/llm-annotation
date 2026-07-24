import asyncio
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

from src.systems.llm_annotation_system.pipeline import AnnotationConfig
from src.systems.class_filter_system.pipeline import TwoPhaseAnnotationPipeline


async def main():
    experiment = "two_phase_local"

    config_path = Path("src/api/experiments") / "annotation" / f"{experiment}.json"
    if not config_path.exists():
        logger.error(f"Configuração de experimento não encontrada: {config_path}")
        return

    config = AnnotationConfig(experiment_config=str(config_path))

    if not config.class_filter.enabled:
        logger.warning(
            "class_filter.enabled=False neste experimento. A anotação em 2 fases "
            "só faz sentido com o filtro ligado — ative no JSON."
        )
        return

    pipeline = TwoPhaseAnnotationPipeline(config)
    await pipeline.run(run_baseline=config.class_filter.run_baseline)


if __name__ == "__main__":
    asyncio.run(main())

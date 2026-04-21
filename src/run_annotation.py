import asyncio
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from src.llm_annotation_system.pipeline import AnnotationPipeline, AnnotationConfig


async def main():
    run_type = "single_text" # "single_text" para rodar um texto específico, "dataset" para rodar tudo
    experiment = "local_experiment"
    config_path = Path("src/api/experiments") / "annotation" / f"{experiment}.json"
    if not config_path.exists():
        logger.error(f"Configuração de experimento não encontrada: {config_path}")
        return

    config = AnnotationConfig(
        experiment_config=str(config_path),
    )

    pipeline = AnnotationPipeline(config)
    await pipeline.run(run_type=run_type, single_index=0)  # run_type="single" para debug de um texto específico

if __name__ == "__main__":
    asyncio.run(main())
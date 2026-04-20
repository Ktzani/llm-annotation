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
    # =========================================================================
    # CONFIGURE AQUI
    # =========================================================================
    DEBUG_SINGLE = False  # True = anota apenas um texto; False = dataset completo

    experiment = "large_experiment"
    config_path = Path(
        rf"C:\Users\gabri\Documents\GitHub\llm-annotation\src\experiments\{experiment}.json"
    )

    config = AnnotationConfig(
        dataset_name="dblp",
        experiment_config_path=str(config_path),
    )

    pipeline = AnnotationPipeline(config)
    await pipeline.run(debug_single=DEBUG_SINGLE)


if __name__ == "__main__":
    asyncio.run(main())
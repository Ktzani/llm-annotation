import asyncio
import json
import sys
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from src.systems.llm_annotation_system.pipeline import AnnotationPipeline, AnnotationConfig
from src.systems.llm_annotation_system.pipeline_binary import BinaryAnnotationPipeline, BinaryAnnotationConfig


async def main():
    mode = "multiclass" # "multiclass" (uma classe por texto) ou "binary" (sim/não por classe, experimentos em experiments/binary_annotation/)
    run_type = "single_text" # "single_text" para rodar um texto específico, "dataset" para rodar tudo
    experiment = "local_experiment"
    experiments_dir = "binary_annotation" if mode == "binary" else "annotation"
    config_path = Path("src/api/experiments") / experiments_dir / f"{experiment}.json"
    if not config_path.exists():
        logger.error(f"Configuração de experimento não encontrada: {config_path}")
        return

    if mode == "binary":
        pipeline = BinaryAnnotationPipeline(BinaryAnnotationConfig(experiment_config=str(config_path)))
        await pipeline.run(run_type=run_type, single_index=0)
        return

    config = AnnotationConfig(
        experiment_config=str(config_path),
    )

    pipeline = AnnotationPipeline(config)
    await pipeline.run(run_type=run_type, single_index=0)  # run_type="single" para debug de um texto específico

if __name__ == "__main__":
    asyncio.run(main())

import asyncio
from datetime import datetime
from pathlib import Path
from loguru import logger

from src.api.core.state import annotation_jobs
from src.api.schemas.binary_annotation_experiment.experiment import BinaryAnnotationRequest
from src.systems.llm_annotation_system.pipeline_binary import BinaryAnnotationConfig, BinaryAnnotationPipeline
from src.systems.llm_annotation_system.binary_annotation.binary_llm_annotator import BinaryLLMAnnotator


async def run_binary_annotation_background(
    annotation_id: str,
    config: BinaryAnnotationRequest,
):
    try:
        annotation_jobs[annotation_id].status = "running"
        annotation_jobs[annotation_id].started_at = datetime.now()
        annotation_jobs[annotation_id].message = "Inicializando pipeline de anotação binária..."

        pipeline_config = BinaryAnnotationConfig(
            experiment_config=config,
        )

        annotation_jobs[annotation_id].progress = 0.1
        annotation_jobs[annotation_id].message = "Configuração montada. Carregando dados..."

        pipeline = BinaryAnnotationPipeline(pipeline_config)

        annotation_jobs[annotation_id].progress = 0.2
        annotation_jobs[annotation_id].message = "Pipeline criado. Executando..."

        output_dir, texts, categories, ground_truth = await pipeline.run(run_type="dataset")

        annotation_jobs[annotation_id].status = "completed"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].progress = 1.0
        annotation_jobs[annotation_id].message = "Anotação binária concluída com sucesso"
        annotation_jobs[annotation_id].results = {
            "num_texts": len(texts),
            "num_classes": len(config.class_prompts),
            "num_models": len(config.models),
            "has_ground_truth": ground_truth is not None,
            "output_dir": str(output_dir),
        }

        logger.success(f"[{annotation_id}] Anotação binária concluída!")

    except asyncio.CancelledError:
        # As chamadas aos modelos já foram interrompidas e o que foi anotado está no checkpoint
        checkpoint = Path(config.results.dir) / config.dataset_name / BinaryLLMAnnotator.RESULTS_SUBDIR / "intermediate.csv"
        logger.warning(f"[{annotation_id}] Anotação binária cancelada")
        annotation_jobs[annotation_id].status = "cancelled"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].results = {"checkpoint": str(checkpoint)}
        annotation_jobs[annotation_id].message = "Anotação cancelada. Rodar o mesmo dataset de novo retoma do checkpoint"
        raise

    except Exception as e:
        logger.exception(f"[{annotation_id}] Erro na anotação binária: {e}")
        annotation_jobs[annotation_id].status = "failed"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].message = f"Erro: {str(e)}"

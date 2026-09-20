import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from src.api.core.state import annotation_jobs
from src.api.schemas.annotation_experiment.experiment import AnnotationRequest
from src.systems.llm_annotation_system.pipeline import AnnotationConfig, AnnotationPipeline
from src.systems.class_filter_system.pipeline import TwoPhaseAnnotationPipeline


def _cancelled_results(config: AnnotationRequest, pipeline: Optional[object]) -> dict:
    """Onde ficou o que já foi anotado e como retomar"""
    run_dir = getattr(pipeline, "run_dir", None)
    if run_dir is not None:
        return {"mode": "two_phase", "output_dir": str(run_dir), "resume_from": run_dir.name}
    return {"checkpoint": str(Path(config.results.dir) / config.dataset_name / "intermediate.csv")}


async def run_annotation_background(
    annotation_id: str,
    config: AnnotationRequest,
):
    pipeline = None
    try:
        annotation_jobs[annotation_id].status = "running"
        annotation_jobs[annotation_id].started_at = datetime.now()
        annotation_jobs[annotation_id].message = "Inicializando pipeline de anotação..."

        pipeline_config = AnnotationConfig(
            experiment_config=config,
        )

        annotation_jobs[annotation_id].progress = 0.1
        annotation_jobs[annotation_id].message = "Configuração montada. Carregando dados..."

        # Anotação em 2 fases (filtro de classes) — só quando explicitamente ligada.
        # Default (enabled=False) preserva o caminho clássico intacto.
        if config.class_filter.enabled:
            annotation_jobs[annotation_id].message = "Pipeline 2 fases (filtro de classes). Executando..."
            pipeline = TwoPhaseAnnotationPipeline(pipeline_config)
            annotation_jobs[annotation_id].progress = 0.2

            base_dir = await pipeline.run(run_baseline=config.class_filter.run_baseline)

            annotation_jobs[annotation_id].status = "completed"
            annotation_jobs[annotation_id].completed_at = datetime.now()
            annotation_jobs[annotation_id].progress = 1.0
            annotation_jobs[annotation_id].message = "Anotação (2 fases) concluída com sucesso"
            annotation_jobs[annotation_id].results = {
                "mode": "two_phase",
                "num_models": len(config.models),
                "class_filter_method": config.class_filter.method,
                "k": config.class_filter.k,
                "ran_baseline": config.class_filter.run_baseline,
                "output_dir": str(base_dir),
            }
            logger.success(f"[{annotation_id}] Anotação (2 fases) concluída!")
            return

        pipeline = AnnotationPipeline(pipeline_config)

        annotation_jobs[annotation_id].progress = 0.2
        annotation_jobs[annotation_id].message = f"Pipeline criado. Executando..."

        output_dir, texts, categories, ground_truth = await pipeline.run(run_type="dataset")

        annotation_jobs[annotation_id].status = "completed"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].progress = 1.0
        annotation_jobs[annotation_id].message = "Anotação concluída com sucesso"
        annotation_jobs[annotation_id].results = {
            "num_texts": len(texts),
            "num_models": len(config.models),
            "num_repetitions": config.annotation.num_repetitions_per_llm,
            "categories": categories,
            "has_ground_truth": ground_truth is not None,
            "output_dir": str(output_dir),
        }

        logger.success(f"[{annotation_id}] Anotação concluída!")

    except asyncio.CancelledError:
        # As chamadas aos modelos já foram interrompidas e o que foi anotado está no checkpoint
        results = _cancelled_results(config, pipeline)
        logger.warning(f"[{annotation_id}] Anotação cancelada")
        annotation_jobs[annotation_id].status = "cancelled"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].results = results
        annotation_jobs[annotation_id].message = (
            f"Anotação cancelada. Para retomar: class_filter.resume_from='{results['resume_from']}'"
            if "resume_from" in results
            else "Anotação cancelada. Rodar o mesmo dataset de novo retoma do checkpoint"
        )
        raise

    except Exception as e:
        logger.exception(f"[{annotation_id}] Erro na anotação: {e}")
        annotation_jobs[annotation_id].status = "failed"
        annotation_jobs[annotation_id].completed_at = datetime.now()
        annotation_jobs[annotation_id].message = f"Erro: {str(e)}"

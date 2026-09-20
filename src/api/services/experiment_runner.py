import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from src.api.core.state import experiments
from src.api.schemas.annotation_experiment.experiment import ExperimentRequest
from src.systems.llm_annotation_system.pipeline import AnnotationConfig, AnnotationPipeline
from src.systems.class_filter_system.pipeline import TwoPhaseAnnotationPipeline


def _cancelled_results(config: ExperimentRequest, pipeline: Optional[object]) -> dict:
    """Onde ficou o que já foi anotado e como retomar"""
    run_dir = getattr(pipeline, "run_dir", None)
    if run_dir is not None:
        return {"mode": "two_phase", "output_dir": str(run_dir), "resume_from": run_dir.name}
    return {"checkpoint": str(Path(config.results.dir) / config.dataset_name / "intermediate.csv")}


async def run_experiment_background(
    experiment_id: str,
    config: ExperimentRequest,
):
    pipeline = None
    try:
        experiments[experiment_id].status = "running"
        experiments[experiment_id].started_at = datetime.now()
        experiments[experiment_id].message = "Inicializando pipeline de anotação..."

        pipeline_config = AnnotationConfig(
            experiment_config=config,
        )

        experiments[experiment_id].progress = 0.1
        experiments[experiment_id].message = "Configuração montada. Carregando dados..."

        # Anotação em 2 fases (filtro de classes) — só quando explicitamente ligada.
        # Default (enabled=False) preserva o caminho clássico intacto.
        if config.class_filter.enabled:
            experiments[experiment_id].message = "Pipeline 2 fases (filtro de classes). Executando..."
            pipeline = TwoPhaseAnnotationPipeline(pipeline_config)
            experiments[experiment_id].progress = 0.2

            base_dir = await pipeline.run(run_baseline=config.class_filter.run_baseline)

            experiments[experiment_id].status = "completed"
            experiments[experiment_id].completed_at = datetime.now()
            experiments[experiment_id].progress = 1.0
            experiments[experiment_id].message = "Experimento (2 fases) concluído com sucesso"
            experiments[experiment_id].results = {
                "mode": "two_phase",
                "num_models": len(config.models),
                "class_filter_method": config.class_filter.method,
                "k": config.class_filter.k,
                "ran_baseline": config.class_filter.run_baseline,
                "output_dir": str(base_dir),
            }
            logger.success(f"[{experiment_id}] Experimento (2 fases) concluído!")
            return

        pipeline = AnnotationPipeline(pipeline_config)

        experiments[experiment_id].progress = 0.2
        experiments[experiment_id].message = f"Pipeline criado. Executando..."

        output_dir, texts, categories, ground_truth = await pipeline.run(run_type="dataset")

        experiments[experiment_id].status = "completed"
        experiments[experiment_id].completed_at = datetime.now()
        experiments[experiment_id].progress = 1.0
        experiments[experiment_id].message = "Experimento concluído com sucesso"
        experiments[experiment_id].results = {
            "num_texts": len(texts),
            "num_models": len(config.models),
            "num_repetitions": config.annotation.num_repetitions_per_llm,
            "categories": categories,
            "has_ground_truth": ground_truth is not None,
            "output_dir": str(output_dir),
        }

        logger.success(f"[{experiment_id}] Experimento concluído!")

    except asyncio.CancelledError:
        # As chamadas aos modelos já foram interrompidas e o que foi anotado está no checkpoint
        results = _cancelled_results(config, pipeline)
        logger.warning(f"[{experiment_id}] Experimento cancelado")
        experiments[experiment_id].status = "cancelled"
        experiments[experiment_id].completed_at = datetime.now()
        experiments[experiment_id].results = results
        experiments[experiment_id].message = (
            f"Experimento cancelado. Para retomar: class_filter.resume_from='{results['resume_from']}'"
            if "resume_from" in results
            else "Experimento cancelado. Rodar o mesmo dataset de novo retoma do checkpoint"
        )
        raise

    except Exception as e:
        logger.exception(f"[{experiment_id}] Erro no experimento: {e}")
        experiments[experiment_id].status = "failed"
        experiments[experiment_id].completed_at = datetime.now()
        experiments[experiment_id].message = f"Erro: {str(e)}"

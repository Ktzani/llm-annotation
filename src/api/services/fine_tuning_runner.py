from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.api.core.state import fine_tuning_jobs
from src.api.schemas.fine_tuning.fine_tuning import FineTuningRequest
from src.fine_tune_system.pipeline import FineTuningConfig, FineTuningPipeline


async def run_fine_tuning_background(
    job_id: str,
    config: FineTuningRequest,
) -> None:
    try:
        fine_tuning_jobs[job_id].status = "running"
        fine_tuning_jobs[job_id].started_at = datetime.now()
        fine_tuning_jobs[job_id].message = "Inicializando pipeline de fine-tuning..."

        logger.info(
            f"[{job_id}] Iniciando fine-tuning — dataset={config.dataset.dataset_name}, "
            f"model={config.model_name}, instance_selection={config.instance_selection.enabled} "
            f"({config.instance_selection.method})"
        )

        # A API passa o request já validado (forma em memória do FineTuningConfig).
        ft_config = FineTuningConfig(experiment_config=config)

        fine_tuning_jobs[job_id].progress = 0.1
        fine_tuning_jobs[job_id].message = "Configuração montada. Carregando dados..."

        pipeline = FineTuningPipeline(ft_config)

        fine_tuning_jobs[job_id].progress = 0.2
        fine_tuning_jobs[job_id].message = f"Pipeline criado. Executando run_type='{config.run_type}'..."

        # pipeline.run() é síncrono — rodamos direto (já estamos numa background task)
        results: dict = pipeline.run(run_type=config.run_type, max_parallel_folds=config.max_parallel_folds)

        fine_tuning_jobs[job_id].status = "completed"
        fine_tuning_jobs[job_id].completed_at = datetime.now()
        fine_tuning_jobs[job_id].progress = 1.0
        fine_tuning_jobs[job_id].message = "Fine-tuning concluído com sucesso"
        fine_tuning_jobs[job_id].results = {
            "dataset_name": config.dataset.dataset_name,
            "model_name": config.model_name,
            "run_type": config.run_type,
            "max_parallel_folds": config.max_parallel_folds,
            "instance_selection": config.instance_selection.model_dump(),
            "metrics": results,
        }

        logger.success(f"[{job_id}] Fine-tuning concluído!")

    except Exception as e:
        logger.exception(f"[{job_id}] Erro no fine-tuning: {e}")
        fine_tuning_jobs[job_id].status = "failed"
        fine_tuning_jobs[job_id].completed_at = datetime.now()
        fine_tuning_jobs[job_id].message = f"Erro: {str(e)}"
import asyncio
from datetime import datetime
from typing import Callable

from loguru import logger

from src.api.core.job_process import JobProcess
from src.api.core.state import fine_tuning_jobs
from src.api.schemas.fine_tuning.fine_tuning import FineTuningRequest
from src.systems.fine_tune_system.pipeline import FineTuningConfig, FineTuningPipeline


def run_fine_tuning_job(report: Callable[..., None], config: FineTuningRequest) -> dict:
    """Executa o fine-tuning no processo separado; report envia o progresso para a API"""
    # A API passa o request já validado (forma em memória do FineTuningConfig).
    ft_config = FineTuningConfig(experiment_config=config)
    pipeline = FineTuningPipeline(ft_config)
    report(run_name=pipeline.fine_tune_output_dir.name, output_dir=str(pipeline.fine_tune_output_dir))

    return pipeline.run(run_type=config.run_type, max_parallel_folds=config.max_parallel_folds)


async def run_fine_tuning_background(
    job_id: str,
    config: FineTuningRequest,
) -> None:
    # Processo separado: o treino não trava a API e o cancelamento mata o processo (libera a GPU)
    process = JobProcess(run_fine_tuning_job, config)
    run_info: dict = {}
    results: dict = {}

    try:
        fine_tuning_jobs[job_id].status = "running"
        fine_tuning_jobs[job_id].started_at = datetime.now()
        fine_tuning_jobs[job_id].message = "Inicializando pipeline de fine-tuning..."

        logger.info(
            f"[{job_id}] Iniciando fine-tuning — dataset={config.dataset.dataset_name}, "
            f"model={config.model_name}, instance_selection={config.instance_selection.enabled} "
            f"({config.instance_selection.method})"
        )

        process.start()

        fine_tuning_jobs[job_id].progress = 0.1
        fine_tuning_jobs[job_id].message = "Processo de fine-tuning iniciado. Carregando dados..."

        async for kind, payload in process.messages():
            if kind == "progress":
                run_info = payload
                fine_tuning_jobs[job_id].progress = 0.2
                fine_tuning_jobs[job_id].message = f"Pipeline criado ({run_info['run_name']}). Executando run_type='{config.run_type}'..."
            else:
                results = payload

        fine_tuning_jobs[job_id].status = "completed"
        fine_tuning_jobs[job_id].completed_at = datetime.now()
        fine_tuning_jobs[job_id].progress = 1.0
        fine_tuning_jobs[job_id].message = "Fine-tuning concluído com sucesso"
        fine_tuning_jobs[job_id].results = {
            "dataset_name": config.dataset.dataset_name,
            "run_name": run_info.get("run_name"),
            "output_dir": run_info.get("output_dir"),
            "model_name": config.model_name,
            "run_type": config.run_type,
            "max_parallel_folds": config.max_parallel_folds,
            "instance_selection": config.instance_selection.model_dump(),
            "metrics": results,
        }

        logger.success(f"[{job_id}] Fine-tuning concluído!")

    except asyncio.CancelledError:
        await process.kill()
        logger.warning(f"[{job_id}] Fine-tuning cancelado")
        fine_tuning_jobs[job_id].status = "cancelled"
        fine_tuning_jobs[job_id].completed_at = datetime.now()
        fine_tuning_jobs[job_id].message = "Fine-tuning cancelado"
        if run_info:
            fine_tuning_jobs[job_id].message += f" (arquivos parciais em {run_info['output_dir']})"
        raise

    except Exception as e:
        await process.kill()
        logger.exception(f"[{job_id}] Erro no fine-tuning: {e}")
        fine_tuning_jobs[job_id].status = "failed"
        fine_tuning_jobs[job_id].completed_at = datetime.now()
        fine_tuning_jobs[job_id].message = f"Erro: {str(e)}"

import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from src.api.core.state import fine_tuning_jobs, cancellation_tokens
from src.api.schemas.fine_tuning.fine_tuning import FineTuningRequest, FineTuningStatus
from src.api.services.fine_tuning_runner import run_fine_tuning_background

router = APIRouter(prefix="/fine-tuning", tags=["Fine-Tuning"])


@router.post("/", response_model=FineTuningStatus)
async def create_fine_tuning_job(
    config: FineTuningRequest,
    background_tasks: BackgroundTasks,
):
    """Cria e inicia um novo job de fine-tuning em background."""
    job_id = str(uuid.uuid4())

    job_status = FineTuningStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
    )

    fine_tuning_jobs[job_id] = job_status

    background_tasks.add_task(run_fine_tuning_background, job_id, config)

    logger.info(f"Job de fine-tuning {job_id} criado e agendado")
    return job_status


@router.get("/{job_id}", response_model=FineTuningStatus)
async def get_fine_tuning_job(job_id: str):
    """Retorna o status e resultados de um job de fine-tuning."""
    if job_id not in fine_tuning_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    return fine_tuning_jobs[job_id]


@router.get("/")
async def list_fine_tuning_jobs():
    """Lista todos os jobs de fine-tuning."""
    return {
        "total": len(fine_tuning_jobs),
        "jobs": list(fine_tuning_jobs.values()),
    }


@router.delete("/{job_id}")
async def delete_fine_tuning_job(job_id: str):
    """Remove um job do histórico (não cancela se estiver rodando)."""
    if job_id not in fine_tuning_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    del fine_tuning_jobs[job_id]
    return {"message": "Job removido com sucesso"}


# !! TODO !! - endpoint de cancelamento
@router.post("/{job_id}/cancel")
async def cancel_fine_tuning_job(job_id: str):
    if job_id not in fine_tuning_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    if fine_tuning_jobs[job_id].status != "running":
        raise HTTPException(status_code=400, detail="Job não está em execução")

    cancellation_tokens[job_id].cancel()
    fine_tuning_jobs[job_id].status = "cancelled"
    fine_tuning_jobs[job_id].message = "Cancelamento solicitado"
    return {"message": "Cancelamento solicitado"}
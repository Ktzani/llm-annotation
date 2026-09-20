import uuid
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException
from loguru import logger

from src.api.core.state import consensus_jobs
from src.api.schemas.consensus.consensus import ConsensusRequest, ConsensusStatus
from src.api.services.consensus_runner import run_consensus_background

router = APIRouter(prefix="/consensus", tags=["Consensus"])

@router.post("/", response_model=ConsensusStatus)
async def create_consensus_job(
    config: ConsensusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Aplica o consenso entre LLMs sobre as anotações de um experimento já existente.

    Equivale ao entry point `src/run_consensus.py`, porém executado no próprio
    servidor onde as anotações foram geradas (evita baixar `annotations.csv`,
    rodar localmente e devolver os artefatos).

    Gera, em `<results_dir>/<dataset>/<date>/`:
        consensus/dataset_consenso.csv   Dataset com `resolved_annotation`
        consensus/problematic_cases.csv  Casos de baixo consenso
        summary/alta_confianca.csv       Score >= threshold
        summary/necessita_revisao.csv    Score < threshold
        summary/sumario_experimento.json Métricas resumidas
    """
    job_id = str(uuid.uuid4())

    job_status = ConsensusStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
    )

    consensus_jobs[job_id] = job_status

    background_tasks.add_task(run_consensus_background, job_id, config)

    logger.info(f"Job de consenso {job_id} criado e agendado")
    return job_status


@router.get("/{job_id}", response_model=ConsensusStatus)
async def get_consensus_job(job_id: str):
    """Retorna o status e o relatório de um job de consenso."""
    if job_id not in consensus_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    return consensus_jobs[job_id]


@router.get("/")
async def list_consensus_jobs():
    """Lista todos os jobs de consenso."""
    return {
        "total": len(consensus_jobs),
        "jobs": list(consensus_jobs.values()),
    }


@router.delete("/{job_id}")
async def delete_consensus_job(job_id: str):
    """Remove um job do histórico (não cancela se estiver rodando)."""
    if job_id not in consensus_jobs:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    del consensus_jobs[job_id]
    return {"message": "Job removido com sucesso"}

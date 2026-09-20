from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime

from src.api.core.state import experiments, job_runner
from src.api.schemas.annotation_experiment.experiment import ExperimentRequest, ExperimentStatus
from src.api.services.experiment_runner import run_experiment_background

from loguru import logger

router = APIRouter(prefix="/experiments", tags=["Experiments"])

@router.post("/", response_model=ExperimentStatus)
async def create_experiment(config: ExperimentRequest):
    """Cria e inicia um novo experimento"""
    experiment_id = str(uuid.uuid4())
    
    # Criar registro do experimento
    experiment_status = ExperimentStatus(
        experiment_id=experiment_id,
        status="pending",
        created_at=datetime.now()
    )
    
    experiments[experiment_id] = experiment_status
    
    # Executa em background (cancelável via /cancel)
    job_runner.start(experiment_id, run_experiment_background(experiment_id, config))
    
    logger.info(f"Experimento {experiment_id} criado e agendado")
    return experiment_status

@router.get("/{experiment_id}", response_model=ExperimentStatus)
async def get_experiment(experiment_id: str):
    """Obtém o status de um experimento"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")
    
    return experiments[experiment_id]

@router.get("/")
async def list_experiments():
    """Lista todos os experimentos"""
    return {
        "total": len(experiments),
        "experiments": list(experiments.values())
    }

@router.delete("/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Remove um experimento do histórico"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    if job_runner.is_running(experiment_id):
        raise HTTPException(status_code=409, detail="Experimento em execução: cancele antes de remover")

    del experiments[experiment_id]
    return {"message": "Experimento removido com sucesso"}

@router.post("/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str):
    """Cancela o experimento: interrompe as chamadas aos modelos e salva o que já foi anotado"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    if not job_runner.cancel(experiment_id):
        raise HTTPException(status_code=400, detail="Experimento não está em execução")

    experiments[experiment_id].status = "cancelling"
    experiments[experiment_id].message = "Cancelamento solicitado"
    return {"message": "Cancelamento solicitado"}

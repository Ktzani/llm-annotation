from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime

from src.api.core.state import annotation_jobs, job_runner
from src.api.schemas.annotation_experiment.experiment import AnnotationRequest, AnnotationStatus
from src.api.services.annotation_runner import run_annotation_background

from loguru import logger

router = APIRouter(prefix="/annotation", tags=["Annotation"])

@router.post("/", response_model=AnnotationStatus)
async def create_annotation(config: AnnotationRequest):
    """Cria e inicia uma nova anotação"""
    annotation_id = str(uuid.uuid4())

    # Criar registro da anotação
    annotation_status = AnnotationStatus(
        annotation_id=annotation_id,
        status="pending",
        created_at=datetime.now()
    )

    annotation_jobs[annotation_id] = annotation_status

    # Executa em background (cancelável via /cancel)
    job_runner.start(annotation_id, run_annotation_background(annotation_id, config))

    logger.info(f"Anotação {annotation_id} criada e agendada")
    return annotation_status

@router.get("/{annotation_id}", response_model=AnnotationStatus)
async def get_annotation(annotation_id: str):
    """Obtém o status de uma anotação"""
    if annotation_id not in annotation_jobs:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")

    return annotation_jobs[annotation_id]

@router.get("/")
async def list_annotations():
    """Lista todas as anotações"""
    return {
        "total": len(annotation_jobs),
        "annotations": list(annotation_jobs.values())
    }

@router.delete("/{annotation_id}")
async def delete_annotation(annotation_id: str):
    """Remove uma anotação do histórico"""
    if annotation_id not in annotation_jobs:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")

    if job_runner.is_running(annotation_id):
        raise HTTPException(status_code=409, detail="Anotação em execução: cancele antes de remover")

    del annotation_jobs[annotation_id]
    return {"message": "Anotação removida com sucesso"}

@router.post("/{annotation_id}/cancel")
async def cancel_annotation(annotation_id: str):
    """Cancela a anotação: interrompe as chamadas aos modelos e salva o que já foi anotado"""
    if annotation_id not in annotation_jobs:
        raise HTTPException(status_code=404, detail="Anotação não encontrada")

    if not job_runner.cancel(annotation_id):
        raise HTTPException(status_code=400, detail="Anotação não está em execução")

    annotation_jobs[annotation_id].status = "cancelling"
    annotation_jobs[annotation_id].message = "Cancelamento solicitado"
    return {"message": "Cancelamento solicitado"}

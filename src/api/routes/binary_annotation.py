from fastapi import APIRouter, BackgroundTasks
import uuid
from datetime import datetime

from src.api.core.state import annotation_jobs, consensus_jobs, job_runner
from src.api.schemas.annotation_experiment.experiment import AnnotationStatus
from src.api.schemas.binary_annotation_experiment.experiment import BinaryAnnotationRequest
from src.api.schemas.consensus.consensus import ConsensusStatus
from src.api.schemas.consensus.binary_consensus import BinaryConsensusRequest
from src.api.services.binary_annotation_runner import run_binary_annotation_background
from src.api.services.binary_consensus_runner import run_binary_consensus_background

from loguru import logger

router = APIRouter(prefix="/binary-annotation", tags=["Annotation"])

@router.post("/", response_model=AnnotationStatus)
async def create_binary_annotation(config: BinaryAnnotationRequest):
    """
    Cria e inicia uma nova anotação binária (uma pergunta sim/não por classe).

    O job entra no mesmo registro da anotação: status, listagem, remoção e cancelamento
    seguem pelas rotas /annotation/{annotation_id}.
    """
    annotation_id = str(uuid.uuid4())

    annotation_status = AnnotationStatus(
        annotation_id=annotation_id,
        status="pending",
        created_at=datetime.now()
    )

    annotation_jobs[annotation_id] = annotation_status

    # Executa em background (cancelável via POST /annotation/{annotation_id}/cancel)
    job_runner.start(annotation_id, run_binary_annotation_background(annotation_id, config))

    logger.info(f"Anotação binária {annotation_id} criada e agendada")
    return annotation_status

@router.post("/consensus", response_model=ConsensusStatus)
async def create_binary_consensus_job(
    config: BinaryConsensusRequest,
    background_tasks: BackgroundTasks,
):
    """
    Aplica o consenso existente sobre uma anotação binária já feita (corpo do POST /consensus, sem `categories`).

    `specific_date` é a data da execução em `<results_dir>/<dataset>/binary/` (ou 'latest').
    Além dos artefatos do /consensus, gera `binary_labels.csv`: o consenso como planilha texto × classe.
    O status segue pela rota /consensus/{job_id}.
    """
    job_id = str(uuid.uuid4())

    job_status = ConsensusStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
    )

    consensus_jobs[job_id] = job_status

    background_tasks.add_task(run_binary_consensus_background, job_id, config)

    logger.info(f"Job de consenso binário {job_id} criado e agendado")
    return job_status

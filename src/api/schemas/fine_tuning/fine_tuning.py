from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

from src.api.schemas.fine_tuning.dataset import FineTuningDatasetConfig
from src.api.schemas.fine_tuning.hyperparams import FineTuningHyperparams

class FineTuningRequest(BaseModel):
    """Configuração completa para um job de fine-tuning"""

    dataset: FineTuningDatasetConfig = Field(default_factory=FineTuningDatasetConfig, description="Configuração do dataset para fine-tuning")
    hyperparams: FineTuningHyperparams = Field(default_factory=FineTuningHyperparams, description="Hiperparâmetros de treinamento")

    model_name: str = Field(default="roberta-base", description="Modelo base para fine-tuning")
    run_type: str = Field(
        default="cv",
        description="Tipo de execução: 'cv' (cross-validation) ou 'unique' (teste rápido)",
    )

class FineTuningStatus(BaseModel):
    """Status de um job de fine-tuning"""

    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    results: Optional[Any] = None
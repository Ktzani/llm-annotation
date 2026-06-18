from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

from src.api.schemas.fine_tuning.dataset import FineTuningDatasetConfig
from src.api.schemas.fine_tuning.hyperparams import FineTuningHyperparams
from src.api.schemas.fine_tuning.instance_selection import FineTuningInstanceSelectionConfig

class FineTuningRequest(BaseModel):
    """Configuração completa para um job de fine-tuning"""

    dataset: FineTuningDatasetConfig = Field(
        default_factory=FineTuningDatasetConfig, 
        description="Configuração do dataset para fine-tuning"
    )
    
    hyperparams: FineTuningHyperparams = Field(
        default_factory=FineTuningHyperparams, 
        description="Hiperparâmetros de treinamento"
    )
    
    instance_selection: FineTuningInstanceSelectionConfig = Field(
        default_factory=FineTuningInstanceSelectionConfig,
        description="Configuração da filtragem por seleção de instâncias (biO-IS)",
    )

    model_name: str = Field(
        default="roberta-base",
        description="Modelo base para fine-tuning"
    )
    
    run_type: str = Field(
        default="cross-validation",
        description="Tipo de execução: 'cross-validation' ou 'single' (teste rápido)",
    )

    training_mode: str = Field(
        default="aggregated",
        description=(
            "Estratégia de rotulagem do treino: 'aggregated' (voto majoritário/"
            "consenso agregado em `resolved_annotation`) ou 'perspectivism' (uma "
            "linha por anotação de LLM — preserva o desacordo entre anotadores)."
        ),
    )

    max_parallel_folds: int = Field(default=4, description="Número máximo de folds a serem processados em paralelo (apenas para 'cross-validation')")

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
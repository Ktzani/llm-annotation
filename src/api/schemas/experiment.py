from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.api.schemas.enums import PromptType
from src.api.schemas.dataset_config import DatasetConfig
from src.api.schemas.cache_config import CacheConfig
from src.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy

class ExperimentRequest(BaseModel):
    # Dataset
    dataset_name: str = Field(..., description="Nome do dataset HuggingFace")
    dataset_config: DatasetConfig = Field(default_factory=DatasetConfig)

    # Cache e output
    cache: CacheConfig = Field(default_factory=CacheConfig)
    save_intermediate: bool = Field(default=True)
    results_dir: Optional[str] = Field(default=None)
    
    # Modelos
    models: List[str] = Field(
        default=[  
            "deepseek-r1-8b",
            "qwen3-8b",
            "gemma3-4b",
            "mistral-7b",
            "llama3.1-8b"
        ],
        description="Lista de modelos a usar"
    )
    
    # Prompt
    prompt_type: PromptType = Field(
        default=PromptType.BASE,
        description="Tipo de prompt a utilizar"
    )
    
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Prompt customizado (sobrescreve prompt_type)"
    )
    
    # Execução
    num_repetitions_per_llm: int = Field(
        default=1,
        ge=1,
        description="Número de repetições por LLM"
    )

    use_alternative_params: bool = Field(default=False)
    
    model_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução dos modelos"
    )
    rep_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução das repetições"
    )


class ExperimentStatus(BaseModel):
    experiment_id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

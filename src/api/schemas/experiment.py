from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.api.schemas.prompt_enum import PromptType
from src.api.schemas.dataset import DatasetConfig
from src.api.schemas.cache import CacheConfig
from src.api.schemas.annotation import AnnotationConfig
from src.api.schemas.results import ResultsConfig

class ExperimentRequest(BaseModel):
    """
    Schema principal de configuração de um experimento de anotação com LLMs.

    Este objeto define:
    - qual dataset será usado
    - quais modelos serão executados
    - como será feita a anotação
    - como os resultados serão armazenados
    """

    dataset_name: str = Field(
        ...,
        description=(
            "Nome do dataset no HuggingFace Hub. "
            "Exemplos: 'sst2', 'ag_news', 'emotion'."
        )
    )

    dataset_config: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description=(
            "Configurações de carregamento do dataset, incluindo split, "
            "amostragem, combinação de splits e seed para reprodutibilidade."
        )
    )

    models: List[str] = Field(
        default=[
            "deepseek-r1-8b",
            "qwen3-8b",
            "gemma3-4b",
            "mistral-7b",
            "llama3.1-8b"
        ],
        description=(
            "Lista de modelos LLM que serão utilizados para anotação. "
            "Cada modelo será executado de forma independente "
            "(potencialmente em paralelo)."
        )
    )

    prompt_type: PromptType = Field(
        default=PromptType.BASE,
        description=(
            "Tipo de prompt padrão a ser utilizado no experimento. "
            "Define a estrutura base da instrução enviada às LLMs."
        )
    )

    custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Prompt customizado fornecido manualmente. "
            "Caso definido, sobrescreve completamente o prompt associado ao prompt_type."
        )
    )

    annotation: AnnotationConfig = Field(
        default_factory=AnnotationConfig,
        description=(
            "Configurações do processo de anotação, incluindo número de repetições, "
            "estratégia de paralelismo entre modelos e repetições, "
            "e parâmetros alternativos de inferência."
        )
    )

    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description=(
            "Configurações do sistema de cache. "
            "Controla se o cache será utilizado e onde os resultados serão armazenados."
        )
    )

    results: ResultsConfig = Field(
        default_factory=ResultsConfig,
        description=(
            "Configurações de persistência dos resultados do experimento, "
            "incluindo salvamento intermediário, diretório de saída "
            "e organização dos artefatos gerados."
        )
    )



class ExperimentStatus(BaseModel):
    """
    Representa o estado atual de um experimento em execução ou já finalizado.

    Utilizado para:
    - acompanhamento via polling
    - monitoramento de progresso
    - recuperação de resultados
    """

    experiment_id: str = Field(
        description="Identificador único do experimento (UUID)."
    )

    status: str = Field(
        description=(
            "Estado atual do experimento. "
            "Valores possíveis: 'pending', 'running', 'completed', 'failed'."
        )
    )

    created_at: datetime = Field(
        description="Timestamp de criação do experimento."
    )

    started_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp de início da execução do experimento."
    )

    completed_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp de finalização do experimento."
    )

    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Progresso do experimento no intervalo [0, 1], "
            "onde 0 indica não iniciado e 1 indica concluído."
        )
    )

    message: Optional[str] = Field(
        default=None,
        description=(
            "Mensagem descritiva do estado atual do experimento, "
            "utilizada para feedback ao usuário."
        )
    )

    results: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Resultados resumidos do experimento após a conclusão, "
            "incluindo métricas, caminhos de saída e metadados."
        )
    )


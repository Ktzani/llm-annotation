from pydantic import BaseModel, Field
from typing import Any, List, Literal, Optional
from datetime import datetime


class ConsensusRequest(BaseModel):
    """
    Configuração da aplicação de consenso sobre as anotações de um experimento.

    Espelha os parâmetros de `ConsensusConfig` (mesmos valores usados no entry
    point `src/run_consensus.py`), permitindo rodar o consenso direto no
    servidor onde as anotações foram geradas.
    """

    dataset_name: str = Field(
        ...,
        description="Nome do dataset anotado (ex.: 'movie_review')."
    )

    results_dir: str = Field(
        default=r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\results",
        description=(
            "Diretório base dos resultados de anotação — o mesmo usado em "
            "`results.dir` do experimento."
        )
    )

    specific_date: str = Field(
        default="latest",
        description=(
            "Data/pasta específica do experimento (ex.: '2026-04-09_13-17-23') "
            "ou 'latest' para a mais recente. Atenção: 'latest' usa a data de "
            "modificação e pode apontar para um experimento ainda incompleto."
        )
    )

    consensus_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Score mínimo de concordância para considerar consenso de alta confiança."
    )

    consensus_strategy: Literal[
        "majority_vote",
        "weighted_vote",
        "unanimous_only",
        "remove_outliers",
    ] = Field(
        default="majority_vote",
        description="Estratégia de resolução de conflito quando o score fica abaixo do threshold."
    )

    categories: Optional[List[int]] = Field(
        default=None,
        description=(
            "Categorias válidas do dataset. Se omitido, são derivadas dos "
            "próprios dados anotados."
        )
    )


class ConsensusStatus(BaseModel):
    """Status de um job de consenso."""

    job_id: str
    status: str = "pending"  # pending | running | completed | failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: str = ""
    results: Optional[Any] = None

from pydantic import BaseModel, Field
from typing import Literal


class BinaryConsensusRequest(BaseModel):
    """
    Configuração do consenso sobre uma anotação binária.

    Mesmos parâmetros do `ConsensusRequest`, sem `categories`: na anotação binária as
    categorias são sempre sim/não (1/0), derivadas das próprias respostas.
    """

    dataset_name: str = Field(
        ...,
        description="Nome do dataset anotado (ex.: 'CENARIO_TESTE')."
    )

    results_dir: str = Field(
        default=r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\results",
        description=(
            "Diretório base dos resultados de anotação — o mesmo usado em "
            "`results.dir` do experimento (ex.: 'data/results')."
        )
    )

    specific_date: str = Field(
        default="latest",
        description=(
            "Pasta da execução dentro de `<results_dir>/<dataset>/binary/` (ex.: '2026-10-22_16-35-47') "
            "ou 'latest' para a mais recente. Atenção: 'latest' usa a data de modificação e pode apontar "
            "para uma execução incompleta."
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

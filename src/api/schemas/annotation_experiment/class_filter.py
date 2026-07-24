from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from src.config.class_filter import (
    CLASS_FILTER_STRATEGIES,
    DEFAULT_CLASS_FILTER_METHOD,
    DEFAULT_HOLDOUT_FOLD_INDEX,
    DEFAULT_K,
    DEFAULT_K_SWEEP,
    DEFAULT_N_INNER_FOLDS,
    DEFAULT_TRAIN_FOLD_PATTERN,
    RANDOM_STATE,
)


def _default_filter_params() -> Dict[str, Any]:
    """Hiperparâmetros padrão do método de filtro padrão, vindos da config central."""
    params = dict(CLASS_FILTER_STRATEGIES[DEFAULT_CLASS_FILTER_METHOD])
    params.pop("description", None)
    return params


class ClassFilterConfig(BaseModel):
    """
    Configuração da anotação em 2 fases (filtro de classes + LLM zero-shot).

    Bloco OPCIONAL e retrocompatível: com `enabled=False` (default), o pipeline
    de anotação roda exatamente como o baseline (todas as classes no prompt).

    `params` carrega os hiperparâmetros do classificador escolhido, permitindo
    adicionar novos métodos sem alterar este schema.
    """

    # Exemplo exibido no Swagger/OpenAPI (evita o placeholder genérico
    # `additionalProp1` que o Swagger mostra para Dict[str, Any]).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "method": "logistic_regression",
                "k": 3,
                "k_sweep": [2, 3, 4, 5],
                "n_inner_folds": 5,
                "holdout_fold_index": 0,
                "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
            }
        }
    )

    enabled: bool = Field(
        default=False,
        description=(
            "Se True, ativa a anotação em 2 fases: um classificador filtra as "
            "top-k classes por texto (Fase 1) e o LLM anota zero-shot só entre "
            "elas (Fase 2). Default False mantém o baseline (todas as classes)."
        ),
    )
    method: str = Field(
        default=DEFAULT_CLASS_FILTER_METHOD,
        description="Método do filtro de classes plugável (ex.: 'logistic_regression').",
    )
    k: int = Field(
        default=DEFAULT_K,
        ge=1,
        description="Nº de classes candidatas apresentadas ao LLM na Fase 2.",
    )
    k_sweep: List[int] = Field(
        default_factory=lambda: list(DEFAULT_K_SWEEP),
        description=(
            "k's avaliados no relatório de recall@k (teto de acurácia da Fase 2). "
            "Só validação — não envolve chamadas ao LLM."
        ),
    )
    n_inner_folds: int = Field(
        default=DEFAULT_N_INNER_FOLDS,
        ge=2,
        description=(
            "Nº de pedaços em que o conjunto de treino de cada fold externo é "
            "dividido. O classificador treina em n-1 pedaços; o pedaço held-out "
            "é anotado pelo LLM."
        ),
    )
    holdout_fold_index: int = Field(
        default=DEFAULT_HOLDOUT_FOLD_INDEX,
        ge=0,
        description="Índice (0-based) do pedaço interno usado como held-out.",
    )
    run_baseline: bool = Field(
        default=False,
        description=(
            "Se True, também anota o MESMO held-out com TODAS as classes "
            "(baseline comparável, mesmos textos). Dobra o custo de chamadas ao LLM."
        ),
    )
    train_fold_pattern: str = Field(
        default=DEFAULT_TRAIN_FOLD_PATTERN,
        description="Padrão do nome do arquivo de fold de treino no HF Hub.",
    )
    random_state: int = Field(
        default=RANDOM_STATE,
        description="Semente do split interno e do classificador (reprodutibilidade).",
    )
    params: Dict[str, Any] = Field(
        default_factory=_default_filter_params,
        description="Hiperparâmetros específicos do método de filtro selecionado.",
    )

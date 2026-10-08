from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.config.instance_selection import DEFAULT_IS_METHOD, INSTANCE_SELECTION_STRATEGIES


def _default_is_params() -> Dict[str, Any]:
    """Parâmetros padrão do método de IS padrão (biO-IS), vindos da config central."""
    params = dict(INSTANCE_SELECTION_STRATEGIES[DEFAULT_IS_METHOD])
    params.pop("description", None)
    return params


class FineTuningInstanceSelectionSweep(BaseModel):
    """Curva de IS: um fine-tuning por valor de `param`, com os demais `params` fixos."""

    param: str = Field(default="beta", description="Parâmetro do método variado na curva (ex.: 'beta').")
    values: List[float] = Field(..., min_length=1, description="Valores de `param`, um fine-tuning por valor.")


class FineTuningInstanceSelectionConfig(BaseModel):
    """
    Configuração da filtragem por seleção de instâncias aplicada ao conjunto
    anotado antes do fine-tuning.

    `params` carrega os parâmetros próprios do método escolhido — assim novos
    métodos de filtragem podem ser adicionados sem alterar este schema. O
    default reflete o método padrão (biO-IS) e seus parâmetros
    (`{"beta": 0.25, "theta": 0.5}`), definidos em
    `src/config/instance_selection.py`.
    """

    # Exemplo exibido no Swagger/OpenAPI (evita o placeholder genérico
    # `additionalProp1` que o Swagger mostra para Dict[str, Any]).
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "method": "bio-is",
                "params": {"beta": 0.25, "theta": 0.5},
                "sweep": None,
            }
        }
    )

    enabled: bool = Field(
        default=True,
        description="Se True, filtra instâncias redundantes/ruidosas antes do fine-tuning",
    )
    method: str = Field(
        default=DEFAULT_IS_METHOD,
        description="Método de seleção de instâncias (ex.: 'bio-is')",
    )
    params: Dict[str, Any] = Field(
        default_factory=_default_is_params,
        examples=[{"beta": 0.25, "theta": 0.5}],
        description=(
            "Parâmetros específicos do método selecionado "
            "(biO-IS: {'beta': 0.25, 'theta': 0.5})."
        ),
    )
    sweep: Optional[FineTuningInstanceSelectionSweep] = Field(
        default=None,
        description=(
            "Curva de IS: treina um modelo por valor de `sweep.param` (ex.: beta) com os demais `params` "
            "fixos e salva retenção × macro-F1 em `curve.csv`. Vazio: um único fine-tuning com `params`."
        ),
    )

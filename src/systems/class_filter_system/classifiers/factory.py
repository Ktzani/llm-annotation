"""
Fábrica de filtros de classe.

Espelha `instance_selection_system.selection.selector_factory.get_selector`.
Novos métodos (SVM, embeddings, etc.) podem ser registrados aqui sem alterar o
restante do pipeline.
"""
from src.config.class_filter import CLASS_FILTER_STRATEGIES, RANDOM_STATE
from src.systems.class_filter_system.classifiers.base import ClassFilterBase
from src.systems.class_filter_system.classifiers.logistic_regression import (
    LogisticRegressionClassFilter,
)


def get_class_filter(
    method: str,
    random_state: int = RANDOM_STATE,
    **overrides,
) -> ClassFilterBase:
    """
    Retorna um filtro de classes configurado.

    Parameters
    ----------
    method : str
        Nome do método (ex.: ``"logistic_regression"``).
    random_state : int
        Semente para reprodutibilidade.
    **overrides
        Sobrescreve hiperparâmetros padrão do método.
    """
    if method not in CLASS_FILTER_STRATEGIES:
        raise ValueError(
            f"Método de filtro desconhecido: '{method}'. "
            f"Disponíveis: {list(CLASS_FILTER_STRATEGIES.keys())}"
        )

    params = {**CLASS_FILTER_STRATEGIES[method], **overrides}
    params.pop("description", None)

    if method == "logistic_regression":
        return LogisticRegressionClassFilter(random_state=random_state, **params)

    raise ValueError(f"Método '{method}' registrado mas não implementado.")

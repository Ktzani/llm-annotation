"""
Fábrica de seletores de instâncias.

Espelha o ``get_selector`` do repositório bio-is. Novos métodos (CNN, ENN,
DROP3, e2sc, ...) podem ser registrados aqui sem alterar o restante do pipeline.
"""
from src.config.instance_selection import INSTANCE_SELECTION_STRATEGIES, RANDOM_STATE
from src.instance_selection_system.selection.biois import BIOIS


def get_selector(method: str, random_state: int = RANDOM_STATE, **overrides):
    """
    Retorna um seletor de instâncias configurado.

    Parameters
    ----------
    method : str
        Nome do método (ex.: ``"bio-is"``).
    random_state : int
        Semente para reprodutibilidade.
    **overrides
        Sobrescreve hiperparâmetros padrão do método (ex.: ``beta=0.3``).
    """
    if method not in INSTANCE_SELECTION_STRATEGIES:
        raise ValueError(
            f"Método de seleção desconhecido: '{method}'. "
            f"Disponíveis: {list(INSTANCE_SELECTION_STRATEGIES.keys())}"
        )

    params = {**INSTANCE_SELECTION_STRATEGIES[method], **overrides}

    if method == "bio-is":
        return BIOIS(
            beta=params["beta"],
            theta=params["theta"],
            random_state=random_state,
        )

    raise ValueError(f"Método '{method}' registrado mas não implementado.")

"""
Configuração da filtragem por Seleção de Instâncias (Instance Selection).

Implementa o framework biO-IS (Noise-Oriented and Redundancy-Aware Instance
Selection), de Cunha et al., adaptado para filtrar os dados anotados pelas LLMs
(coluna de consenso agregado `resolved_annotation`) antes do fine-tuning.

Referência:
    Repositório: https://github.com/waashk/bio-is
"""

# -----------------------------------------------------------------------------
# Hiperparâmetros por método de seleção de instâncias
# -----------------------------------------------------------------------------
INSTANCE_SELECTION_STRATEGIES = {
    "bio-is": {
        "description": (
            "Framework bi-objetivo: remove instâncias redundantes (beta) via "
            "classificador fraco calibrado (Regressão Logística) e instâncias "
            "ruidosas (theta) via critério de entropia das probabilidades."
        ),
        # Taxa de remoção de redundância (fração do total de instâncias rotuladas).
        "beta": 0.25,
        # Taxa de remoção de ruído (fração das instâncias que o classificador
        # fraco classifica ERRADO).
        "theta": 0.50,
    },
}

# Método padrão (corresponde a `biois.BIOIS(beta=0.25, theta=0.50)` do repo).
DEFAULT_IS_METHOD = "bio-is"

# -----------------------------------------------------------------------------
# Pré-processamento textual (TF-IDF) — espelha o pré-processamento do bio-is:
# remoção de stopwords (lista padrão do scikit-learn) e descarte de termos que
# aparecem em menos de `min_df` documentos.
# -----------------------------------------------------------------------------
TFIDF_CONFIG = {
    "stop_words": "english",
    "min_df": 2,
    "sublinear_tf": True,
    "norm": "l2",
}

# -----------------------------------------------------------------------------
# Colunas esperadas no dataset anotado (genéricas para todo o framework)
# -----------------------------------------------------------------------------
TEXT_COLUMN = "text"
LABEL_COLUMN = "resolved_annotation"
ID_COLUMN = "text_id"

# Rótulo considerado inválido (LLM respondeu fora das categorias configuradas).
INVALID_LABEL = -1

# Semente para reprodutibilidade (o biO-IS é estocástico).
RANDOM_STATE = 42

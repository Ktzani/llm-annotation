"""
Configuração do Filtro de Classes (Fase 1 da anotação em 2 fases).

Um classificador leve, treinado nos dados anotados (nesta versão, o ground-truth
do conjunto de treino), reduz o espaço de classes apresentado ao LLM: em vez de
todas as classes, apenas as top-k mais prováveis por texto. O LLM permanece
zero-shot — a única mudança é o tamanho da lista de classes.

Toda a supervisão vive aqui, no classificador. Nenhuma probabilidade, ranking ou
exemplo rotulado é injetado no prompt.
"""

# -----------------------------------------------------------------------------
# Estratégias de filtro de classes (plugáveis via factory.get_class_filter)
# -----------------------------------------------------------------------------
CLASS_FILTER_STRATEGIES = {
    "logistic_regression": {
        "description": (
            "Regressão Logística sobre TF-IDF. Devolve as top-k classes mais "
            "prováveis por texto (predict_proba). Primeira implementação do "
            "filtro plugável."
        ),
        # Hiperparâmetros do LogisticRegression (mesmos do classificador fraco do
        # biO-IS, para consistência com o restante do repositório).
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
    },
}

# Método padrão do filtro.
DEFAULT_CLASS_FILTER_METHOD = "logistic_regression"

# k padrão (nº de classes candidatas apresentadas ao LLM na Fase 2).
DEFAULT_K = 3

# k's varridos no relatório de recall@k (validação — não envolve o LLM).
DEFAULT_K_SWEEP = [2, 3, 4, 5]

# Divisão interna do conjunto de treino (Fase 1): nº de pedaços e qual fica
# held-out (o LLM anota esse pedaço; o classificador treina nos demais).
DEFAULT_N_INNER_FOLDS = 5
DEFAULT_HOLDOUT_FOLD_INDEX = 0

# Padrão dos arquivos de fold externo no HF Hub (mesma convenção do fine-tuning).
DEFAULT_TRAIN_FOLD_PATTERN = "train_fold_{fold}.parquet"

# Semente para reprodutibilidade (split interno e classificador).
RANDOM_STATE = 42

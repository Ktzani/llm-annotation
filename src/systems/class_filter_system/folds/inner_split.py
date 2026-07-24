"""
Divisão interna do conjunto de treino de um fold externo.

O treino é dividido em `n_inner_folds` pedaços; um pedaço designado
(`holdout_fold_index`) fica de fora (held-out) e os demais formam o conjunto de
treino do classificador da Fase 1. O held-out é o que o LLM anota e onde o
recall@k é medido — sem vazamento, pois o classificador nunca o vê.

Usa `StratifiedKFold` (semeado), caindo para `KFold` quando alguma classe é rara
demais para estratificar.
"""
from typing import Tuple

import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold


def inner_split(
    df_train: pd.DataFrame,
    n_inner_folds: int,
    holdout_fold_index: int,
    random_state: int,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide `df_train` em (fit_part, holdout_part).

    Parameters
    ----------
    df_train : DataFrame
        Conjunto de treino do fold externo (com colunas de texto e rótulo).
    n_inner_folds : int
        Número de pedaços internos.
    holdout_fold_index : int
        Índice (0-based) do pedaço usado como held-out.
    random_state : int
        Semente do split.
    label_column : str
        Coluna de rótulo usada para estratificar.

    Returns
    -------
    (df_fit, df_holdout) : Tuple[DataFrame, DataFrame]
        Ambos com o índice ORIGINAL de `df_train` preservado (reset_index à parte
        fica a cargo do chamador).
    """
    if not 0 <= holdout_fold_index < n_inner_folds:
        raise ValueError(
            f"holdout_fold_index={holdout_fold_index} fora do range "
            f"[0, {n_inner_folds - 1}]."
        )

    y = df_train[label_column].to_numpy()

    min_class_count = df_train[label_column].value_counts().min()
    if min_class_count >= n_inner_folds:
        splitter = StratifiedKFold(
            n_splits=n_inner_folds, shuffle=True, random_state=random_state
        )
        splits = splitter.split(df_train, y)
        logger.info(
            f"Split interno estratificado: {n_inner_folds} pedaços "
            f"(classe mais rara tem {min_class_count} exemplos)"
        )
    else:
        splitter = KFold(
            n_splits=n_inner_folds, shuffle=True, random_state=random_state
        )
        splits = splitter.split(df_train)
        logger.warning(
            f"Classe mais rara tem {min_class_count} < {n_inner_folds} exemplos — "
            f"caindo para KFold (sem estratificação)."
        )

    fit_idx, holdout_idx = None, None
    for i, (fit_i, holdout_i) in enumerate(splits):
        if i == holdout_fold_index:
            fit_idx, holdout_idx = fit_i, holdout_i
            break

    df_fit = df_train.iloc[fit_idx]
    df_holdout = df_train.iloc[holdout_idx]

    logger.info(
        f"Split interno: fit={len(df_fit)} textos | held-out={len(df_holdout)} textos"
    )
    return df_fit, df_holdout

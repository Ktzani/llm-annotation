"""
Recall@k — teto da Fase 2.

`recall@k` é a fração de textos cujo rótulo verdadeiro está entre as k classes
mais prováveis do classificador da Fase 1. Como o LLM só vê essas k classes, o
que o filtro descarta é irrecuperável: `recall@k` é o teto de acerto da Fase 2.

Reportamos DUAS versões:
- `recall_at_k` (MICRO): fração de instâncias — teto da **accuracy**. Ponderado
  por instância, então as classes majoritárias dominam.
- `recall_at_k_macro` (MACRO): recall@k por classe, depois média entre classes —
  teto do **f1-macro**. Em dataset desbalanceado é a versão coerente (cada classe
  pesa igual).

Tudo aqui é medido no MESMO held-out e com o MESMO `predict_proba` que geram as
candidatas do LLM — senão o "teto" não corresponde ao que o LLM enfrenta.
"""
from typing import List

import numpy as np
import pandas as pd


def recall_at_k_sweep(
    y_true: List[int],
    proba: np.ndarray,
    classes: List[int],
    k_values: List[int],
) -> pd.DataFrame:
    """
    Calcula recall@k para vários k a partir de uma única matriz `predict_proba`.

    Parameters
    ----------
    y_true : lista de rótulos verdadeiros (canônicos) do held-out.
    proba : matriz (n_texts x n_classes), colunas alinhadas a `classes`.
    classes : rótulos canônicos na ordem das colunas de `proba`.
    k_values : lista de k's a avaliar.

    Returns
    -------
    DataFrame com colunas: k, recall_at_k (micro), recall_at_k_macro, n_texts.
    """
    y_true = np.asarray(y_true)
    classes = np.asarray(classes)
    n_texts = len(y_true)

    # posição da coluna correspondente ao rótulo verdadeiro de cada texto
    class_to_col = {int(c): j for j, c in enumerate(classes)}

    # ranking das classes por probabilidade (desc) para cada texto
    ranking = np.argsort(proba, axis=1)[:, ::-1]  # (n_texts x n_classes) de índices de coluna

    gt_classes = sorted(set(int(v) for v in y_true))  # classes presentes no held-out

    rows = []
    n_classes = len(classes)
    for k in k_values:
        k_eff = min(k, n_classes)
        topk_cols = ranking[:, :k_eff]  # colunas das top-k por texto

        # hit por instância: rótulo verdadeiro (não visto no treino → nunca cai no top-k)
        hit = np.zeros(n_texts, dtype=bool)
        for i in range(n_texts):
            true_col = class_to_col.get(int(y_true[i]))
            if true_col is not None and true_col in topk_cols[i]:
                hit[i] = True

        micro = hit.mean() if n_texts else 0.0
        # MACRO: recall@k por classe, média entre as classes presentes
        per_class = [hit[y_true == c].mean() for c in gt_classes]
        macro = float(np.mean(per_class)) if per_class else 0.0

        rows.append({
            "k": k,
            "recall_at_k": micro,          # micro (teto da accuracy)
            "recall_at_k_macro": macro,    # macro (teto do f1-macro)
            "n_texts": n_texts,
        })

    return pd.DataFrame(rows)

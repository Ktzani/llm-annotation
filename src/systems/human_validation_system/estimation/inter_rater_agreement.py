"""
Inter Rater Agreement - Concordância entre avaliadores (kappa de Fleiss e acordo bruto)
"""
from itertools import combinations
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


class InterRaterAgreement:
    """Calcula kappa de Fleiss e acordo bruto entre avaliadores sobre os documentos completos."""

    def __init__(self, n_evaluators: int):
        self.n_evaluators = n_evaluators
        logger.debug("InterRaterAgreement inicializado")

    def _table(self, responses: pd.DataFrame, doc_ids) -> pd.DataFrame:
        """Documento x avaliador com o rótulo escolhido (só documentos completos)."""
        subset = responses[responses["id_anonimo"].isin(doc_ids)].dropna(subset=["rotulo_escolhido"])
        return subset.pivot(index="id_anonimo", columns="avaliador", values="rotulo_escolhido").dropna()

    def raw(self, responses: pd.DataFrame, doc_ids) -> Dict[str, float]:
        """Fração de documentos unânimes e acordo médio entre pares de avaliadores."""
        table = self._table(responses, doc_ids)
        if table.empty:
            return {"acordo_unanime": np.nan, "acordo_par_a_par": np.nan}
        pairwise = np.mean([(table[a] == table[b]).mean() for a, b in combinations(table.columns, 2)])
        return {"acordo_unanime": float((table.nunique(axis=1) == 1).mean()), "acordo_par_a_par": float(pairwise)}

    def fleiss(self, responses: pd.DataFrame, doc_ids) -> float:
        """Kappa de Fleiss do rótulo escolhido nos documentos indicados (NaN se < 2)."""
        table = self._table(responses, doc_ids)
        if len(table) < 2:
            return np.nan
        codes = table.apply(lambda col: pd.Categorical(col, categories=sorted(set(table.values.ravel()))).codes)
        counts, _ = aggregate_raters(codes.to_numpy())
        # Indefinido (NaN) quando todos usam um único rótulo: concordância esperada = 1
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(fleiss_kappa(counts))

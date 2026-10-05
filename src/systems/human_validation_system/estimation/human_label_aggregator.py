"""
Human Label Aggregator - Consolida os avaliadores por voto majoritário
"""
import numpy as np
import pandas as pd
from loguru import logger


class HumanLabelAggregator:
    """
    Consolida as respostas dos avaliadores em um valor por documento.

    Responsabilidades:
    - Considerar só documentos respondidos por todos os avaliadores
      (rodadas parcialmente preenchidas não enviesam a estimativa)
    - Rótulo humano = rótulo de mais da metade dos avaliadores; sem maioria fica NaN
    - Calcular as métricas por documento (0/1, NaN só para documentos incompletos);
      sem maioria conta como 0: o rótulo não foi confirmado pelos humanos
    """

    def __init__(self, n_evaluators: int):
        self.n_evaluators = n_evaluators
        logger.debug(f"HumanLabelAggregator inicializado ({n_evaluators} avaliadores)")

    def _majority(self, values: pd.Series):
        counts = values.value_counts()
        return counts.index[0] if len(counts) and counts.iloc[0] > self.n_evaluators / 2 else np.nan

    def aggregate(self, responses: pd.DataFrame, answer_key: pd.DataFrame) -> pd.DataFrame:
        """Gabarito com rótulo humano, status e métricas por documento."""
        answered = responses.dropna(subset=["rotulo_escolhido"])
        by_doc = answered.groupby("id_anonimo")
        summary = pd.DataFrame({
            "n_respostas": by_doc["avaliador"].nunique(),
            "rotulo_humano": by_doc["rotulo_escolhido"].agg(self._majority),
            "outro_rotulo_maioria": by_doc["outro_rotulo_possivel"].agg(self._majority),
        })

        docs = answer_key.merge(summary, left_on="id_anonimo", right_index=True, how="left")
        docs["n_respostas"] = docs["n_respostas"].fillna(0).astype(int)
        docs["completo"] = docs["n_respostas"] == self.n_evaluators

        complete = docs["completo"]
        docs["acerto_referencia"] = np.where(complete, docs["rotulo_humano"] == docs["classe_referencia"], np.nan)
        docs["acerto_llm"] = np.where(complete, docs["rotulo_humano"] == docs["rotulo_consolidado"], np.nan)
        has_other = docs["completo"] & docs["outro_rotulo_maioria"].notna()
        docs["outro_rotulo_possivel"] = np.where(has_other, docs["outro_rotulo_maioria"] == "sim", np.nan)

        pending = int((~docs["completo"]).sum())
        no_majority = int((docs["completo"] & docs["rotulo_humano"].isna()).sum())
        logger.info(
            f"Documentos: {len(docs)} | completos: {int(docs['completo'].sum())} | "
            f"pendentes: {pending} | sem maioria: {no_majority}"
        )
        return docs

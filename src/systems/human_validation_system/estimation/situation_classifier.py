"""
Situation Classifier - Classifica cada documento nas quatro situações do RQ4
"""
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


class SituationClassifier:
    """
    Deriva a situação de cada documento a partir das respostas dos avaliadores.

    Apoio de um rótulo = nº de avaliadores que o escolheram ou o indicaram como
    outro rótulo possível. O "lado oposto do conflito" é o rótulo das LLMs quando
    a maioria escolheu a referência, e a referência caso contrário; se referência
    e LLMs coincidem (grupo C), é o alternativo com mais apoio.

    Regras (em ordem):
    - insufficient_information: maioria escolheu a opção de informação insuficiente
    - genuine_ambiguity: sem maioria (cada avaliador escolheu um rótulo) ou
      lado oposto do conflito com apoio da maioria
    - benchmark_correct: maioria = referência
    - benchmark_mislabeling: maioria = outra classe
    """

    SITUATIONS = ("benchmark_correct", "benchmark_mislabeling", "genuine_ambiguity", "insufficient_information")

    def __init__(self, n_evaluators: int, insufficient_option: str):
        self.min_support = n_evaluators // 2 + 1
        self.insufficient_option = insufficient_option
        logger.debug(f"SituationClassifier inicializado (apoio mínimo={self.min_support})")

    @staticmethod
    def _support(answers: pd.DataFrame) -> Counter:
        """Apoio por rótulo, contando cada avaliador no máximo uma vez por rótulo."""
        support = Counter()
        for _, a in answers.iterrows():
            labels = {a["rotulo_escolhido"]}
            if a["outro_rotulo_possivel"] == "sim" and pd.notna(a["qual_outro_rotulo"]):
                labels.add(a["qual_outro_rotulo"])
            support.update(l for l in labels if pd.notna(l))
        return support

    def _opposite(self, majority: str, reference: str, consolidated: str, support: Counter) -> Optional[str]:
        if reference != consolidated:
            return consolidated if majority == reference else reference
        others = [(n, l) for l, n in support.items() if l not in (majority, self.insufficient_option)]
        return max(others)[1] if others else None

    def _classify_doc(self, doc: pd.Series, answers: pd.DataFrame) -> Dict:
        if not doc["completo"]:
            return {"situacao": np.nan, "rotulo_oposto": np.nan, "apoio_oposto": np.nan}
        if doc["rotulo_humano"] == self.insufficient_option:
            return {"situacao": "insufficient_information", "rotulo_oposto": np.nan, "apoio_oposto": np.nan}
        if pd.isna(doc["rotulo_humano"]):
            return {"situacao": "genuine_ambiguity", "rotulo_oposto": np.nan, "apoio_oposto": np.nan}

        support = self._support(answers)
        majority = doc["rotulo_humano"]
        opposite = self._opposite(majority, doc["classe_referencia"], doc["rotulo_consolidado"], support)
        opposite_support = support.get(opposite, 0) if opposite else 0

        if opposite and opposite_support >= self.min_support:
            situation = "genuine_ambiguity"
        elif majority == doc["classe_referencia"]:
            situation = "benchmark_correct"
        else:
            situation = "benchmark_mislabeling"
        return {"situacao": situation, "rotulo_oposto": opposite, "apoio_oposto": opposite_support}

    @staticmethod
    def _observations(answers: pd.DataFrame) -> str:
        notes: List[str] = [
            f"{a['avaliador']}: {str(a['observacao']).strip()}"
            for _, a in answers.iterrows()
            if pd.notna(a.get("observacao")) and str(a["observacao"]).strip()
        ]
        return " | ".join(notes)

    def classify(self, responses: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
        """Acrescenta situação, lado oposto do conflito, observações e indicadores 0/1 por situação."""
        by_doc = {i: g for i, g in responses.groupby("id_anonimo")}
        empty = responses.head(0)

        rows = []
        for _, doc in docs.iterrows():
            answers = by_doc.get(doc["id_anonimo"], empty)
            rows.append({**self._classify_doc(doc, answers), "observacoes": self._observations(answers)})
        out = pd.concat([docs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        for situation in self.SITUATIONS:
            out[situation] = np.where(out["situacao"].notna(), out["situacao"] == situation, np.nan)

        counts = out["situacao"].value_counts().to_dict()
        logger.info(f"Situações (documentos completos): {counts}")
        return out

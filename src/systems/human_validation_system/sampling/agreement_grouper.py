"""
Agreement Grouper - Classifica cada documento no padrão de concordância A/B/C
"""
from typing import List

import pandas as pd
from loguru import logger


class AgreementGrouper:
    """
    Classifica documentos pelo padrão de concordância entre LLMs e referência.

    Responsabilidades:
    - Detectar as colunas de rótulo por LLM (`<modelo>_consensus`)
    - Descartar documentos com resposta inválida de alguma LLM
    - Atribuir o grupo A, B ou C (demais padrões, inclusive 1x1x1, ficam fora da população)
    """

    def __init__(
        self,
        label_suffix: str = "_consensus",
        reference_column: str = "ground_truth",
        consolidated_column: str = "resolved_annotation",
        invalid_label: int = -1,
    ):
        self.label_suffix = label_suffix
        self.reference_column = reference_column
        self.consolidated_column = consolidated_column
        self.invalid_label = invalid_label
        logger.debug(f"AgreementGrouper inicializado (sufixo='{label_suffix}')")

    def detect_llm_columns(self, df: pd.DataFrame) -> List[str]:
        """Colunas `<modelo>_consensus` acompanhadas de `<modelo>_consensus_score`."""
        columns = [
            c for c in df.columns
            if c.endswith(self.label_suffix) and c != self.label_suffix and f"{c}_score" in df.columns
        ]
        if len(columns) != 3:
            raise ValueError(f"Esperadas 3 colunas de rótulo por LLM, encontradas {len(columns)}: {columns}")
        return columns

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna apenas os documentos dos grupos A/B/C, com a coluna `grupo`."""
        llm_columns = self.detect_llm_columns(df)
        label_columns = llm_columns + [self.reference_column, self.consolidated_column]

        if df["text_id"].duplicated().any():
            raise ValueError("text_id duplicado no dataset de consenso")

        labels = df[label_columns].apply(pd.to_numeric, errors="coerce")
        valid = labels.notna().all(axis=1) & (labels[llm_columns] != self.invalid_label).all(axis=1)
        logger.info(f"Descartados {int((~valid).sum())} documentos com resposta inválida de alguma LLM")

        out = df.loc[valid, ["text_id", "text"]].copy()
        out[label_columns] = labels.loc[valid].astype(int)

        llm = out[llm_columns]
        all_same = llm.nunique(axis=1) == 1
        n_agree = llm.eq(out[self.reference_column], axis=0).sum(axis=1)

        out["grupo"] = None
        out.loc[all_same & (n_agree == 0), "grupo"] = "A"
        # B é 2x1: as duas LLMs que divergem da referência concordam entre si
        out.loc[(n_agree == 1) & (llm.nunique(axis=1) == 2), "grupo"] = "B"
        out.loc[all_same & (n_agree == 3), "grupo"] = "C"

        logger.info(f"Fora dos grupos A/B/C: {int(out['grupo'].isna().sum())} documentos")
        out = out.dropna(subset=["grupo"]).reset_index(drop=True)
        logger.info(f"Grupos: {out['grupo'].value_counts().sort_index().to_dict()}")
        return out

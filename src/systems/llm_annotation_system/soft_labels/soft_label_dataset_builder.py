"""Soft Label Dataset Builder - Monta o dataset de soft labels (distribuição dos votos das LLMs por texto)"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from src.config.datasets_collected import LABEL_MEANINGS
from src.systems.llm_annotation_system.perspectivism.perspectivism_dataset_builder import PerspectivismDatasetBuilder
from src.utils.data_loader import add_label_description
from src.utils.get_text_id_from_text import get_text_id_from_text


class SoftLabelDatasetBuilder:
    """
    Gera uma linha por texto com a distribuição dos votos válidos das LLMs sobre as classes
    Responsabilidades: detectar as colunas de voto, contar os votos válidos por classe e normalizar em `soft_label`
    """

    def __init__(self, dataset_name: str, label_suffix: str = "_consensus", invalid_label: int = -1):
        self.dataset_name = dataset_name
        self.label_suffix = label_suffix
        self.invalid_label = invalid_label
        self.classes: List[int] = sorted(int(k) for k in LABEL_MEANINGS[dataset_name])
        logger.debug(f"SoftLabelDatasetBuilder: {dataset_name} | classes={self.classes}")

    def vote_columns(self, df: pd.DataFrame) -> List[str]:
        return PerspectivismDatasetBuilder(self.dataset_name, self.label_suffix).detect_llm_label_columns(df)

    def build(self, df_consensus: pd.DataFrame) -> pd.DataFrame:
        """Colunas: text_id, text, label (voto majoritário), soft_label (lista por classe), label_description"""
        votes = df_consensus[self.vote_columns(df_consensus)].to_numpy()
        counts = np.stack([(votes == c).sum(axis=1) for c in self.classes], axis=1).astype(float)
        totals = counts.sum(axis=1, keepdims=True)

        # Sem nenhum voto válido não há distribuição: o texto fica de fora
        has_votes = totals[:, 0] > 0
        df = df_consensus.loc[has_votes, ["text", "resolved_annotation"]].rename(columns={"resolved_annotation": "label"})
        df["text"] = df["text"].astype(str).str.strip()
        df["text_id"] = df["text"].apply(get_text_id_from_text)
        df["soft_label"] = (counts[has_votes] / totals[has_votes]).tolist()

        df = add_label_description(df.reset_index(drop=True), dataset_name=self.dataset_name)
        logger.info(f"Soft labels: {len(df)} textos ({(~has_votes).sum()} sem voto válido descartados)")
        return df[["text_id", "text", "label", "soft_label", "label_description"]]

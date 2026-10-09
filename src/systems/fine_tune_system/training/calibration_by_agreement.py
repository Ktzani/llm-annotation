"""Calibration By Agreement - Desempenho e calibração do classificador por padrão de concordância das LLMs (3x0 × 2x1)"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

from src.systems.fine_tune_system.training.calibration import (
    PREDICTIONS_FILE,
    brier_score,
    expected_calibration_error,
)
from src.utils.get_text_id_from_text import get_text_id_from_text


class CalibrationByAgreement:
    """
    Mede macro-F1, acurácia, ECE e Brier de cada fold separadamente por estrato de concordância das LLMs
    Responsabilidades: mapear text_id → estrato pelo consenso, ler as predições dos folds e agregar média/desvio por estrato
    """

    STRATA = {3: "3x0", 2: "2x1"}
    OTHER_STRATUM = "1x1x1"  # discordância total: fora do dataset de consenso
    INVALID_VOTE_STRATUM = "voto_invalido"  # alguma LLM votou -1: não é divergência real (fora do treino)
    FOLD_PATTERN = re.compile(r"fold_(\d+)")

    def __init__(self, agreement: pd.Series):
        self.agreement = agreement  # text_id → estrato
        logger.debug(f"CalibrationByAgreement: {agreement.value_counts().to_dict()}")

    @classmethod
    def from_consensus(cls, df_consensus: pd.DataFrame, vote_columns: List[str]) -> "CalibrationByAgreement":
        text_ids = df_consensus["text"].astype(str).str.strip().apply(get_text_id_from_text)
        strata = df_consensus["most_common_count"].map(cls.STRATA).fillna(cls.OTHER_STRATUM)
        strata[(df_consensus[vote_columns] == -1).any(axis=1)] = cls.INVALID_VOTE_STRATUM
        return cls(pd.Series(strata.to_numpy(), index=text_ids.to_numpy()))

    def evaluate(self, output_dir: Path) -> Dict:
        """{"folds": {fold: {estrato: métricas}}, "cv": {estrato: {métrica: {mean, std}}}}"""
        folds = {
            self._fold_index(path): self._evaluate_fold(pd.read_csv(path))
            for path in sorted(Path(output_dir).rglob(PREDICTIONS_FILE))
        }
        return {"folds": folds, "cv": self._aggregate(folds)}

    def _fold_index(self, path: Path) -> int:
        match = self.FOLD_PATTERN.search(path.parent.name)
        return int(match.group(1)) if match else 0

    def _evaluate_fold(self, predictions: pd.DataFrame) -> Dict:
        prob_cols = [c for c in predictions.columns if c.startswith("prob_")]
        strata = predictions["text_id"].map(self.agreement).fillna(self.OTHER_STRATUM)

        metrics = {}
        for stratum, group in predictions.groupby(strata):
            probs = group[prob_cols].to_numpy()
            labels = group["label"].to_numpy().astype(int)
            preds = probs.argmax(axis=1)
            metrics[stratum] = {
                "n": int(len(group)),
                "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(labels, preds)),
                "ece": expected_calibration_error(probs, labels),
                "brier": brier_score(probs, labels),
            }
        return metrics

    @staticmethod
    def _aggregate(folds: Dict) -> Dict:
        strata = sorted({s for fold in folds.values() for s in fold})
        aggregated = {}
        for stratum in strata:
            per_fold = [fold[stratum] for fold in folds.values() if stratum in fold]
            aggregated[stratum] = {
                metric: {
                    "mean": float(np.mean([m[metric] for m in per_fold])),
                    "std": float(np.std([m[metric] for m in per_fold])),
                }
                for metric in per_fold[0]
            }
        return aggregated

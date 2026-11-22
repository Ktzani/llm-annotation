"""Control Target Builders - Alvos de controle da RQ6 (label smoothing e minoria embaralhada) a partir da distribuição dos votos"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class ControlTargetBuilder:
    """
    Base dos controles: mantém a fração da classe majoritária e redistribui o restante (a minoria) entre as demais classes
    Responsabilidades: validar a maioria estrita (3x0/2x1), separar majoritária e restante e delegar a redistribuição à subclasse
    """

    name = "control"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe o dataset de soft labels (coluna `soft_label`) e devolve o mesmo dataset com o alvo de controle"""
        dist = np.array(df["soft_label"].tolist(), dtype=float)
        majority = dist.argmax(axis=1)
        share = dist.max(axis=1)
        if (share <= 0.5).any():
            raise ValueError(f"{self.name}: {(share <= 0.5).sum()} textos sem maioria estrita (1x1x1 deveriam ter sido removidos)")

        rest = 1.0 - share
        minority_rows = rest > 1e-9  # 2x1; nos 3x0 o alvo é igual ao das demais condições
        target = np.zeros_like(dist)
        target[np.arange(len(dist)), majority] = share
        target[minority_rows] += self.redistribute(df[minority_rows], dist[minority_rows], majority[minority_rows], rest[minority_rows])

        logger.info(f"{self.name}: {minority_rows.sum()} textos 2x1 redistribuídos, {(~minority_rows).sum()} 3x0 inalterados")
        return df.assign(soft_label=target.tolist())

    def redistribute(self, df: pd.DataFrame, dist: np.ndarray, majority: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Massa do restante por classe (zero na majoritária) para as linhas 2x1"""
        raise NotImplementedError


class LabelSmoothingTargetBuilder(ControlTargetBuilder):
    """Label smoothing: o restante é dividido igualmente entre todas as outras classes (inclusive a da minoria real)"""

    name = "label_smoothing"

    def redistribute(self, df, dist, majority, rest):
        n_classes = dist.shape[1]
        mass = np.repeat((rest / (n_classes - 1))[:, None], n_classes, axis=1)
        mass[np.arange(len(dist)), majority] = 0.0
        return mass


class ShuffledMinorityTargetBuilder(ControlTargetBuilder):
    """
    Minoria embaralhada: o restante vai para uma única classe sorteada entre as não majoritárias
    Responsabilidades: sortear por documento com semente fixa (independe da ordem e do conjunto) e salvar/reutilizar o mapeamento texto → classe sorteada
    """

    name = "shuffled_minority"

    def __init__(self, classes: list[int], seed: int, mapping_path: Optional[Path] = None):
        self.classes = list(classes)
        self.seed = seed
        self.mapping_path = Path(mapping_path) if mapping_path else None

    def draw(self, text_id: str, majority: int, n_classes: int) -> int:
        """Índice sorteado uniformemente entre as classes ≠ majoritária, com RNG semeado por (seed, text_id)"""
        rng = np.random.default_rng([self.seed, int(text_id[:8], 16)])
        candidates = [c for c in range(n_classes) if c != majority]
        return int(rng.choice(candidates))

    def redistribute(self, df, dist, majority, rest):
        n_classes = dist.shape[1]
        saved = self._load_mapping()
        drawn = np.array([
            self.classes.index(saved[tid]) if tid in saved else self.draw(tid, m, n_classes)
            for tid, m in zip(df["text_id"], majority)
        ])
        if (drawn == majority).any():
            raise ValueError(f"{self.name}: mapeamento salvo sorteou a classe majoritária")

        if not saved:  # retomada: o mapeamento original é mantido intacto
            self._save_mapping(df, dist, majority, drawn)
        mass = np.zeros_like(dist)
        mass[np.arange(len(dist)), drawn] = rest
        return mass

    def _load_mapping(self) -> dict:
        if self.mapping_path is None or not self.mapping_path.exists():
            return {}
        logger.info(f"Mapeamento da minoria embaralhada reutilizado: {self.mapping_path}")
        mapping = pd.read_csv(self.mapping_path)
        return dict(zip(mapping["text_id"], mapping["drawn_label"]))

    def _save_mapping(self, df, dist, majority, drawn) -> None:
        """Uma linha por texto 2x1: majoritária, minoria real e classe sorteada (coincidências mantidas, sem tratamento especial)"""
        masked = dist.copy()
        masked[np.arange(len(dist)), majority] = -1.0
        original = masked.argmax(axis=1)
        mapping = pd.DataFrame({
            "text_id": df["text_id"].to_numpy(),
            "majority_label": [self.classes[i] for i in majority],
            "original_minority_label": [self.classes[i] for i in original],
            "drawn_label": [self.classes[i] for i in drawn],
        })
        mapping["drawn_is_original"] = mapping["drawn_label"] == mapping["original_minority_label"]
        logger.info(f"Minoria embaralhada: classe sorteada = minoria real em {mapping['drawn_is_original'].sum()}/{len(mapping)} textos (acaso)")
        if self.mapping_path is not None:
            mapping.to_csv(self.mapping_path, index=False, encoding="utf-8")
            logger.info(f"Mapeamento salvo em: {self.mapping_path} (seed={self.seed})")

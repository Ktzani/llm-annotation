from typing import List, Dict, Any, Optional
import pandas as pd
from collections import Counter
from loguru import logger

from src.config.conflict_resolution import CONFLICT_RESOLUTION_STRATEGIES


class ConsensusCalculator:
    """
    Calcula consenso e aplica estratégias de resolução de conflitos
    conforme definido nas configurações globais.
    """

    def __init__(
        self,
        consensus_threshold: float,
        default_strategy: str = "majority_vote"
    ):
        self.consensus_threshold = consensus_threshold
        self.conflict_resolution_strategies = CONFLICT_RESOLUTION_STRATEGIES
        self.default_strategy = default_strategy


    def _extract_consensus_columns(self, df: pd.DataFrame) -> List[str]:
        consensus_cols = [
            col for col in df.columns
            if "_consensus" in col and "_score" not in col
        ]
        if not consensus_cols:
            raise ValueError("Nenhuma coluna '_consensus' encontrada.")
        return consensus_cols

    def _classify_consensus_level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        if score >= 0.6:
            return "medium"
        return "low"

    def _is_problematic(self, annotations: List[Any]) -> bool:
        counter = Counter(annotations)
        counts = sorted(counter.values(), reverse=True)
        return len(counts) >= 2 and counts[0] == counts[1]

    # -------------------------------------------------------------------------
    # ESTRATÉGIAS DE RESOLUÇÃO DE CONFLITOS
    # -------------------------------------------------------------------------
    def _apply_majority_vote(self, annotations: List[Any]) -> Any:
        return Counter(annotations).most_common(1)[0][0]

    def _apply_weighted_vote(self, annotations: List[Any], model_names: List[str]) -> Any:
        strategy = self.conflict_resolution_strategies["weighted_vote"]
        weights = strategy.get("weights", {})

        weighted_counts = Counter()

        for ann, model in zip(annotations, model_names):
            weight = weights.get(model, 1.0)
            weighted_counts[ann] += weight

        return weighted_counts.most_common(1)[0][0]

    def _apply_unanimous_only(self, annotations: List[Any]) -> Optional[Any]:
        if len(set(annotations)) == 1:
            return annotations[0]
        return None

    def _apply_remove_outliers(self, annotations: List[Any]) -> Any:
        """
        Remove classes minoritárias se forem muito distantes
        (baseado em frequência relativa).
        """
        counter = Counter(annotations)
        total = len(annotations)
        threshold = self.conflict_resolution_strategies["remove_outliers"]["outlier_threshold"]

        filtered = [c for c, cnt in counter.items() if cnt / total >= threshold]
        if len(filtered) == 1:
            return filtered[0]
        return None  # ainda tem conflito

    # -------------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -------------------------------------------------------------------------

    def calculate_consensus(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculando consenso...")

        consensus_cols = self._extract_consensus_columns(df)
        df["all_annotations"] = df[consensus_cols].apply(list, axis=1)

        # Nome dos modelos capturados via prefixo da coluna
        model_names = [col.replace("_consensus", "") for col in consensus_cols]

        # Métricas básicas
        df["unique_annotations"] = df["all_annotations"].apply(lambda x: len(set(x)))
        df["most_common_annotation"] = df["all_annotations"].apply(
            lambda x: Counter(x).most_common(1)[0][0] if x else None
        )
        df["most_common_count"] = df["all_annotations"].apply(
            lambda x: Counter(x).most_common(1)[0][1] if x else 0
        )

        num_models = len(consensus_cols)
        df["consensus_score"] = df["most_common_count"] / num_models
        df["consensus_level"] = df["consensus_score"].apply(self._classify_consensus_level)

        df["is_problematic"] = df["all_annotations"].apply(self._is_problematic)

        # Apply strategy
        df["resolved_annotation"] = df.apply(
            lambda row: self._resolve_conflict(
                row["all_annotations"],
                model_names,
                row["consensus_score"]
            ),
            axis=1
        )

        # Flag para revisão manual (se estratégia determinar isso)
        df["needs_review"] = df["resolved_annotation"].isna()

        self._log_stats(df)
        return df

    # -------------------------------------------------------------------------
    # RESOLUÇÃO DE CONFLITO
    # -------------------------------------------------------------------------

    def _resolve_conflict(
        self,
        annotations: List[Any],
        model_names: List[str],
        score: float
    ) -> Optional[Any]:

        # Se já atingiu consenso, retorna a classe majoritária
        if score >= self.consensus_threshold:
            return Counter(annotations).most_common(1)[0][0]

        strategy_name = self.default_strategy
        strategy = self.conflict_resolution_strategies[strategy_name]

        # Escolhe estratégia
        if strategy_name == "majority_vote":
            return self._apply_majority_vote(annotations)

        elif strategy_name == "weighted_vote":
            return self._apply_weighted_vote(annotations, model_names)

        elif strategy_name == "unanimous_only":
            return self._apply_unanimous_only(annotations)

        elif strategy_name == "remove_outliers":
            return self._apply_remove_outliers(annotations)

        logger.warning(f"Estratégia desconhecida: {strategy_name}")
        return None

    def _log_stats(self, df: pd.DataFrame):
        high = (df["consensus_level"] == "high").sum()
        medium = (df["consensus_level"] == "medium").sum()
        low = (df["consensus_level"] == "low").sum()
        problematic = df["is_problematic"].sum()
        needs_review = df["needs_review"].sum()

        total = len(df)

        logger.success("Consenso calculado:")
        logger.info(f"  Alto (≥80%): {high} ({high/total:.1%})")
        logger.info(f"  Médio (60-80%): {medium} ({medium/total:.1%})")
        logger.info(f"  Baixo (<60%): {low} ({low/total:.1%})")
        logger.info(f"  Problemáticos: {problematic} ({problematic/total:.1%})")
        logger.info(f"  Itens que precisam de revisão: {needs_review} ({needs_review/total:.1%})")

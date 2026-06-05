"""
Filtragem de anotações por Seleção de Instâncias.

Interface de alto nível que aplica um método de seleção de instâncias (por
padrão, biO-IS) sobre um DataFrame anotado pelas LLMs, filtrando instâncias
redundantes e ruidosas com base na coluna de consenso agregado
(`resolved_annotation`).

Genérico para todo o framework: depende apenas das colunas de texto, de rótulo
agregado e de id — não dos modelos individuais que geraram a anotação.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from loguru import logger

from src.config.instance_selection import (
    DEFAULT_IS_METHOD,
    ID_COLUMN,
    INVALID_LABEL,
    LABEL_COLUMN,
    RANDOM_STATE,
    TEXT_COLUMN,
)
from src.is_system.core.text_vectorizer import TextVectorizer
from src.is_system.selection.selector_factory import get_selector


@dataclass
class FilterResult:
    """Resultado da filtragem por seleção de instâncias."""

    filtered_df: pd.DataFrame                 # instâncias mantidas (conjunto limpo)
    removed_df: pd.DataFrame                  # instâncias removidas (+ removal_reason)
    excluded_df: pd.DataFrame                 # excluídas da seleção (sem rótulo / inválidas)
    stats: dict = field(default_factory=dict)  # métricas da filtragem


class AnnotationFilter:
    """Aplica seleção de instâncias sobre um DataFrame anotado."""

    def __init__(
        self,
        method: str = DEFAULT_IS_METHOD,
        text_column: str = TEXT_COLUMN,
        label_column: str = LABEL_COLUMN,
        id_column: str = ID_COLUMN,
        invalid_label=INVALID_LABEL,
        drop_invalid: bool = True,
        random_state: int = RANDOM_STATE,
        **selector_overrides,
    ):
        self.method = method
        self.text_column = text_column
        self.label_column = label_column
        self.id_column = id_column
        self.invalid_label = invalid_label
        self.drop_invalid = drop_invalid
        self.random_state = random_state
        self.selector_overrides = selector_overrides

    # ------------------------------------------------------------------
    def filter(self, df: pd.DataFrame) -> FilterResult:
        self._validate(df)
        df = df.reset_index(drop=True)
        n_total = len(df)

        # 1. Separa instâncias sem rótulo válido (não entram na seleção).
        labeled, excluded = self._split_unlabeled(df)
        logger.info(
            f"{len(labeled)} instâncias rotuladas | {len(excluded)} excluídas "
            f"(sem consenso / rótulo inválido)"
        )

        if len(labeled) < 2 or labeled[self.label_column].nunique() < 2:
            logger.warning("Poucas instâncias/classes — filtragem ignorada.")
            stats = self._build_stats(n_total, labeled, labeled, labeled.iloc[:0], excluded, None, None)
            return FilterResult(labeled, labeled.iloc[:0].copy(), excluded, stats)

        # 2. Codifica os rótulos para 0..n-1 (exigido pelo algoritmo).
        encoder = LabelEncoder()
        y = encoder.fit_transform(labeled[self.label_column].astype(int).values)

        # 3. Vetoriza os textos (TF-IDF).
        X = TextVectorizer().fit_transform(labeled[self.text_column].astype(str).values)

        # 4. Seleção de instâncias.
        selector = get_selector(
            self.method, random_state=self.random_state, **self.selector_overrides
        )
        selector.fit(X, y)

        mask = selector.mask
        kept = labeled[mask].copy()
        removed = self._annotate_removal_reason(labeled, labeled[~mask].copy(), selector)

        stats = self._build_stats(n_total, labeled, kept, removed, excluded, selector, encoder)
        self._log_stats(stats)

        return FilterResult(
            filtered_df=kept,
            removed_df=removed,
            excluded_df=excluded,
            stats=stats,
        )

    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame) -> None:
        for col in (self.text_column, self.label_column):
            if col not in df.columns:
                raise ValueError(
                    f"Coluna obrigatória '{col}' ausente. Colunas: {list(df.columns)}"
                )
        if self.id_column not in df.columns:
            logger.warning(
                f"Coluna de id '{self.id_column}' ausente — relatório sem rastreio por id."
            )

    def _split_unlabeled(self, df: pd.DataFrame):
        label = df[self.label_column]
        invalid_mask = label.isna()
        if self.drop_invalid:
            invalid_mask = invalid_mask | (label == self.invalid_label)

        excluded = df[invalid_mask].copy()
        labeled = df[~invalid_mask].copy().reset_index(drop=True)
        return labeled, excluded

    def _annotate_removal_reason(self, labeled, removed, selector):
        reason = np.array(["kept"] * len(labeled), dtype=object)
        reason[getattr(selector, "_idx_redundant", [])] = "redundant"
        reason[getattr(selector, "_idx_noise", [])] = "noise"

        removed = removed.copy()
        removed["removal_reason"] = reason[removed.index.to_numpy()]
        return removed

    # ------------------------------------------------------------------
    def _build_stats(self, n_total, labeled, kept, removed, excluded, selector, encoder) -> dict:
        return {
            "method": self.method,
            "total_instances": int(n_total),
            "labeled_instances": int(len(labeled)),
            "excluded_instances": int(len(excluded)),
            "kept_instances": int(len(kept)),
            "removed_instances": int(len(removed)),
            "removed_redundant": int(len(getattr(selector, "_idx_redundant", []))),
            "removed_noise": int(len(getattr(selector, "_idx_noise", []))),
            # Redução relativa às instâncias rotuladas (definição do bio-is).
            "reduction_rate": float(getattr(selector, "reduction_", 0.0)),
            "n_classes": int(len(encoder.classes_)) if encoder is not None else 0,
            "weak_classifier": "LogisticRegression",
            "weak_classifier_accuracy": getattr(selector, "weak_clf_accuracy", None),
            "weak_classifier_f1_macro": getattr(selector, "weak_clf_f1_macro", None),
            "beta": getattr(selector, "beta", None),
            "theta": getattr(selector, "theta", None),
            "random_state": self.random_state,
        }

    def _log_stats(self, stats: dict) -> None:
        logger.success("Filtragem por seleção de instâncias concluída:")
        logger.info(f"  Total: {stats['total_instances']}")
        logger.info(f"  Rotuladas: {stats['labeled_instances']} | Excluídas: {stats['excluded_instances']}")
        logger.info(f"  Mantidas: {stats['kept_instances']} | Removidas: {stats['removed_instances']}")
        logger.info(f"    • Redundantes: {stats['removed_redundant']}")
        logger.info(f"    • Ruidosas:    {stats['removed_noise']}")
        logger.info(f"  Redução (sobre rotuladas): {stats['reduction_rate']:.2%}")
        if stats.get("weak_classifier_accuracy") is not None:
            logger.info(
                f"  Classificador fraco (LR) — acurácia: {stats['weak_classifier_accuracy']:.2%}"
                f" | F1-macro: {stats['weak_classifier_f1_macro']:.4f}"
            )


def save_filter_result(result: FilterResult, output_dir) -> None:
    """
    Salva os artefatos de uma filtragem em ``output_dir``:

        dataset_filtrado.csv             instâncias mantidas
        instancias_removidas.csv         removidas (+ `removal_reason`), se houver
        instancias_excluidas.csv         sem rótulo / inválidas, se houver
        instance_selection_report.json   métricas da filtragem

    Reutilizada tanto pelo pipeline dedicado quanto pela filtragem inline do
    fine-tuning, garantindo o mesmo formato de saída.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.filtered_df.to_csv(
        output_dir / "dataset_filtrado.csv", index=False, encoding="utf-8"
    )
    if len(result.removed_df):
        result.removed_df.to_csv(
            output_dir / "instancias_removidas.csv", index=False, encoding="utf-8"
        )
    if len(result.excluded_df):
        result.excluded_df.to_csv(
            output_dir / "instancias_excluidas.csv", index=False, encoding="utf-8"
        )

    with open(output_dir / "instance_selection_report.json", "w", encoding="utf-8") as f:
        json.dump(result.stats, f, indent=4, ensure_ascii=False)

    logger.success(f"Artefatos da filtragem salvos em: {output_dir}")

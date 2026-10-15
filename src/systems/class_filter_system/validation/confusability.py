"""Confusability - Reconstrói as candidatas Top-k da Fase 1 e mede a confusão entre classes quando a classe correta está entre elas (RQ3)"""

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.config.class_filter import DEFAULT_HOLDOUT_FOLD_INDEX, DEFAULT_N_INNER_FOLDS, DEFAULT_TRAIN_FOLD_PATTERN, RANDOM_STATE
from src.systems.class_filter_system.classifiers.factory import get_class_filter
from src.systems.class_filter_system.folds.inner_split import inner_split
from src.utils.data_loader import load_hf_dataset_as_dataframe
from src.utils.get_text_id_from_text import get_text_id_from_text

INVALID = -1


class CandidateRebuilder:
    """
    Refaz a Fase 1 do pipeline em 2 fases (determinística, sem LLM), pois as candidatas não são salvas em disco
    Responsabilidades: carregar cada fold de treino, repetir dedup + split interno + regressão logística e devolver as top-k por texto do held-out
    """

    def __init__(
        self,
        dataset: str,
        cache_dir: str = "data/.cache",
        k: int = 3,
        method: str = "logistic_regression",
        n_inner_folds: int = DEFAULT_N_INNER_FOLDS,
        holdout_fold_index: int = DEFAULT_HOLDOUT_FOLD_INDEX,
        random_state: int = RANDOM_STATE,
    ):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.k = k
        self.method = method
        self.n_inner_folds = n_inner_folds
        self.holdout_fold_index = holdout_fold_index
        self.random_state = random_state
        logger.debug(f"CandidateRebuilder: {dataset} k={k} seed={random_state}")

    def _load_train_fold(self, fold: int) -> Optional[pd.DataFrame]:
        config = DatasetConfig(hf_file=DEFAULT_TRAIN_FOLD_PATTERN.format(fold=fold), random_state=self.random_state)
        try:
            df, _ = load_hf_dataset_as_dataframe(self.dataset, self.cache_dir, config)
        except Exception:
            return None
        df = df.copy()
        df["text"] = df["text"].astype(str)
        df["text_id"] = df["text"].apply(get_text_id_from_text)
        return df.drop_duplicates(subset="text_id").reset_index(drop=True)

    def rebuild(self, fold: int) -> Optional[pd.DataFrame]:
        """held-out do fold com text_id, label e candidates (top-k em ordem canônica)"""
        df_train = self._load_train_fold(fold)
        if df_train is None:
            return None
        df_fit, df_holdout = inner_split(df_train, self.n_inner_folds, self.holdout_fold_index, self.random_state, "label")
        filt = get_class_filter(self.method, random_state=self.random_state)
        filt.fit(df_fit["text"].tolist(), df_fit["label"].tolist())
        candidates = filt.topk(df_holdout["text"].tolist(), k=self.k)
        return pd.DataFrame({
            "fold": fold,
            "text_id": df_holdout["text_id"].values,
            "label": df_holdout["label"].astype(int).values,
            "candidates": candidates,
        })

    def folds(self) -> Iterator[pd.DataFrame]:
        fold = 0
        while (df := self.rebuild(fold)) is not None:
            logger.info(f"[{self.dataset}] fold {fold}: {len(df)} candidatas reconstruídas")
            yield df
            fold += 1


class ConfusabilityAnalyzer:
    """
    Mede a confusão semântica restrita aos textos cuja classe correta está entre as candidatas
    Responsabilidades: P(ŷ≠y | y∈Top-k) global e por classe, matriz de confusão condicional e entropia da distribuição dos erros
    """

    def __init__(self, class_names: Dict[int, str]):
        self.class_names = class_names
        logger.debug(f"ConfusabilityAnalyzer: {len(class_names)} classes")

    @staticmethod
    def conditioned(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """Linhas com y ∈ Top-k e predição válida"""
        in_topk = [y in c for y, c in zip(df["label"], df["candidates"])]
        return df[np.array(in_topk) & (df[pred_col] != INVALID)]

    def conditional_error(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """P(ŷ≠y | y∈Top-k) por classe verdadeira, com a linha 'global'"""
        cond = self.conditioned(df, pred_col).assign(error=lambda d: d[pred_col] != d["label"])
        by_class = cond.groupby("label")["error"].agg(n="size", p_error="mean").reset_index()
        by_class["classe"] = by_class["label"].map(self.class_names)
        overall = pd.DataFrame([{"label": -99, "n": len(cond), "p_error": cond["error"].mean(), "classe": "global"}])
        return pd.concat([by_class, overall], ignore_index=True)

    def conditional_confusion(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """Matriz y × ŷ normalizada por linha, restrita a y∈Top-k"""
        cond = self.conditioned(df, pred_col)
        matrix = pd.crosstab(cond["label"], cond[pred_col], normalize="index")
        names = lambda idx: [self.class_names.get(int(i), str(i)) for i in idx]
        matrix.index, matrix.columns = names(matrix.index), names(matrix.columns)
        return matrix

    def error_entropy(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        """Entropia normalizada (0 = erro concentrado num alvo, 1 = espalhado) dos erros de cada classe, com o alvo principal"""
        errors = self.conditioned(df, pred_col)
        errors = errors[errors[pred_col] != errors["label"]]
        n_targets = max(len(self.class_names) - 1, 2)
        rows = []
        for label, group in errors.groupby("label"):
            p = group[pred_col].value_counts(normalize=True)
            rows.append({
                "classe": self.class_names[int(label)],
                "n_erros": len(group),
                "entropia_norm": self._entropy(p) / np.log2(n_targets),
                "alvo_principal": self.class_names[int(p.index[0])],
                "share_alvo_principal": p.iloc[0],
            })
        pairs = errors.groupby(["label", pred_col]).size()
        rows.append({
            "classe": "global (pares)",
            "n_erros": len(errors),
            "entropia_norm": self._entropy(pairs / pairs.sum()) / np.log2(len(self.class_names) * (len(self.class_names) - 1)),
            "alvo_principal": self._pair_name(pairs.idxmax()) if len(pairs) else None,
            "share_alvo_principal": pairs.max() / pairs.sum() if len(pairs) else np.nan,
        })
        return pd.DataFrame(rows)

    def top_pairs(self, df: pd.DataFrame, pred_col: str, n: int = 5) -> pd.DataFrame:
        """Pares (verdadeira → predita) mais frequentes entre os erros condicionais, com a fração acumulada"""
        errors = self.conditioned(df, pred_col)
        errors = errors[errors[pred_col] != errors["label"]]
        pairs = errors.groupby(["label", pred_col]).size().sort_values(ascending=False)
        share = pairs / pairs.sum()
        return pd.DataFrame({
            "par": [self._pair_name(p) for p in pairs.index[:n]],
            "n": pairs.values[:n],
            "share": share.values[:n],
            "share_acumulado": share.cumsum().values[:n],
        })

    def _pair_name(self, pair: Tuple[int, int]) -> str:
        return f"{self.class_names[int(pair[0])]} → {self.class_names[int(pair[1])]}"

    @staticmethod
    def _entropy(p: pd.Series) -> float:
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

"""Paired System Comparator - Comparação de sistemas pareada por fold (IC 95% por t de Student e t pareado com Bonferroni/BH)"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from statsmodels.stats.multitest import multipletests


class PairedSystemComparator:
    """
    Compara sistemas avaliados nos mesmos folds (método da comparação de sistemas de Jain, como no notebook de linha de base)
    Responsabilidades: IC individual de cada sistema, IC da diferença pareada por fold e t pareado com correção de múltiplas comparações
    """

    def __init__(self, conf: float = 0.95, alpha: float = 0.05):
        self.conf = conf
        self.alpha = alpha
        logger.debug(f"PairedSystemComparator: conf={conf} alpha={alpha}")

    def mean_ci(self, values) -> Tuple[float, float, float]:
        """Média e IC (low, high) via t de Student"""
        x = np.asarray(values, dtype=float)
        m = float(x.mean())
        if len(x) < 2:
            return m, np.nan, np.nan
        h = stats.sem(x) * stats.t.ppf((1 + self.conf) / 2, len(x) - 1)
        return m, m - h, m + h

    def individual(self, folds: pd.DataFrame) -> pd.DataFrame:
        """IC de cada sistema; `folds` em formato longo: dataset, system, metric, fold, value"""
        rows = []
        for (dataset, metric, system), group in folds.groupby(["dataset", "metric", "system"]):
            m, lo, hi = self.mean_ci(group["value"])
            rows.append({"dataset": dataset, "metric": metric, "system": system, "n_folds": len(group),
                         "mean": m, "ci_low": lo, "ci_high": hi})
        return pd.DataFrame(rows)

    def paired(
        self,
        folds: pd.DataFrame,
        metric: str,
        contrasts: List[Tuple[str, str]],
        lower_is_better: bool = False,
    ) -> pd.DataFrame:
        """diff = A − B por fold, para cada dataset e contraste; correção de múltiplas comparações dentro da métrica"""
        data = folds[folds["metric"] == metric]
        rows = []
        for dataset, by_dataset in data.groupby("dataset"):
            wide = by_dataset.pivot_table(index="fold", columns="system", values="value")
            for a, b in contrasts:
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna()
                diff = pair[a] - pair[b]
                m, lo, hi = self.mean_ci(diff)
                p = stats.ttest_rel(pair[a], pair[b]).pvalue if len(pair) >= 2 else np.nan
                rows.append({
                    "dataset": dataset, "metric": metric, "system_a": a, "system_b": b, "n_folds": len(pair),
                    "mean_a": pair[a].mean(), "mean_b": pair[b].mean(),
                    "diff_mean": m, "diff_ci_low": lo, "diff_ci_high": hi,
                    "conclusion": self._conclusion(a, b, lo, hi, lower_is_better),
                    "t_paired_p": p,
                })

        result = pd.DataFrame(rows)
        return self._correct(result) if not result.empty else result

    @staticmethod
    def _conclusion(a: str, b: str, lo: float, hi: float, lower_is_better: bool) -> str:
        if np.isnan(lo) or lo <= 0 <= hi:
            return "equivalentes (IC contém 0)"
        a_better = (lo > 0) != lower_is_better
        return f"{a} melhor" if a_better else f"{b} melhor"

    def _correct(self, result: pd.DataFrame) -> pd.DataFrame:
        """Bonferroni (FWER) e Benjamini-Hochberg (FDR) sobre os p-valores válidos da métrica"""
        valid = result["t_paired_p"].notna()
        for method, column in [("bonferroni", "p_bonferroni"), ("fdr_bh", "p_bh")]:
            result[column] = np.nan
            if valid.any():
                result.loc[valid, column] = multipletests(result.loc[valid, "t_paired_p"], alpha=self.alpha, method=method)[1]
            result[f"signif_{column[2:]}"] = result[column] < self.alpha
        return result

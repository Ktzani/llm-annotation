"""
Stratified Estimator - Estimativa de proporção sob amostragem estratificada + IC de Wald
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


class StratifiedEstimator:
    """
    Estima uma proporção sob amostragem estratificada (Merlo et al., ECIR'26).

    θ̂ = Σ W_h θ̂_h, com W_h = N_h / N;  V(θ̂) = Σ W_h² V_h;
    IC de Wald: θ̂ ± z_{α/2} √V e MoE = z_{α/2} √V.

    Com `pseudo_count` = k > 0, a variância do estrato usa p̃_h = (x_h + k) / (n_h + 2k):
    V_h = p̃_h (1 - p̃_h) / n_h. Evita V_h = 0 quando todos os documentos do estrato
    coincidem (MoE otimista com n pequeno); θ̂ continua o do artigo. k = 0 = Wald original.

    Responsabilidades:
    - Ponderar cada estrato pelo seu tamanho na população (não na amostra)
    - Sinalizar estratos sem amostra (pesos renormalizados entre os representados)
    """

    def __init__(self, confidence_level: float, pseudo_count: float):
        self.confidence_level = confidence_level
        self.pseudo_count = pseudo_count
        self.z = float(norm.ppf(1 - (1 - confidence_level) / 2))
        logger.debug(f"StratifiedEstimator inicializado (confiança={confidence_level}, z={self.z:.3f}, k={pseudo_count})")

    def _stratum_variance(self, v: pd.Series, stratum: int, where: str) -> float:
        """Variância de θ̂_h (com pseudo-contagem, se configurada)."""
        n = len(v)
        if self.pseudo_count > 0:
            p = (v.sum() + self.pseudo_count) / (n + 2 * self.pseudo_count)
            return p * (1 - p) / n
        if n == 1:
            logger.warning(f"{where}: estrato {stratum} com n_h = 1 (variância conservadora 0,25)")
            return 0.25
        return v.var(ddof=1) / n

    def estimate(self, values: pd.DataFrame, strata_sizes: Dict[int, int], where: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Args:
            values: colunas `estrato` (código da classe) e `valor` (0/1)
            strata_sizes: N_h por estrato

        Returns:
            Resumo (θ̂, V, MoE, IC, n) e detalhe por estrato
        """
        rows = []
        for stratum, size in sorted(strata_sizes.items()):
            v = values.loc[values["estrato"] == stratum, "valor"].astype(float)
            n = len(v)
            theta_h = v.mean() if n else np.nan
            var_h = self._stratum_variance(v, stratum, where) if n else np.nan
            rows.append({"estrato": stratum, "N_h": size, "n_h": n, "theta_h": theta_h, "var_h": var_h})

        detail = pd.DataFrame(rows)
        represented = detail["n_h"] > 0
        if (~represented).any():
            logger.warning(f"{where}: estratos sem amostra {detail.loc[~represented, 'estrato'].tolist()} (pesos renormalizados)")

        detail["W_h"] = np.where(represented, detail["N_h"] / detail.loc[represented, "N_h"].sum(), 0.0)
        used = detail[represented]
        theta = float((used["W_h"] * used["theta_h"]).sum()) if len(used) else np.nan
        variance = float((used["W_h"] ** 2 * used["var_h"]).sum()) if len(used) else np.nan
        moe = self.z * np.sqrt(variance)

        summary = {
            "n": int(detail["n_h"].sum()),
            "estratos_sem_amostra": int((~represented).sum()),
            "theta": theta,
            "variancia": variance,
            "moe": moe,
            "ic_inferior": max(0.0, theta - moe),
            "ic_superior": min(1.0, theta + moe),
        }
        return summary, detail

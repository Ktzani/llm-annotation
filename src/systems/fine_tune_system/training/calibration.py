"""Calibration - Métricas de calibração (ECE e Brier) a partir das probabilidades preditas"""

import numpy as np

ECE_BINS = 15
PREDICTIONS_FILE = "predictions.csv"  # probabilidades por instância de cada fold (melhor modelo)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """ECE sobre a confiança máxima, em faixas de mesma largura"""
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.minimum((confidences * n_bins).astype(int), n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidences[in_bin].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier multiclasse: média de Σ_k (p_k - y_k)² por instância"""
    one_hot = np.eye(probs.shape[1])[labels]
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())

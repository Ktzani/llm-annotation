"""
biO-IS: Noise-Oriented and Redundancy-Aware Instance Selection.

Porte fiel do algoritmo proposto por Cunha et al. (waashk/bio-is,
``src/main/python/iSel/biois.py``), com as seguintes adaptações para o
scikit-learn moderno e maior robustez:

  * ``LogisticRegression(solver="warn", multi_class="warn")`` -> ``solver="lbfgs"``.
    Os valores ``"warn"`` eram apenas defaults legados, removidos nas versões
    recentes do scikit-learn.
  * Realocação das probabilidades para a posição correta da classe via
    ``classifier.classes_`` (substitui a inserção posicional de
    ``fix_proba_columns_if_necessary``), equivalente porém correta para
    qualquer conjunto de rótulos codificado em 0..n-1.
  * ``random_state`` para reprodutibilidade (o algoritmo é estocástico).
  * Guardas para casos extremos (sem erros do classificador, classe única,
    poucas amostras por classe).

Ideia geral
-----------
Dois objetivos simultâneos sobre o conjunto de treino:

  1. **Redundância (beta)**: um classificador fraco calibrado (Regressão
     Logística) é avaliado via validação cruzada estratificada (predições
     out-of-fold). Instâncias classificadas CORRETAMENTE com alta confiança são
     "fáceis"/redundantes e recebem maior probabilidade de remoção.

  2. **Ruído (theta)**: entre as instâncias classificadas ERRADAS, baixa
     entropia da distribuição posterior indica que o classificador está
     confiante numa classe errada — forte sinal de ruído. Essas instâncias
     recebem maior probabilidade de remoção (proporcional ao inverso da entropia
     normalizada).

Referências
-----------
[1] Washington Cunha, Alejandro Moreo, Andrea Esuli, Fabrizio Sebastiani,
    Leonardo Rocha e Marcos A. Gonçalves. "A Noise-Oriented and
    Redundancy-Aware Instance Selection Framework." ACM TOIS.
[2] Washington Cunha et al. "An Effective, Efficient, and Scalable
    Confidence-based Instance Selection Framework for Transformer-based Text
    Classification." SIGIR'23.
"""
import copy

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y
from loguru import logger

from src.is_system.core.base import InstanceSelectionMixin


class BIOIS(InstanceSelectionMixin):
    """
    Framework bi-objetivo de seleção de instâncias (biO-IS).

    Parameters
    ----------
    beta : float, default=0.25
        Taxa de remoção de redundância (fração do total de instâncias rotuladas).
    theta : float, default=0.50
        Taxa de remoção de ruído (fração das instâncias classificadas erradas
        pelo classificador fraco).
    n_splits : int, default=5
        Número de folds da validação cruzada estratificada. Reduzido
        automaticamente se a classe minoritária tiver poucas amostras.
    random_state : int, default=42
        Semente para reprodutibilidade.

    Attributes
    ----------
    mask : ndarray of bool
        Máscara das instâncias selecionadas (True = mantida).
    sample_indices_ : ndarray
        Índices posicionais das instâncias selecionadas.
    reduction_ : float
        Taxa de redução R = (|T| - |S|) / |T|.
    classes_ : ndarray
        Rótulos únicos do conjunto.
    """

    def __init__(self, beta=0.25, theta=0.50, n_splits=5, random_state=42):
        self.beta = beta
        self.theta = theta
        self.n_splits = n_splits
        self.random_state = random_state

        self.sample_indices_ = []
        self._idx_redundant = np.array([], dtype=int)
        self._idx_noise = np.array([], dtype=int)
        self._rng = np.random.RandomState(random_state)

        # Métricas do classificador fraco (preenchidas em fitting_alpha).
        self.weak_clf_accuracy = None
        self.weak_clf_f1_macro = None

    # ------------------------------------------------------------------
    # Objetivo 1 — redundância: probabilidade de remoção das instâncias
    # classificadas CORRETAMENTE (out-of-fold) pelo classificador fraco.
    # ------------------------------------------------------------------
    def fitting_alpha(self, X, y):
        nrows = X.shape[0]
        self.classes_ = unique_labels(y)
        ncolumns = len(self.classes_)
        proba_everyone = np.zeros((nrows, ncolumns))

        n_splits = self._safe_n_splits(y)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        for train_index, val_index in skf.split(X, y):
            X_train, y_train = X[train_index], y[train_index]
            X_val = X[val_index]

            classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            classifier.fit(X_train, y_train)

            # Realoca cada coluna para a posição da classe verdadeira; classes
            # ausentes no fold de treino ficam com probabilidade 0.
            probas = classifier.predict_proba(X_val)
            full = np.zeros((probas.shape[0], ncolumns))
            full[:, classifier.classes_] = probas
            proba_everyone[val_index] = full

        pred = np.argmax(proba_everyone, axis=1)
        self.weak_clf_accuracy = float(accuracy_score(y, pred))
        self.weak_clf_f1_macro = float(f1_score(y, pred, average="macro", zero_division=0))
        logger.info(
            f"Classificador fraco (LR) | "
            f"Acurácia: {self.weak_clf_accuracy:.4f} | "
            f"F1-macro: {self.weak_clf_f1_macro:.4f}"
        )

        y_proba_of_pred = proba_everyone[np.arange(nrows), pred]

        self._proba_everyone = copy.copy(proba_everyone)
        self._pred = copy.copy(pred)
        self._y_proba_of_pred = copy.copy(y_proba_of_pred)

        # Probabilidade de remoção ∝ confiança nas instâncias CORRETAS.
        correct_proba = copy.copy(y_proba_of_pred)
        correct_proba[pred != y] = 0.0
        total = correct_proba.sum()
        if total > 0:
            correct_proba = correct_proba / total
        return correct_proba

    # ------------------------------------------------------------------
    # Objetivo 2 — ruído: probabilidade de remoção das instâncias
    # classificadas ERRADAS, ∝ inverso da entropia (baixa entropia = mais
    # confiante no erro = mais provável ser ruído).
    # ------------------------------------------------------------------
    def identify_noise_by_lower_entropy(self, X):
        wrong_idx = self.y_ != self._pred
        nwrong = int(np.sum(wrong_idx))
        if nwrong == 0:
            logger.info("Sem erros do classificador fraco: nada a remover por ruído.")
            return np.array([], dtype=int)

        entropy = np.array(
            [stats.entropy(p) for p in self._proba_everyone[wrong_idx]]
        )
        span = entropy.max() - entropy.min()
        if span > 0:
            entropy = (entropy - entropy.min()) / span
            entropy = 1.0 - entropy  # inverte: baixa entropia -> alta prob.
        else:
            entropy = np.ones_like(entropy)  # entropias iguais -> distribuição uniforme

        s = entropy.sum()
        entropy = entropy / s if s > 0 else np.ones_like(entropy) / len(entropy)

        proba_to_remove = np.zeros(X.shape[0])
        proba_to_remove[wrong_idx] = entropy

        ntoremove = min(int(self.theta * nwrong), int(np.count_nonzero(proba_to_remove)))
        if ntoremove <= 0:
            return np.array([], dtype=int)

        return self._rng.choice(
            a=X.shape[0], size=ntoremove, replace=False, p=proba_to_remove
        )

    def select_end(self, alpha):
        n = len(alpha)
        n_remove = min(int(n * self.beta), int(np.count_nonzero(alpha)))
        if n_remove <= 0:
            return np.array([], dtype=int)
        return self._rng.choice(a=n, size=n_remove, replace=False, p=alpha)

    # ------------------------------------------------------------------
    def select_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse="csr")

        len_original = len(y)
        self.mask = np.ones(y.size, dtype=bool)
        self.classes_ = np.unique(y)
        self.y_ = y

        # Guarda: sem variação de classes ou amostras insuficientes para CV.
        _, counts = np.unique(y, return_counts=True)
        if len(self.classes_) < 2 or counts.min() < 2:
            logger.warning(
                "Classes insuficientes para validação cruzada — nenhuma "
                "instância removida pela seleção."
            )
            self.sample_indices_ = np.arange(len_original)
            self.reduction_ = 0.0
            return X, y

        # Objetivo 1: redundância.
        alpha = self.fitting_alpha(X, y)
        self._idx_redundant = self.select_end(alpha)
        self.mask[self._idx_redundant] = False

        # Objetivo 2: ruído.
        self._idx_noise = self.identify_noise_by_lower_entropy(X)
        self.mask[self._idx_noise] = False

        self.sample_indices_ = np.arange(len_original)[self.mask]
        self.reduction_ = 1.0 - float(self.mask.sum()) / len_original

        logger.info(
            f"biO-IS | redundantes: {len(self._idx_redundant)} | "
            f"ruidosas: {len(self._idx_noise)} | redução: {self.reduction_:.2%}"
        )
        return X[self.mask], y[self.mask]

    # ------------------------------------------------------------------
    def _safe_n_splits(self, y):
        """Reduz n_splits se a classe minoritária tiver menos amostras."""
        _, counts = np.unique(y, return_counts=True)
        min_class = int(counts.min())
        n_splits = max(2, min(self.n_splits, min_class))
        if n_splits != self.n_splits:
            logger.warning(
                f"n_splits reduzido de {self.n_splits} para {n_splits} "
                f"(classe minoritária com {min_class} amostras)."
            )
        return n_splits

"""
Interface base do filtro de classes (Fase 1).

Qualquer classificador plugável implementa este contrato. A implementação padrão
é `LogisticRegressionClassFilter`; novos métodos (SVM, embeddings, etc.) podem
ser registrados na factory sem alterar o restante do pipeline.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class ClassFilterBase(ABC):
    """
    Contrato mínimo de um filtro de classes.

    Convenções:
    - `fit` deve ser chamado APENAS com os dados de treino (fit part). Nunca com
      o pedaço held-out que será previsto — isso vazaria o experimento.
    - `classes_` é a lista de rótulos canônicos (ints) vistos no treino, na mesma
      ordem das colunas de `predict_proba`.
    """

    @abstractmethod
    def fit(self, texts: List[str], labels: List[int]) -> "ClassFilterBase":
        """Treina o vetorizador + classificador nos dados de treino."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Retorna a matriz de probabilidades (n_texts x n_classes), com as colunas
        alinhadas a `self.classes_`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def classes_(self) -> List[int]:
        """Rótulos canônicos (ints) na ordem das colunas de `predict_proba`."""
        raise NotImplementedError

    def topk(self, texts: List[str], k: int) -> List[List[int]]:
        """
        Deriva as top-k classes candidatas por texto a partir de `predict_proba`.

        As candidatas são devolvidas em ORDEM CANÔNICA ASCENDENTE (não em ordem
        de probabilidade) — a apresentação ao LLM não deve revelar o ranking do
        classificador. O corte top-k usa a probabilidade; a ordenação final, não.
        """
        proba = self.predict_proba(texts)
        classes = np.asarray(self.classes_)
        k_eff = min(k, len(classes))

        candidates: List[List[int]] = []
        for row in proba:
            # índices das k maiores probabilidades (desempate estável pela ordem)
            top_idx = np.argsort(row)[::-1][:k_eff]
            labels = sorted(int(classes[i]) for i in top_idx)  # ordem canônica
            candidates.append(labels)
        return candidates

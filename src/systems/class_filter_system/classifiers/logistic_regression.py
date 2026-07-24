"""
Filtro de classes por Regressão Logística sobre TF-IDF.

Primeira implementação de `ClassFilterBase`. Vetorização e classificador são
`fit`ados apenas no conjunto de treino (fit part); o held-out é `transform`ado,
nunca `fit_transform`ado — evita vazamento de vocabulário do held-out.
"""
from typing import List

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.utils.text_vectorizer import TextVectorizer
from src.systems.class_filter_system.classifiers.base import ClassFilterBase


class LogisticRegressionClassFilter(ClassFilterBase):
    """
    TF-IDF + LogisticRegression, reaproveitando o `TextVectorizer` compartilhado
    (mesmos parâmetros de TF-IDF do restante do repo).

    Usa `fit` no fit part e `transform` no held-out (não `fit_transform`), para
    que o vocabulário/IDF nunca seja aprendido com o pedaço previsto — sem
    vazamento.
    """

    def __init__(
        self,
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        **tfidf_overrides,
    ):
        self._vectorizer = TextVectorizer(**tfidf_overrides)
        self._clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, texts: List[str], labels: List[int]) -> "LogisticRegressionClassFilter":
        X = self._vectorizer.fit_transform(texts)
        y = np.asarray(labels)
        self._clf.fit(X, y)
        self._fitted = True
        logger.info(
            f"LogisticRegressionClassFilter treinado: {X.shape[0]} textos x "
            f"{X.shape[1]} termos | {len(self.classes_)} classes"
        )
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Chame fit() antes de predict_proba().")
        X = self._vectorizer.transform(texts)
        return self._clf.predict_proba(X)

    @property
    def classes_(self) -> List[int]:
        if not self._fitted:
            raise RuntimeError("classes_ indisponível antes de fit().")
        return [int(c) for c in self._clf.classes_]

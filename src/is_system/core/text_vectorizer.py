"""
Vetorização TF-IDF dos textos para a seleção de instâncias.

Reproduz o pré-processamento do framework bio-is: remoção de stopwords (lista
padrão do scikit-learn) e descarte de termos pouco frequentes (``min_df``).
"""
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from src.config.instance_selection import TFIDF_CONFIG


class TextVectorizer:
    """Wrapper de ``TfidfVectorizer`` com os parâmetros padrão do framework."""

    def __init__(self, **tfidf_kwargs):
        self._params = {**TFIDF_CONFIG, **tfidf_kwargs}
        self.vectorizer = TfidfVectorizer(**self._params)

    def fit_transform(self, texts) -> csr_matrix:
        logger.info(f"Vetorizando {len(texts)} textos (TF-IDF: {self._params})")
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Matriz TF-IDF: {X.shape[0]} instâncias x {X.shape[1]} termos")
        return X.tocsr()

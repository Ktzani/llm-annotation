"""
Classe base para métodos de Seleção de Instâncias.

Adaptado de waashk/bio-is (src/main/python/iSel/base.py). A única mudança
estrutural é a remoção da dependência de `sklearn.externals.six` — removida nas
versões modernas do scikit-learn — em favor de `abc` da biblioteca padrão.
"""
import warnings
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator


class InstanceReductionWarning(UserWarning):
    """Aviso emitido por técnicas de redução de instâncias."""


warnings.simplefilter("always", InstanceReductionWarning)


class InstanceSelectionBase(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        ...


class InstanceSelectionMixin(InstanceSelectionBase):
    """Mixin comum a todas as técnicas de seleção/redução de instâncias."""

    def select_data(self, X, y):
        """
        Procedimento de redução dos dados.

        Parameters
        ----------
        X : matriz de features (ex.: TF-IDF esparso).
        y : vetor de rótulos.
        """
        raise NotImplementedError

    def fit(self, X, y):
        """Executa a seleção de instâncias e retorna ``self``."""
        self.X = X
        self.y = y
        self.select_data(X, y)
        return self

"""
Seeded Hasher - Ordem pseudoaleatória reproduzível a partir de semente e text_id
"""
import hashlib
from typing import Iterable

import pandas as pd


class SeededHasher:
    """
    Gera chaves de ordenação determinísticas por documento.

    O text_id identifica o documento, mas a ordem dos text_ids é a mesma para
    qualquer semente. Ordenar por sha256(semente|propósito|text_id) equivale a
    embaralhar com aquela semente, sem depender da ordem das linhas do CSV nem
    da versão do numpy/pandas; o `purpose` separa sorteios independentes
    (amostragem, exemplos do guia, ordem da planilha, id anônimo).
    """

    def __init__(self, seed: int, dataset_name: str):
        self.seed = seed
        self.dataset_name = dataset_name

    def key(self, purpose: str, text_id: str) -> str:
        """Chave hexadecimal de um documento para um propósito."""
        raw = f"{self.seed}|{self.dataset_name}|{purpose}|{text_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def keys(self, purpose: str, text_ids: Iterable[str]) -> pd.Series:
        """Chaves de vários documentos, alinhadas ao índice de entrada se for Series."""
        index = text_ids.index if isinstance(text_ids, pd.Series) else None
        return pd.Series([self.key(purpose, str(t)) for t in text_ids], index=index)

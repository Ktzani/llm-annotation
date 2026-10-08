"""
Consensus File Manager - Recebe e valida o dataset_consenso.csv enviado pela interface
"""
import hashlib
import io
from pathlib import Path
from typing import Dict

import pandas as pd
from loguru import logger

from src.systems.human_validation_system.sampling.agreement_grouper import AgreementGrouper


class InvalidConsensusError(ValueError):
    """Arquivo enviado não é um dataset de consenso utilizável."""


class ConsensusFileManager:
    """
    Guarda o CSV de consenso de cada experimento no servidor.

    Responsabilidades:
    - Validar o arquivo (colunas obrigatórias e as 3 colunas de rótulo das LLMs)
    - Gravar de forma atômica em `<results>/<dataset>/<date>/consensus/dataset_consenso.csv`
    """

    FILE_NAME = "dataset_consenso.csv"
    REQUIRED_COLUMNS = ("text_id", "text", "ground_truth", "resolved_annotation")

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.grouper = AgreementGrouper()
        logger.debug(f"ConsensusFileManager inicializado ({self.results_dir})")

    def path(self, dataset: str, date: str) -> Path:
        return self.results_dir / dataset / date / "consensus" / self.FILE_NAME

    def exists(self, dataset: str, date: str) -> bool:
        return self.path(dataset, date).exists()

    @staticmethod
    def sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def validate(self, content: bytes) -> pd.DataFrame:
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise InvalidConsensusError(f"Não foi possível ler o CSV: {e}")
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise InvalidConsensusError(f"Colunas obrigatórias ausentes: {missing}")
        try:
            self.grouper.detect_llm_columns(df)
        except ValueError as e:
            raise InvalidConsensusError(str(e))
        if df.empty:
            raise InvalidConsensusError("O CSV não tem linhas")
        return df

    def save(self, dataset: str, date: str, content: bytes) -> Dict:
        """Valida e grava (arquivo temporário + replace); retorna linhas e sha256."""
        df = self.validate(content)
        target = self.path(dataset, date)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".csv.tmp")
        tmp.write_bytes(content)
        tmp.replace(target)
        logger.success(f"{dataset}: CSV de consenso salvo ({len(df)} linhas) em {target}")
        return {"linhas": int(len(df)), "sha256": self.sha256(content)}

"""
Stopping Status - Convenção de arquivos da estimativa e leitura do status de parada
"""
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger


class StoppingStatusReader:
    """
    Lê quais grupos já atingiram o critério de parada.

    Responsabilidades:
    - Centralizar onde a estimativa de cada rodada é salva
    - Informar ao sorteio os grupos com status `parar` na estimativa da última rodada
    - Recusar a decisão se a estimativa ainda tem documentos pendentes (rodada incompleta)
    """

    OUTPUT_SUBDIR = "estimativas"
    SUMMARY_PATTERN = "estimativa_ate_rodada_{:02d}.csv"
    STOP = "parar"

    def __init__(self):
        logger.debug("StoppingStatusReader inicializado")

    @classmethod
    def summary_path(cls, validation_dir: Path, n_rounds: int) -> Path:
        return Path(validation_dir) / cls.OUTPUT_SUBDIR / cls.SUMMARY_PATTERN.format(n_rounds)

    def stopped_groups(self, validation_dir: Path, n_rounds: int) -> List[str]:
        """Grupos com status `parar` na métrica principal após `n_rounds` rodadas."""
        path = self.summary_path(validation_dir, n_rounds)
        if not path.exists():
            raise FileNotFoundError(
                f"Estimativa da rodada {n_rounds} não encontrada ({path}). Rode "
                f"run_human_validation_estimate.py com as planilhas preenchidas antes de pedir a próxima rodada."
            )
        summary = pd.read_csv(path)
        primary = summary[summary["principal"]]
        pending = int((primary["n_entregues"] - primary["n_completos"]).sum())
        if pending:
            raise ValueError(
                f"A estimativa da rodada {n_rounds} tem {pending} documento(s) sem resposta de todos os "
                f"avaliadores. Aguarde as planilhas completas e recalcule antes de pedir a próxima rodada."
            )
        return sorted(primary.loc[primary["status"] == self.STOP, "grupo"].tolist())

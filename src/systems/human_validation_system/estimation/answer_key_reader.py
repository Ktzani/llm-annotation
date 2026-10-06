"""
Answer Key Reader - Lê os gabaritos de todas as rodadas entregues
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.systems.human_validation_system.output.answer_key_writer import AnswerKeyWriter


class AnswerKeyReader:
    """Concatena `rodada_XX/gabarito.csv` das rodadas 1..n com a coluna `rodada`."""

    def __init__(self):
        logger.debug("AnswerKeyReader inicializado")

    def read(self, validation_dir: Path, n_rounds: int) -> pd.DataFrame:
        keys = [
            pd.read_csv(Path(validation_dir) / f"rodada_{r:02d}" / AnswerKeyWriter.FILE_NAME).assign(rodada=r)
            for r in range(1, n_rounds + 1)
        ]
        return pd.concat(keys, ignore_index=True)

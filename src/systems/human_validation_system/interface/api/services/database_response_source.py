"""
Database Response Source - Entrega à estimação as respostas gravadas pela interface
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from loguru import logger

from src.systems.human_validation_system.estimation.answer_key_reader import AnswerKeyReader
from src.systems.human_validation_system.interface.api.services.response_store import ANSWER_FIELDS, ResponseStore


class DatabaseResponseSource:
    """Mesmo contrato do EvaluationResponseLoader, lendo as respostas do banco da interface."""

    def __init__(self, store: ResponseStore, experiment_key: str):
        self.store = store
        self.experiment_key = experiment_key
        self.answer_key_reader = AnswerKeyReader()
        logger.debug(f"DatabaseResponseSource inicializado ({experiment_key})")

    def load(self, validation_dir: Path, n_rounds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        responses = self.store.all_responses(self.experiment_key, n_rounds)
        responses = responses[["id_anonimo", *ANSWER_FIELDS, "avaliador", "rodada"]]
        return responses, self.answer_key_reader.read(validation_dir, n_rounds)

"""
Response Loader - Lê as planilhas preenchidas e os gabaritos de todas as rodadas
"""
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from loguru import logger

from src.systems.human_validation_system.estimation.answer_key_reader import AnswerKeyReader
from src.systems.human_validation_system.output.evaluation_sheet_writer import EvaluationSheetWriter


class EvaluationResponseLoader:
    """
    Carrega as respostas dos avaliadores.

    Responsabilidades:
    - Ler `planilha_avaliacao_<avaliador>.xlsx` e `gabarito.csv` de cada rodada
    - Normalizar rótulos (caixa/espaços/acentos) para o nome canônico da classe
    - Tratar valores fora da lista como ausentes, registrando-os no log
    """

    ANSWER_COLUMNS = ["rotulo_escolhido", "outro_rotulo_possivel", "qual_outro_rotulo", "observacao"]

    def __init__(self, evaluators: List[str], class_names: List[str], insufficient_option: str):
        self.evaluators = evaluators
        self._labels = {self._key(n): n for n in class_names}
        self._chosen = {**self._labels, self._key(insufficient_option): insufficient_option}
        self._yes_no = {self._key(v): v for v in EvaluationSheetWriter.YES_NO}
        self.answer_key_reader = AnswerKeyReader()
        logger.debug(f"EvaluationResponseLoader inicializado ({len(evaluators)} avaliadores)")

    @staticmethod
    def _key(value: str) -> str:
        text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
        return " ".join(text.lower().split())

    def _normalize(self, values: pd.Series, vocabulary: dict, where: str) -> pd.Series:
        """Mapeia para o vocabulário; valores desconhecidos viram NaN."""
        filled = values.dropna().astype(str).str.strip()
        filled = filled[filled != ""]
        mapped = filled.map(lambda v: vocabulary.get(self._key(v)))
        invalid = sorted(set(filled[mapped.isna()]))
        if invalid:
            logger.warning(f"{where}: valores fora da lista ignorados: {invalid}")
        return mapped.reindex(values.index)

    def _load_sheet(self, path: Path, evaluator: str, round_number: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            logger.warning(f"Rodada {round_number}: planilha de '{evaluator}' ausente ({path.name})")
            return None

        sheet = pd.read_excel(path, dtype=str)
        missing = set(EvaluationSheetWriter.COLUMNS) - set(sheet.columns)
        if missing:
            raise ValueError(f"{path}: colunas ausentes {sorted(missing)}")

        where = f"Rodada {round_number} / {evaluator}"
        out = pd.DataFrame({"id_anonimo": sheet["id_anonimo"].str.strip()})
        out["rotulo_escolhido"] = self._normalize(sheet["rotulo_escolhido"], self._chosen, where)
        out["outro_rotulo_possivel"] = self._normalize(sheet["outro_rotulo_possivel"], self._yes_no, where)
        out["qual_outro_rotulo"] = self._normalize(sheet["qual_outro_rotulo"], self._labels, where)
        out["observacao"] = sheet["observacao"]
        out["avaliador"] = evaluator
        out["rodada"] = round_number
        return out

    def load(self, output_dir: Path, n_rounds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Respostas em formato longo (documento x avaliador) e gabarito de todas as rodadas."""
        answer_key = self.answer_key_reader.read(output_dir, n_rounds)
        responses = []
        for round_number in range(1, n_rounds + 1):
            round_dir = Path(output_dir) / f"rodada_{round_number:02d}"
            key = answer_key[answer_key["rodada"] == round_number]

            for evaluator in self.evaluators:
                sheet = self._load_sheet(round_dir / f"planilha_avaliacao_{evaluator}.xlsx", evaluator, round_number)
                if sheet is not None:
                    unknown = set(sheet["id_anonimo"]) - set(key["id_anonimo"])
                    if unknown:
                        raise ValueError(f"Rodada {round_number} / {evaluator}: id_anonimo fora do gabarito: {sorted(unknown)[:5]}")
                    responses.append(sheet)

        columns = ["id_anonimo"] + self.ANSWER_COLUMNS + ["avaliador", "rodada"]
        responses_df = pd.concat(responses, ignore_index=True) if responses else pd.DataFrame(columns=columns)
        return responses_df, answer_key

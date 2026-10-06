"""
Answer Validator - Valida e normaliza a resposta enviada pelo avaliador
"""
from typing import Dict, List, Optional

from loguru import logger

from src.systems.human_validation_system.output.evaluation_sheet_writer import EvaluationSheetWriter


class InvalidAnswerError(ValueError):
    """Resposta fora das opções permitidas."""


class AnswerValidator:
    """
    Garante que a resposta usa só os valores permitidos.

    Regras:
    - rotulo_escolhido: uma classe ou a opção de indecisão
    - outro_rotulo_possivel: sim/não
    - qual_outro_rotulo: obrigatório com "sim" (classe diferente da escolhida); vazio com "não"
    - observacao: opcional, até MAX_OBSERVATION caracteres
    """

    MAX_OBSERVATION = 2000

    def __init__(self, class_names: List[str], undecidable_option: str):
        self.class_names = class_names
        self.undecidable_option = undecidable_option
        self.chosen_options = class_names + [undecidable_option]
        logger.debug(f"AnswerValidator inicializado ({len(class_names)} classes)")

    def validate(
        self,
        rotulo_escolhido: str,
        outro_rotulo_possivel: str,
        qual_outro_rotulo: Optional[str],
        observacao: Optional[str],
    ) -> Dict:
        if rotulo_escolhido not in self.chosen_options:
            raise InvalidAnswerError(f"Rótulo inválido: {rotulo_escolhido!r}")
        if outro_rotulo_possivel not in EvaluationSheetWriter.YES_NO:
            raise InvalidAnswerError("Indique 'sim' ou 'não' para outro rótulo possível")

        other = (qual_outro_rotulo or "").strip() or None
        if outro_rotulo_possivel == "sim":
            if other not in self.class_names:
                raise InvalidAnswerError("Escolha qual outro rótulo poderia ser justificado")
            if other == rotulo_escolhido:
                raise InvalidAnswerError("O outro rótulo deve ser diferente do escolhido")
        else:
            other = None

        note = (observacao or "").strip() or None
        if note and len(note) > self.MAX_OBSERVATION:
            raise InvalidAnswerError(f"Observação com mais de {self.MAX_OBSERVATION} caracteres")

        return {
            "rotulo_escolhido": rotulo_escolhido,
            "outro_rotulo_possivel": outro_rotulo_possivel,
            "qual_outro_rotulo": other,
            "observacao": note,
        }

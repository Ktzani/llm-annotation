"""
Schemas do avaliador - o que as telas de avaliação podem receber
"""
from typing import List, Optional

from pydantic import BaseModel

# Nenhum destes modelos tem campo de rótulo de referência, LLM, grupo ou classe:
# o response_model descarta qualquer campo extra antes de chegar ao navegador.

class ClassGuide(BaseModel):
    rotulo: str
    descricao: str
    exemplos: List[str]


class OptionsOut(BaseModel):
    classes: List[ClassGuide]
    opcao_indecidivel: str
    sim_nao: List[str]


class DatasetProgressOut(BaseModel):
    dataset: str
    rodada: Optional[int]
    rodada_aberta: bool
    respondidos: int
    total: int


class AnswerIn(BaseModel):
    rotulo_escolhido: str
    outro_rotulo_possivel: str
    qual_outro_rotulo: Optional[str] = None


class AnswerOut(BaseModel):
    id_anonimo: str
    posicao: int
    rotulo_escolhido: str
    outro_rotulo_possivel: str
    qual_outro_rotulo: Optional[str]
    atualizado_em: str


class DocumentOut(BaseModel):
    id_anonimo: str
    texto: str
    posicao: int
    total: int
    resposta: Optional[AnswerOut] = None


class NextDocumentOut(BaseModel):
    concluido: bool
    respondidos: int
    total: int
    documento: Optional[DocumentOut] = None
    # Quando todos terminaram: até quando ainda dá para revisar (fechamento automático)
    revisao_ate: Optional[str] = None

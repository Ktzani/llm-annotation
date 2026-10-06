"""
Rotas do avaliador: documentos da rodada aberta e as próprias respostas.

Só leem a lista cega (id_anonimo + texto) e as respostas do próprio avaliador;
nunca acessam gabarito, grupo, classe ou respostas dos demais.
"""
from typing import List, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request

from src.systems.human_validation_system.interface.api.core.auth import require_evaluator
from src.systems.human_validation_system.interface.api.schemas.evaluator import (
    AnswerIn,
    AnswerOut,
    DatasetProgressOut,
    DocumentOut,
    NextDocumentOut,
    OptionsOut,
)
from src.systems.human_validation_system.interface.api.services.answer_validator import (
    AnswerValidator,
    InvalidAnswerError,
)
from src.systems.human_validation_system.output.evaluation_sheet_writer import EvaluationSheetWriter

router = APIRouter(prefix="/api/avaliacao", tags=["Avaliação"])


def _controller(request: Request):
    return request.app.state.controller


def _open_round(request: Request, dataset: str) -> Tuple[str, int]:
    """Chave do experimento no banco e número da rodada aberta."""
    if dataset not in request.app.state.settings.experiments:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    key = _controller(request).store_key(dataset)
    round_number = request.app.state.store.open_round(key)
    if round_number is None:
        raise HTTPException(status_code=409, detail="Não há rodada aberta para este dataset")
    return key, round_number


@router.get("/datasets", response_model=List[DatasetProgressOut])
def list_datasets(request: Request, user: str = Depends(require_evaluator)):
    store = request.app.state.store
    out = []
    for dataset in request.app.state.settings.experiments:
        key = _controller(request).store_key(dataset)
        current = store.current_round(key)
        is_open = bool(current and current["estado"] == "aberta")
        answered = total = 0
        if is_open:
            total = store.round_size(key, current["rodada"])
            answered = store.progress(key, current["rodada"], [user])[user]
        out.append(DatasetProgressOut(
            dataset=dataset,
            rodada=current["rodada"] if current else None,
            rodada_aberta=is_open,
            respondidos=answered,
            total=total,
        ))
    return out


@router.get("/{dataset}/opcoes", response_model=OptionsOut)
def options(dataset: str, request: Request, user: str = Depends(require_evaluator)):
    try:
        guide = _controller(request).guide(dataset)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")
    return OptionsOut(**guide, sim_nao=EvaluationSheetWriter.YES_NO)


@router.get("/{dataset}/proximo", response_model=NextDocumentOut)
def next_document(dataset: str, request: Request, user: str = Depends(require_evaluator)):
    key, round_number = _open_round(request, dataset)
    store = request.app.state.store
    total = store.round_size(key, round_number)
    answered = store.progress(key, round_number, [user])[user]
    doc = store.next_unanswered(key, round_number, user)
    if doc is None:
        deadline = _controller(request).review_deadline(dataset)
        return NextDocumentOut(
            concluido=True, respondidos=answered, total=total,
            revisao_ate=deadline.isoformat(timespec="seconds") if deadline else None,
        )
    return NextDocumentOut(
        concluido=False, respondidos=answered, total=total, documento=DocumentOut(**doc, total=total)
    )


@router.get("/{dataset}/respostas", response_model=List[AnswerOut])
def my_answers(dataset: str, request: Request, user: str = Depends(require_evaluator)):
    key, round_number = _open_round(request, dataset)
    return request.app.state.store.evaluator_responses(key, round_number, user)


@router.get("/{dataset}/documentos/{id_anonimo}", response_model=DocumentOut)
def get_document(dataset: str, id_anonimo: str, request: Request, user: str = Depends(require_evaluator)):
    key, round_number = _open_round(request, dataset)
    store = request.app.state.store
    doc = store.get_document(key, round_number, id_anonimo)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento não pertence à rodada aberta")
    answer = store.get_response(key, round_number, user, id_anonimo)
    return DocumentOut(
        **doc,
        total=store.round_size(key, round_number),
        resposta=AnswerOut(id_anonimo=id_anonimo, posicao=doc["posicao"], **answer) if answer else None,
    )


@router.put("/{dataset}/respostas/{id_anonimo}", response_model=AnswerOut)
def save_answer(dataset: str, id_anonimo: str, body: AnswerIn, request: Request, user: str = Depends(require_evaluator)):
    key, round_number = _open_round(request, dataset)
    store = request.app.state.store
    doc = store.get_document(key, round_number, id_anonimo)
    if doc is None:
        raise HTTPException(status_code=404, detail="Documento não pertence à rodada aberta")

    controller = _controller(request)
    validator = AnswerValidator(controller.class_names(dataset), controller.guide(dataset)["opcao_indecidivel"])
    try:
        answer = validator.validate(**body.model_dump())
    except InvalidAnswerError as e:
        raise HTTPException(status_code=422, detail=str(e))

    store.save_response(key, round_number, user, id_anonimo, answer)
    saved = store.get_response(key, round_number, user, id_anonimo)
    return AnswerOut(id_anonimo=id_anonimo, posicao=doc["posicao"], **saved)

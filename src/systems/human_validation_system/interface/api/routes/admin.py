"""
Rotas do administrador: acompanhamento, abertura/fechamento de rodadas e planilha consolidada.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from src.systems.human_validation_system.interface.api.core.auth import require_admin
from src.systems.human_validation_system.interface.api.services.round_controller import RoundStateError

router = APIRouter(prefix="/api/admin", tags=["Administração"], dependencies=[Depends(require_admin)])


def _controller(request: Request):
    return request.app.state.controller


def _check(request: Request, dataset: str) -> None:
    if dataset not in request.app.state.settings.experiments:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")


@router.get("/status")
def status(request: Request):
    """Estado e progresso por dataset e avaliador."""
    controller = _controller(request)
    return [controller.status(d) for d in request.app.state.settings.experiments]


@router.post("/{dataset}/rodadas")
def start_round(dataset: str, request: Request):
    """Abre a próxima rodada (sorteio só para os grupos que ainda não pararam)."""
    _check(request, dataset)
    try:
        return _controller(request).start_round(dataset)
    except RoundStateError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/{dataset}/fechar")
def close_round(dataset: str, request: Request):
    """Fecha a rodada se os três terminaram; consolida, estima e atualiza a planilha."""
    _check(request, dataset)
    try:
        return _controller(request).close_round(dataset)
    except RoundStateError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/{dataset}/planilha")
def download_workbook(dataset: str, request: Request):
    _check(request, dataset)
    path = _controller(request).workbook_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Planilha consolidada ainda não gerada")
    return FileResponse(path, filename=path.name)

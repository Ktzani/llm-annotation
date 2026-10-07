"""
Rotas do administrador: acompanhamento, abertura/fechamento de rodadas e planilha consolidada.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

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


class AutomationRequest(BaseModel):
    proxima_automatica: Optional[bool] = None
    fechamento_automatico: Optional[bool] = None


@router.get("/automacao")
def get_automation(request: Request):
    return _controller(request).automation()


@router.put("/automacao")
def set_automation(body: AutomationRequest, request: Request):
    """Liga/desliga a abertura automática da próxima rodada e o fechamento automático."""
    return _controller(request).set_automation(body.proxima_automatica, body.fechamento_automatico)


RESET_WORD = "REINICIAR"


class ResetRequest(BaseModel):
    confirmacao: str


@router.post("/reiniciar")
def reset_all(body: ResetRequest, request: Request):
    """Apaga rodadas, respostas, estimativas e planilhas de todos os experimentos (com backup)."""
    if body.confirmacao != RESET_WORD:
        raise HTTPException(status_code=400, detail=f"Digite {RESET_WORD} para confirmar")
    return {"backups": _controller(request).reset_all()}


@router.post("/{dataset}/reiniciar")
def reset_dataset(dataset: str, body: ResetRequest, request: Request):
    """Apaga rodadas, respostas, estimativas e planilha só deste dataset (com backup)."""
    _check(request, dataset)
    if body.confirmacao != dataset:
        raise HTTPException(status_code=400, detail=f"Digite {dataset} para confirmar")
    return {"backups": [_controller(request).reset_dataset(dataset)]}


@router.get("/{dataset}/painel")
def dashboard(dataset: str, request: Request):
    """Rodadas fechadas do dataset com duração e métricas acumuladas até cada uma."""
    _check(request, dataset)
    return _controller(request).dashboard(dataset)


@router.get("/{dataset}/planilha")
def download_workbook(dataset: str, request: Request):
    _check(request, dataset)
    path = _controller(request).workbook_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Planilha consolidada ainda não gerada")
    return FileResponse(path, filename=path.name)

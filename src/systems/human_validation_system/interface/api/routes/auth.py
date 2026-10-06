"""
Rota de login (nome + código de acesso) da interface.
"""
from typing import List

from fastapi import APIRouter, Request

from src.systems.human_validation_system.interface.api.core.auth import check_credentials
from src.systems.human_validation_system.interface.api.schemas.auth import LoginRequest, LoginResponse

router = APIRouter(prefix="/api", tags=["Login"])


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request):
    role = check_credentials(request.app.state.settings, body.usuario, body.codigo)
    token = request.app.state.store.create_session(body.usuario, role)
    return LoginResponse(token=token, usuario=body.usuario, papel=role)


@router.get("/usuarios", response_model=List[str])
def list_evaluators(request: Request):
    """Nomes dos avaliadores para a tela de login."""
    return request.app.state.settings.evaluators

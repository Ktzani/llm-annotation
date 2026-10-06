"""
Auth - Login por nome + código de acesso e dependências de papel (avaliador/admin)
"""
import secrets
from typing import Tuple

from fastapi import Header, HTTPException, Request

from src.config.human_validation import ADMIN_USER
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings

EVALUATOR_ROLE = "avaliador"
ADMIN_ROLE = "admin"


def check_credentials(settings: InterfaceSettings, user: str, code: str) -> str:
    """Papel do usuário se o código confere; erro 401 caso contrário."""
    if user == ADMIN_USER:
        expected, role = settings.admin_code, ADMIN_ROLE
    elif user in settings.evaluator_codes:
        expected, role = settings.evaluator_codes[user], EVALUATOR_ROLE
    else:
        raise HTTPException(status_code=401, detail="Usuário ou código inválido")
    if not secrets.compare_digest(code.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Usuário ou código inválido")
    return role


def current_session(request: Request, authorization: str = Header(default="")) -> Tuple[str, str]:
    token = authorization.removeprefix("Bearer ").strip()
    session = request.app.state.store.get_session(token) if token else None
    if session is None:
        raise HTTPException(status_code=401, detail="Sessão inválida; faça login novamente")
    return session


def require_evaluator(request: Request, authorization: str = Header(default="")) -> str:
    user, role = current_session(request, authorization)
    if role != EVALUATOR_ROLE:
        raise HTTPException(status_code=403, detail="Acesso restrito a avaliadores")
    return user


def require_admin(request: Request, authorization: str = Header(default="")) -> str:
    user, role = current_session(request, authorization)
    if role != ADMIN_ROLE:
        raise HTTPException(status_code=403, detail="Acesso restrito ao administrador")
    return user

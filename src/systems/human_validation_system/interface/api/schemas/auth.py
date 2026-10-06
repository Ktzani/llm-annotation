"""
Schemas de autenticação da interface
"""
from typing import Literal

from pydantic import BaseModel


class LoginRequest(BaseModel):
    usuario: str
    codigo: str


class LoginResponse(BaseModel):
    token: str
    usuario: str
    papel: Literal["avaliador", "admin"]

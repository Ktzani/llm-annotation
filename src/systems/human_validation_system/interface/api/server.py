"""
Servidor da interface web da validação humana (API FastAPI + telas de `interface/web`).

Telas:  /        login
        /avaliar avaliação (avaliador)
        /admin   acompanhamento e controle das rodadas (administrador)

Deploy (Docker / servidor):
    uvicorn src.systems.human_validation_system.interface.api.server:create_app_from_env --factory
com as variáveis HV_* do .env (ver docs/VALIDACAO_HUMANA_INTERFACE.md).
"""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.routes.admin import router as admin_router
from src.systems.human_validation_system.interface.api.routes.auth import router as auth_router
from src.systems.human_validation_system.interface.api.routes.evaluator import router as evaluator_router
from src.systems.human_validation_system.interface.api.services.response_store import ResponseStore
from src.systems.human_validation_system.interface.api.services.round_controller import RoundController

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PAGES = {"/": "index.html", "/avaliar": "avaliar.html", "/admin": "admin.html"}


def create_app(settings: InterfaceSettings) -> FastAPI:
    app = FastAPI(title="Validação Humana", docs_url=None, redoc_url=None, openapi_url=None)

    store = ResponseStore(settings.db_path)
    app.state.settings = settings
    app.state.store = store
    app.state.controller = RoundController(settings, store)

    app.include_router(auth_router)
    app.include_router(evaluator_router)
    app.include_router(admin_router)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    for route, page in PAGES.items():
        app.add_api_route(route, lambda page=page: FileResponse(WEB_DIR / page), include_in_schema=False)

    @app.get("/api/saude", include_in_schema=False)
    def health():
        return {"status": "ok"}

    return app


def create_app_from_env() -> FastAPI:
    """Fábrica para `uvicorn --factory`: lê HV_* do ambiente/.env."""
    load_dotenv()
    settings = InterfaceSettings.from_env()
    if not settings.experiments:
        raise RuntimeError("HV_EXPERIMENTS não configurado (ex.: books:2026-04-09_13-21-37,dblp:...)")
    return create_app(settings)

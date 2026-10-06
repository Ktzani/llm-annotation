"""
Servidor da interface web da validação humana (API FastAPI + telas de `interface/web`).

Telas:  /        login
        /avaliar avaliação (avaliador)
        /admin   acompanhamento e controle das rodadas (administrador)

Deploy (Docker / servidor):
    uvicorn src.systems.human_validation_system.interface.api.server:create_app_from_env --factory
com as variáveis HV_* do .env (ver docs/VALIDACAO_HUMANA_INTERFACE.md).
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config.human_validation import AUTO_CLOSE_CHECK_SECONDS
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.routes.admin import router as admin_router
from src.systems.human_validation_system.interface.api.routes.auth import router as auth_router
from src.systems.human_validation_system.interface.api.routes.evaluator import router as evaluator_router
from src.systems.human_validation_system.interface.api.services.auto_close_scheduler import AutoCloseScheduler
from src.systems.human_validation_system.interface.api.services.response_store import ResponseStore
from src.systems.human_validation_system.interface.api.services.round_controller import RoundController

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
PAGES = {"/": "index.html", "/avaliar": "avaliar.html", "/admin": "admin.html"}


def web_version() -> str:
    """Versão das telas (última modificação em web/): abas abertas percebem atualizações."""
    return str(int(max(p.stat().st_mtime for p in WEB_DIR.rglob("*") if p.is_file())))


def create_app(settings: InterfaceSettings) -> FastAPI:
    store = ResponseStore(settings.db_path)
    controller = RoundController(settings, store)
    scheduler = AutoCloseScheduler(controller, AUTO_CLOSE_CHECK_SECONDS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fechamento automático ao fim da janela de revisão (respeita a chave ligada/desligada na tela)
        task = asyncio.create_task(scheduler.run())
        yield
        task.cancel()

    app = FastAPI(title="Validação Humana", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.controller = controller

    @app.middleware("http")
    async def no_cache_screens(request, call_next):
        # Telas/JS/CSS sempre revalidados: mudanças aparecem com um F5
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.include_router(auth_router)
    app.include_router(evaluator_router)
    app.include_router(admin_router)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    def render(page: str) -> HTMLResponse:
        # CSS/JS com ?v=<versão>: o navegador nunca mistura arquivos antigos (cache) com telas novas
        html = (WEB_DIR / page).read_text(encoding="utf-8").replace("__VERSAO__", web_version())
        return HTMLResponse(html)

    for route, page in PAGES.items():
        app.add_api_route(route, lambda page=page: render(page), include_in_schema=False)

    @app.get("/api/versao", include_in_schema=False)
    def version():
        return {"versao": web_version()}

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

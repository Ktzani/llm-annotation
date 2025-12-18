from fastapi import FastAPI

from src.api.core.config import setup_logger, setup_cors
from src.api.routes.experiments import router as experiments_router
from src.api.routes.datasets import router as datasets_router
from src.api.routes.health import router as health_router

setup_logger()

app = FastAPI(
    title="LLM Annotation API",
    description="API para experimentação de anotação com múltiplos LLMs",
    version="1.0.0",
)

setup_cors(app)

app.include_router(experiments_router)
app.include_router(datasets_router)
app.include_router(health_router)

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "LLM Annotation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /experiments": "Criar novo experimento",
            "GET /experiments/{experiment_id}": "Obter status do experimento",
            "GET /experiments": "Listar todos os experimentos",
            "GET /datasets": "Listar datasets disponíveis",
            "GET /health": "Health check"
        }
    }

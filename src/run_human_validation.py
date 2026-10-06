"""
Entry point: VALIDAÇÃO HUMANA (amostragem incremental por rodadas e estimação).

Modos (`mode` no main):
    "interface"   Sobe a interface web (avaliadores + acompanhamento do administrador);
                  respostas no SQLite, rodadas abertas/fechadas pela tela /admin.
                  Códigos de acesso no .env (HV_ADMIN_CODE, HV_CODE_AVALIADOR_1..3).
    "rodada"      Gera em `<results>/validacao_humana/<dataset>/<date>/` o guia do
                  avaliador e, para a rodada pedida, as planilhas cegas (uma por
                  avaliador) e o gabarito interno.
    "estimativa"  Lê as planilhas preenchidas de todas as rodadas e recalcula θ̂, IC
                  e MoE por grupo, indicando quais grupos ainda precisam de rodada.

Fluxo de cada rodada:
    1. mode = "rodada" com `round_number = None` gera a próxima rodada
       (a 1ª tem INITIAL_ROUND_SIZE por grupo; as seguintes, `increment_size`);
       `round_number = k` regenera a rodada k (mesmos documentos e ordem;
       planilhas existentes não são sobrescritas)
    2. As planilhas preenchidas voltam para `validacao_humana/rodada_XX/` com o mesmo nome
    3. mode = "estimativa" recalcula as estimativas; a próxima rodada só é sorteada
       depois disso e apenas para os grupos que ainda não pararam

As configurações são definidas estaticamente abaixo.
"""
import sys

import uvicorn
from dotenv import load_dotenv
from loguru import logger

# Garante UTF-8 no console do Windows (evita UnicodeEncodeError com emojis/acentos).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)

from src.config.human_validation import INTERFACE_HOST, INTERFACE_PORT
from src.systems.human_validation_system.estimation.pipeline import (
    HumanValidationEstimationConfig,
    HumanValidationEstimationPipeline,
)
from src.systems.human_validation_system.interface.api.core.settings import InterfaceSettings
from src.systems.human_validation_system.interface.api.server import create_app
from src.systems.human_validation_system.pipeline import (
    DEFAULT_RESULTS_DIR,
    HumanValidationConfig,
    HumanValidationPipeline,
)


def run_interface(experiments: dict, increment_size: int) -> None:
    load_dotenv()
    settings = InterfaceSettings.from_env(experiments, DEFAULT_RESULTS_DIR, increment_size)
    logger.info(f"Interface em http://localhost:{INTERFACE_PORT} (banco: {settings.db_path})")
    uvicorn.run(create_app(settings), host=INTERFACE_HOST, port=INTERFACE_PORT)


def run_round(experiments: dict, round_number, increment_size: int) -> None:
    for dataset_name, specific_date in experiments.items():
        config = HumanValidationConfig(
            dataset_name=dataset_name,
            specific_date=specific_date,
            increment_size=increment_size,
        )
        HumanValidationPipeline(config).run(round_number=round_number)


def run_estimate(experiments: dict) -> None:
    for dataset_name, specific_date in experiments.items():
        config = HumanValidationEstimationConfig(dataset_name=dataset_name, specific_date=specific_date)
        HumanValidationEstimationPipeline(config).run()


def main() -> None:
    # Configuração estática
    mode = "interface"  # "interface" | "rodada" | "estimativa"
    experiments = {
        "books": "2026-04-09_13-21-37",
        "dblp": "2026-04-09_14-05-21",
    }
    round_number = None  # None = próxima rodada; k = regenera a rodada k
    increment_size = 20  # documentos por grupo nas rodadas após a 1ª

    if mode == "interface":
        run_interface(experiments, increment_size)
    elif mode == "rodada":
        run_round(experiments, round_number, increment_size)
    elif mode == "estimativa":
        run_estimate(experiments)
    else:
        raise ValueError(f"mode inválido: {mode}")


if __name__ == "__main__":
    main()

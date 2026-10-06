"""
Auto Close Scheduler - Fecha sozinha a rodada ao fim da janela de revisão
"""
import asyncio

from fastapi.concurrency import run_in_threadpool
from loguru import logger

from src.systems.human_validation_system.interface.api.services.round_controller import RoundController


class AutoCloseScheduler:
    """
    Verifica periodicamente se alguma rodada completa passou da janela de revisão.

    Responsabilidades:
    - A cada `interval_seconds`, pedir ao RoundController para fechar as rodadas vencidas
      (o fechamento já calcula estimativas, planilha e abre a próxima rodada)
    - Nunca derrubar o servidor: erros vão para o log e a verificação continua
    """

    def __init__(self, controller: RoundController, interval_seconds: int):
        self.controller = controller
        self.interval_seconds = interval_seconds
        logger.debug(f"AutoCloseScheduler inicializado (a cada {interval_seconds}s)")

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.interval_seconds)
            try:
                closed = await run_in_threadpool(self.controller.close_due_rounds)
                if closed:
                    logger.success(f"Rodadas fechadas automaticamente: {closed}")
            except Exception as e:
                logger.error(f"Falha na verificação de fechamento automático: {e}")

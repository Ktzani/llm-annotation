"""
Job Runner - Executa os jobs da API em background e permite cancelá-los
"""

import asyncio
from typing import Coroutine, Dict


class JobRunner:
    """
    Gerencia os jobs em execução na API
    Responsabilidades: iniciar cada job como task asyncio, guardar a referência e cancelar sob demanda
    """

    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}

    def start(self, job_id: str, coro: Coroutine) -> asyncio.Task:
        """Inicia o job em background"""
        task = asyncio.create_task(coro, name=job_id)
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return task

    def is_running(self, job_id: str) -> bool:
        task = self._tasks.get(job_id)
        return task is not None and not task.done()

    def cancel(self, job_id: str) -> bool:
        """Solicita o cancelamento (uma vez só, para não interromper a limpeza); False se não está rodando"""
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        if not task.cancelling():
            task.cancel()
        return True

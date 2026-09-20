"""
Job Process - Executa um job pesado (ex.: fine-tuning) num processo separado da API
"""

import asyncio
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from typing import Any, AsyncIterator, Callable, Tuple

from loguru import logger


def _run_in_child(target: Callable[..., Any], conn, args: tuple) -> None:
    """Ponto de entrada do processo filho: executa o job e devolve progresso/resultado pelo pipe"""
    if hasattr(os, "setsid"):
        os.setsid()  # Grupo próprio: o kill alcança também os processos criados pelo job

    from src.api.core.config import setup_logger
    setup_logger()

    try:
        result = target(lambda **progress: conn.send(("progress", progress)), *args)
        conn.send(("result", result))
    except BaseException as e:
        logger.exception(f"Erro no processo do job: {e}")
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


class JobProcess:
    """
    Executa uma função num processo filho, fora do event loop da API
    Responsabilidades: iniciar o processo, repassar as mensagens dele e matá-lo junto com os filhos
    """

    def __init__(self, target: Callable[..., Any], *args: Any):
        """
        Args:
            target: função de módulo (picklable) chamada como target(report, *args);
                report(**progress) envia progresso para a API
        """
        ctx = mp.get_context("spawn")
        self._conn, self._child_conn = ctx.Pipe(duplex=False)
        self._process = ctx.Process(target=_run_in_child, args=(target, self._child_conn, args))

    def start(self) -> None:
        self._process.start()
        self._child_conn.close()  # Só o filho escreve; sem isso o recv não detecta o fim do processo

    async def messages(self) -> AsyncIterator[Tuple[str, Any]]:
        """Mensagens do job: ("progress", dict) durante a execução e ("result", valor) no fim"""
        while True:
            try:
                kind, payload = await asyncio.to_thread(self._conn.recv)
            except EOFError:
                await asyncio.to_thread(self._process.join)
                raise RuntimeError(f"Processo do job encerrou sem resultado (exit code {self._process.exitcode})")

            if kind == "error":
                raise RuntimeError(payload)
            yield kind, payload
            if kind == "result":
                return

    async def kill(self) -> None:
        """Mata o processo e os filhos dele (ex.: folds do cross-validation), liberando a GPU"""
        pid = self._process.pid
        if pid is None:
            return

        if sys.platform == "win32":
            if self._process.is_alive():
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                self._process.kill()  # Ainda não criou o grupo (nem filhos)

        await asyncio.to_thread(self._process.join)
        logger.info(f"Processo do job {pid} encerrado")

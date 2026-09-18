"""
Run Versioner - Versiona as execuções da anotação em 2 fases de um dataset
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class TwoPhaseRunVersioner:
    """
    Versiona as execuções da anotação em 2 fases em <results>/<dataset>/two_phase/<data>/
    Responsabilidades: criar ou retomar a pasta da execução, salvar a config e isolar o checkpoint
    """

    RUNS_DIR = "two_phase"
    CHECKPOINTS_DIR = "_checkpoints"
    CONFIG_FILE = "config.json"

    def __init__(self, dataset_dir: Path):
        self.runs_dir = Path(dataset_dir) / self.RUNS_DIR

    def get_run_dir(self, resume_from: Optional[str] = None) -> Path:
        """Execução nova (do zero) ou, com resume_from, a execução existente a retomar"""
        if resume_from:
            return self.resume_run_dir(resume_from)
        return self.create_run_dir()

    def create_run_dir(self) -> Path:
        """Cria a pasta <data> da execução (nunca reutiliza uma existente)"""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.runs_dir / stamp

        suffix = 1
        while True:
            try:
                run_dir.mkdir(parents=True)
                logger.info(f"Execução 2 fases: {run_dir.name}")
                return run_dir
            except FileExistsError:
                suffix += 1
                run_dir = self.runs_dir / f"{stamp}_{suffix}"

    def resume_run_dir(self, run_name: str) -> Path:
        """Pasta de uma execução existente, para retomar do seu checkpoint"""
        run_dir = self.runs_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Execução 2 fases não encontrada para retomar: {run_dir}")
        logger.info(f"Retomando execução 2 fases: {run_name}")
        return run_dir

    def save_config(self, run_dir: Path, config: Dict[str, Any]) -> Path:
        """Salva a config da execução; ao retomar, mantém a original e avisa se mudou"""
        config_path = Path(run_dir) / self.CONFIG_FILE
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                saved = json.load(f)
            if self._without_resume(saved) != self._without_resume(config):
                logger.warning(
                    f"Config diferente da usada em {Path(run_dir).name}: "
                    f"as anotações retomadas podem misturar configurações"
                )
            return config_path

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return config_path

    def checkpoint_dir(self, run_dir: Path) -> Path:
        """Checkpoint dentro da execução: execução nova começa do zero"""
        return Path(run_dir) / self.CHECKPOINTS_DIR

    @staticmethod
    def _without_resume(config: Dict[str, Any]) -> Dict[str, Any]:
        """Config sem o resume_from (que muda justamente ao retomar)"""
        class_filter = {k: v for k, v in config.get("class_filter", {}).items() if k != "resume_from"}
        return {**config, "class_filter": class_filter}

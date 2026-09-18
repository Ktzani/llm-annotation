"""
Run Versioner - Versiona as execuções da anotação em 2 fases de um dataset
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class TwoPhaseRunVersioner:
    """
    Versiona as execuções da anotação em 2 fases dentro de <results>/<dataset>/
    Responsabilidades: criar a pasta two_phase_<data>, salvar a config e isolar o checkpoint por config
    """

    RUN_PREFIX = "two_phase_"
    CHECKPOINTS_DIR = "_checkpoints_two_phase"
    CONFIG_FILE = "config.json"
    CHECKPOINT_CONFIG_FILE = "checkpoint_config.json"

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)

    def create_run_dir(self) -> Path:
        """Cria a pasta two_phase_<data> da execução (nunca reutiliza uma existente)"""
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.dataset_dir / f"{self.RUN_PREFIX}{stamp}"

        suffix = 1
        while True:
            try:
                run_dir.mkdir(parents=True)
                logger.info(f"Execução 2 fases: {run_dir.name}")
                return run_dir
            except FileExistsError:
                suffix += 1
                run_dir = self.dataset_dir / f"{self.RUN_PREFIX}{stamp}_{suffix}"

    def save_config(self, run_dir: Path, config: Dict[str, Any]) -> Path:
        """Salva a configuração usada na execução"""
        config_path = Path(run_dir) / self.CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return config_path

    def checkpoint_dir(self, k: int, key: Dict[str, Any]) -> Path:
        """
        Checkpoint estável por config: a mesma config retoma, config diferente começa do zero

        Args:
            k: Nº de classes candidatas (prefixo legível da pasta)
            key: Tudo que altera a anotação de um texto (modelos, params, prompt, filtro)
        """
        serialized = json.dumps(key, sort_keys=True, default=str)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:8]

        path = self.dataset_dir / self.CHECKPOINTS_DIR / f"k{k}_{digest}"
        path.mkdir(parents=True, exist_ok=True)

        config_path = path / self.CHECKPOINT_CONFIG_FILE
        if not config_path.exists():
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(key, f, indent=4, ensure_ascii=False, default=str)

        logger.info(f"Checkpoint 2 fases: {path.name}")
        return path

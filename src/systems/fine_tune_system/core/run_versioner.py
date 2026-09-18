"""
Run Versioner - Versiona as execuções de fine-tuning de uma mesma anotação
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger


class FineTuningRunVersioner:
    """
    Versiona as execuções de fine-tuning dentro de <anotação>/finetuning/
    Responsabilidades: gerar o nome vN_<nome>, criar a pasta sem sobrescrever e salvar a config usada
    """

    VERSION_PATTERN = re.compile(r"v(\d+)_")
    CONFIG_FILE = "config.json"

    def __init__(self, finetuning_dir: Path):
        self.finetuning_dir = Path(finetuning_dir)
        self.finetuning_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def default_run_name(
        model_name: str,
        training_mode: str,
        run_type: str,
        instance_selection_method: Optional[str] = None,
    ) -> str:
        """Nome descritivo: <modelo>_<modo>[_<seleção>][_single]"""
        parts = [model_name.split("/")[-1], training_mode]
        if instance_selection_method:
            parts.append(instance_selection_method)
        if run_type == "single":
            parts.append("single")
        return "_".join(parts)

    def next_version(self) -> int:
        """Próximo número de versão a partir das pastas vN_ existentes"""
        versions = [
            int(match.group(1))
            for path in self.finetuning_dir.iterdir()
            if path.is_dir() and (match := self.VERSION_PATTERN.match(path.name))
        ]
        return max(versions, default=0) + 1

    def create_run_dir(self, run_name: str) -> Path:
        """
        Cria a pasta vN_<run_name> da nova execução

        Returns:
            Caminho da pasta criada (nunca reutiliza uma existente)
        """
        slug = self._slugify(run_name)

        # exist_ok=False: dois jobs simultâneos nunca caem na mesma versão
        while True:
            run_dir = self.finetuning_dir / f"v{self.next_version()}_{slug}"
            try:
                run_dir.mkdir()
                logger.info(f"Versão do fine-tuning: {run_dir.name}")
                return run_dir
            except FileExistsError:
                continue

    def save_config(self, run_dir: Path, config: Dict[str, Any]) -> Path:
        """Salva a configuração usada na execução"""
        config_path = Path(run_dir) / self.CONFIG_FILE
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return config_path

    @staticmethod
    def _slugify(name: str) -> str:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-")
        if not slug:
            raise ValueError(f"run_name inválido: '{name}'")
        return slug

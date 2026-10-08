"""Selection Store - Versiona os artefatos da seleção de instâncias por método e parâmetros"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from src.config.instance_selection import INSTANCE_SELECTION_STRATEGIES
from src.systems.instance_selection_system.filtering.annotation_filter import (
    FilterResult,
    save_filter_result,
)


class InstanceSelectionStore:
    """
    Versiona a seleção de uma anotação em instance_selection/<método>_<params>_seed<N>/
    Responsabilidades: resolver os parâmetros efetivos, nomear a pasta, salvar e recarregar a seleção
    """

    FILTERED_FILE = "dataset_filtrado.csv"
    REPORT_FILE = "instance_selection_report.json"

    def __init__(
        self,
        base_dir: Path,
        method: str,
        params: Optional[Dict[str, Any]],
        random_state: int,
    ):
        self.method = method
        self.params = self.resolve_params(method, params)
        self.random_state = random_state
        self.run_dir = Path(base_dir) / self.run_name(method, self.params, random_state)
        logger.debug(f"InstanceSelectionStore: {self.run_dir}")

    @staticmethod
    def resolve_params(method: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Defaults do método (config central) sobrescritos pelos params informados"""
        defaults = {k: v for k, v in INSTANCE_SELECTION_STRATEGIES.get(method, {}).items() if k != "description"}
        return {**defaults, **(params or {})}

    @staticmethod
    def run_name(method: str, params: Dict[str, Any], random_state: int) -> str:
        """Nome determinístico: <método>_<param><valor>..._seed<N> (params em ordem alfabética)"""
        def fmt(value):
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
            return f"{value:g}" if is_number else str(value)

        parts = [method, *(f"{k}{fmt(v)}" for k, v in sorted(params.items())), f"seed{random_state}"]
        return "_".join(parts)

    def exists(self) -> bool:
        return (self.run_dir / self.FILTERED_FILE).exists()

    def load_filtered(self) -> pd.DataFrame:
        return pd.read_csv(self.run_dir / self.FILTERED_FILE)

    def load_report(self) -> Dict[str, Any]:
        with open(self.run_dir / self.REPORT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, result: FilterResult) -> None:
        save_filter_result(result, self.run_dir)

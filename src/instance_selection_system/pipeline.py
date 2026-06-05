"""
Pipeline de filtragem por Seleção de Instâncias (biO-IS).

Carrega o dataset anotado (consenso das LLMs) de um experimento, aplica a
filtragem de instâncias redundantes e ruidosas e salva o conjunto filtrado,
pronto para o fine-tuning supervisionado.

Estrutura de saída (em ``<results>/<dataset>/<date>/instance_selection/``):
    dataset_filtrado.csv             Conjunto limpo (instâncias mantidas)
    instancias_removidas.csv         Removidas, com a coluna `removal_reason`
    instancias_excluidas.csv         Sem consenso / rótulo inválido (se houver)
    instance_selection_report.json   Métricas da filtragem
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.instance_selection import DEFAULT_IS_METHOD, RANDOM_STATE
from src.instance_selection_system.filtering.annotation_filter import (
    AnnotationFilter,
    FilterResult,
    save_filter_result,
)
from src.utils.get_latest_results_date import get_latest_results_date

DEFAULT_RESULTS_DIR = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\results"


class InstanceSelectionConfig:
    """Configurações do experimento de seleção de instâncias."""

    def __init__(
        self,
        dataset_name: str = "books",
        results_dir: str = DEFAULT_RESULTS_DIR,
        specific_date: str = "latest",
        method: str = DEFAULT_IS_METHOD,
        params: dict = None,
        random_state: int = RANDOM_STATE,
    ):
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.specific_date = specific_date
        self.method = method
        # Parâmetros próprios do método de IS (ex.: {"beta": .., "theta": ..}).
        self.params = params
        self.random_state = random_state


class InstanceSelectionPipeline:
    """Pipeline principal de filtragem por seleção de instâncias."""

    def __init__(self, config: InstanceSelectionConfig):
        self.config = config
        self.results_dataset_path = self._get_results_path()
        self.output_dir = self.results_dataset_path / "instance_selection"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.success(f"✓ Setup completo — saída em: {self.output_dir}")

    def _get_results_path(self) -> Path:
        date = self.config.specific_date
        if date == "latest":
            date = get_latest_results_date(self.config.results_dir, self.config.dataset_name)
        return Path(self.config.results_dir) / self.config.dataset_name / date

    def load_annotated_data(self) -> pd.DataFrame:
        path = self.results_dataset_path / "summary" / "dataset_anotado_completo.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset anotado não encontrado: {path}")

        df = pd.read_csv(path)
        logger.info(f"Carregado: {len(df)} instâncias de {path}")
        return df

    def _build_filter(self) -> AnnotationFilter:
        return AnnotationFilter(
            method=self.config.method,
            random_state=self.config.random_state,
            **(self.config.params or {}),
        )

    def _save(self, result: FilterResult) -> None:
        save_filter_result(result, self.output_dir)

    def run(self) -> FilterResult:
        logger.info("=" * 60)
        logger.info(f"Seleção de instâncias [{self.config.method}] — {self.config.dataset_name}")
        logger.info("=" * 60)

        df = self.load_annotated_data()
        annotation_filter = self._build_filter()
        result = annotation_filter.filter(df)

        self._save(result)
        return result

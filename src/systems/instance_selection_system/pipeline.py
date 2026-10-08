"""
Pipeline de filtragem por Seleção de Instâncias (biO-IS).

Carrega o dataset anotado (consenso das LLMs) de um experimento, aplica a
filtragem de instâncias redundantes e ruidosas e salva o conjunto filtrado,
pronto para o fine-tuning supervisionado.

Cada combinação de parâmetros é salva em sua própria pasta
(``<results>/<dataset>/<date>/instance_selection/<método>_<params>_seed<N>/``):
    dataset_filtrado.csv             Conjunto limpo (instâncias mantidas)
    instancias_removidas.csv         Removidas, com a coluna `removal_reason`
    instancias_excluidas.csv         Sem consenso / rótulo inválido (se houver)
    instance_selection_report.json   Métricas da filtragem
"""
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from src.config.instance_selection import DEFAULT_IS_METHOD, RANDOM_STATE
from src.systems.instance_selection_system.filtering.annotation_filter import (
    AnnotationFilter,
    FilterResult,
)
from src.systems.instance_selection_system.filtering.selection_store import InstanceSelectionStore
from src.utils.get_latest_results_date import get_latest_results_date
from src.utils.get_text_id_from_text import get_text_id_from_text

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
        self.store = self.store_for(config.params)
        logger.success(f"✓ Setup completo — saída em: {self.store.run_dir}")

    def _get_results_path(self) -> Path:
        date = self.config.specific_date
        if date == "latest":
            date = get_latest_results_date(self.config.results_dir, self.config.dataset_name)
        return Path(self.config.results_dir) / self.config.dataset_name / date

    def store_for(self, params: Dict[str, Any]) -> InstanceSelectionStore:
        """Pasta versionada da seleção para um conjunto de params"""
        return InstanceSelectionStore(
            base_dir=self.output_dir,
            method=self.config.method,
            params=params,
            random_state=self.config.random_state,
        )

    def load_annotated_data(self) -> pd.DataFrame:
        """Consenso sem os casos problemáticos — o mesmo conjunto que o fine-tuning treina"""
        path = self.results_dataset_path / "consensus" / "dataset_consenso.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset de consenso não encontrado: {path}")

        df = pd.read_csv(path)
        logger.info(f"Carregado: {len(df)} instâncias de {path}")

        problematic_path = self.results_dataset_path / "consensus" / "problematic_cases.csv"
        if problematic_path.exists():
            problematic_ids = set(pd.read_csv(problematic_path)["text_id"])
            df = df[~df["text"].apply(get_text_id_from_text).isin(problematic_ids)].reset_index(drop=True)
            logger.info(f"Sem casos problemáticos: {len(df)} instâncias")
        return df

    def _filter_and_save(self, df: pd.DataFrame, store: InstanceSelectionStore) -> FilterResult:
        annotation_filter = AnnotationFilter(
            method=self.config.method,
            random_state=self.config.random_state,
            **store.params,
        )
        result = annotation_filter.filter(df)
        store.save(result)
        return result

    def run(self) -> FilterResult:
        logger.info("=" * 60)
        logger.info(f"Seleção de instâncias [{self.config.method}] — {self.config.dataset_name}")
        logger.info("=" * 60)

        return self._filter_and_save(self.load_annotated_data(), self.store)

    def load_or_run(self) -> pd.DataFrame:
        """Instâncias selecionadas: reutiliza a seleção salva com estes params ou executa e salva"""
        if self.store.exists():
            logger.info(f"Seleção reutilizada: {self.store.run_dir.name}")
            return self.store.load_filtered()
        return self.run().filtered_df

    def sweep(self, param: str, values: List[float]) -> pd.DataFrame:
        """Varia um parâmetro (demais fixos na config) e resume a retenção de cada seleção"""
        df = self.load_annotated_data()
        rows = []

        for value in values:
            store = self.store_for({**(self.config.params or {}), param: value})
            if not store.exists():
                self._filter_and_save(df, store)
            rows.append(self._sweep_row(store))

        summary = pd.DataFrame(rows)
        summary_path = self.output_dir / f"sweep_{param}.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        logger.success(f"Sweep salvo em: {summary_path}")
        return summary

    def _sweep_row(self, store: InstanceSelectionStore) -> Dict[str, Any]:
        stats = store.load_report()
        labeled = stats["labeled_instances"]
        beta = stats.get("beta")
        return {
            "dataset": self.config.dataset_name,
            "annotation": self.results_dataset_path.name,
            **store.params,
            "labeled_instances": labeled,
            "kept_instances": stats["kept_instances"],
            "retention": stats["kept_instances"] / labeled if labeled else None,
            "removed_redundant": stats["removed_redundant"],
            "removed_noise": stats["removed_noise"],
            # biO-IS só remove como redundante o que o classificador fraco acerta: acima disso o beta satura
            "beta_saturated": beta is not None and int(labeled * beta) > stats["removed_redundant"],
            "weak_classifier_accuracy": stats["weak_classifier_accuracy"],
            "selection_dir": store.run_dir.name,
        }

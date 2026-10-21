"""
Pipeline de consenso da anotação binária: o ConsensusPipeline existente sobre os itens (texto, classe)
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.get_latest_results_date import get_latest_results_date
from src.systems.llm_annotation_system.consensus.pipeline import ConsensusConfig, ConsensusPipeline
from src.systems.llm_annotation_system.binary_annotation.binary_llm_annotator import BinaryLLMAnnotator
from src.systems.llm_annotation_system.binary_annotation.binary_labels_builder import BinaryLabelsBuilder


class BinaryConsensusPipeline:
    """
    Consenso de uma execução da anotação binária
    Responsabilidades: apontar o ConsensusPipeline para binary/<data> e montar a planilha do consenso (binary_labels.csv)
    """

    LABELS_FILE = "binary_labels.csv"
    ROWS_FILE = "dataset_rows.csv"

    def __init__(self, config: ConsensusConfig):
        config.specific_date = self._binary_date(config)
        self.consensus = ConsensusPipeline(config)
        self.labels_builder = BinaryLabelsBuilder()
        logger.success(f"✓ Consenso binário em: {self.consensus.results_dataset_path}")

    @staticmethod
    def _binary_date(config: ConsensusConfig) -> str:
        """Pasta da execução binária: binary/<data>; 'latest' é a mais recente dentro de binary/"""
        subdir = BinaryLLMAnnotator.RESULTS_SUBDIR
        date = config.specific_date
        if date.startswith(f"{subdir}/"):
            return date
        if date == "latest":
            date = get_latest_results_date(str(Path(config.results_dir) / config.dataset_name), subdir)
        return f"{subdir}/{date}"

    @property
    def labels_path(self) -> Path:
        return self.consensus.results_dataset_path / self.LABELS_FILE

    def run(self) -> dict:
        """Roda o consenso existente (um item por texto × classe) e salva o consenso como planilha texto × classe"""
        result = self.consensus.run()

        df_labels = self.labels_builder.labels_by_text(result["df_with_consensus"], column="resolved_annotation")
        df_rows = pd.read_csv(self.consensus.results_dataset_path / self.ROWS_FILE)
        self.labels_builder.expand_to_rows(df_labels, df_rows).to_csv(self.labels_path, index=False)
        logger.success(f"✓ Planilha do consenso salva em: {self.labels_path}")

        return result

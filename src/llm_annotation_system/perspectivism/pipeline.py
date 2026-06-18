"""
Pipeline de geração do dataset de PERSPECTIVISMO.

Gera o dataset de perspectivismo (uma linha por anotação de LLM) a partir do
`dataset_consenso.csv` de um experimento, de forma independente do
fine-tuning. Assim o dataset pode ser materializado uma única vez e reutilizado
pelo fine-tuning (que pula a geração caso o arquivo já exista).

Estrutura de saída (em ``<results>/<dataset>/<date>/perspectivismo/``):
    dataset_perspectivismo.csv   Dataset longo: text_id, text, label, llm, label_description
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from src.utils.get_latest_results_date import get_latest_results_date
from src.llm_annotation_system.perspectivism.perspectivism_dataset_builder import PerspectivismDatasetBuilder

DEFAULT_RESULTS_DIR = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\results"


class PerspectivismConfig:
    """Configurações da geração do dataset de perspectivismo."""

    def __init__(
        self,
        dataset_name: str = "movie_review",
        results_dir: str = DEFAULT_RESULTS_DIR,
        specific_date: str = "latest",
        label_suffix: str = "_consensus",
        force: bool = False,
    ):
        self.dataset_name = dataset_name
        self.results_dir = results_dir
        self.specific_date = specific_date
        # Sufixo das colunas de rótulo por LLM (ex.: `<modelo>_consensus`).
        self.label_suffix = label_suffix
        # Regenera mesmo se o dataset já existir.
        self.force = force


class PerspectivismPipeline:
    """Pipeline principal de geração do dataset de perspectivismo."""

    def __init__(self, config: PerspectivismConfig):
        self.config = config
        self.results_dataset_path = self._get_results_path()
        self.output_dir = self.results_dataset_path / "perspectivismo"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.success(f"✓ Setup completo — saída em: {self.output_dir}")

    def _get_results_path(self) -> Path:
        date = self.config.specific_date
        if date == "latest":
            date = get_latest_results_date(self.config.results_dir, self.config.dataset_name)
        return Path(self.config.results_dir) / self.config.dataset_name / date

    def load_annotated_data(self) -> pd.DataFrame:
        path = self.results_dataset_path / "consensus" / "dataset_consenso.csv"
        if not path.exists():
            raise FileNotFoundError(f"Dataset de consenso não encontrado: {path}")

        df = pd.read_csv(path)
        logger.info(f"Carregado: {len(df)} instâncias de {path}")
        return df

    def run(self) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info(f"Perspectivismo — {self.config.dataset_name}")
        logger.info("=" * 60)

        builder = PerspectivismDatasetBuilder(
            dataset_name=self.config.dataset_name,
            label_suffix=self.config.label_suffix,
        )

        output_path = builder.output_path(self.output_dir)

        # Evita carregar o CSV completo se já houver dataset materializado.
        if output_path.exists() and not self.config.force:
            df = pd.read_csv(output_path)
            logger.info(
                f"Perspectivismo: dataset existente reutilizado "
                f"({len(df)} linhas): {output_path}"
            )
            return df

        df_full = self.load_annotated_data()
        return builder.build_and_save(df_full, self.output_dir)

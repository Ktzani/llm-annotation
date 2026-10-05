"""
Answer Key Writer - Gera o gabarito interno de cada rodada
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger


class AnswerKeyWriter:
    """
    Gera o gabarito que liga id_anonimo ao documento original e aos rótulos.

    Responsabilidades:
    - Montar as colunas do gabarito a partir da rodada já ordenada
    - Exibir rótulos de LLM e consolidado pelo nome da classe (mesmo vocabulário
      das planilhas); rotulo_referencia mantém o código numérico
    """

    COLUMNS = [
        "id_anonimo", "id_original", "grupo", "classe_referencia", "rotulo_referencia",
        "anotacao_llm_1", "anotacao_llm_2", "anotacao_llm_3", "rotulo_consolidado",
    ]
    FILE_NAME = "gabarito.csv"

    def __init__(
        self,
        label_names: Dict[int, str],
        llm_columns: List[str],
        reference_column: str = "ground_truth",
        consolidated_column: str = "resolved_annotation",
    ):
        self.label_names = label_names
        self.llm_columns = llm_columns
        self.reference_column = reference_column
        self.consolidated_column = consolidated_column
        logger.debug("AnswerKeyWriter inicializado")

    def build(self, sample: pd.DataFrame) -> pd.DataFrame:
        """Gabarito na mesma ordem da planilha (`sample` já ordenado e com id_anonimo)."""
        key = pd.DataFrame({
            "id_anonimo": sample["id_anonimo"].values,
            "id_original": sample["text_id"].values,
            "grupo": sample["grupo"].values,
            "classe_referencia": sample[self.reference_column].map(self.label_names).values,
            "rotulo_referencia": sample[self.reference_column].values,
        })
        for i, col in enumerate(self.llm_columns, start=1):
            key[f"anotacao_llm_{i}"] = sample[col].map(self.label_names).values
        key["rotulo_consolidado"] = sample[self.consolidated_column].map(self.label_names).values
        return key[self.COLUMNS]

    def write(self, sample: pd.DataFrame, output_dir: Path) -> Path:
        path = Path(output_dir) / self.FILE_NAME
        self.build(sample).to_csv(path, index=False)
        logger.success(f"Gabarito salvo: {path}")
        return path

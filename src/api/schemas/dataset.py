from pydantic import BaseModel, Field
from typing import Optional, List


class DatasetConfig(BaseModel):
    """
    Configuração de carregamento e amostragem do dataset.
    """

    split: str = Field(
        default="train",
        description="Split principal do dataset HuggingFace a ser utilizado (ex: train, test, validation)."
    )

    combine_splits: Optional[List[str]] = Field(
        default=None,
        description="Lista de splits a serem combinados em um único dataset (ex: ['train', 'test'])."
    )

    sample_size: Optional[int] = Field(
        default=None,
        description="Número máximo de exemplos a serem carregados. Use None para carregar todo o dataset."
    )

    random_state: int = Field(
        default=42,
        description="Seed para reprodutibilidade da amostragem do dataset."
    )

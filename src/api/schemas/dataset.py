from pydantic import BaseModel, Field
from typing import Optional, List

class DatasetRemovedTextsConfig(BaseModel):
    """
    Configurações específicas para remoção de textos já anotados.
    """
    remove: bool = Field(
        default=False,
        description="Indica se os textos já anotados devem ser removidos do dataset."
    )

    annotated_texts_path: Optional[str] = Field(
        default=None,
        description="Caminho para o CSV contendo os textos já anotados. Necessário se remove_annotated_texts for True."
    )

class DatasetConfig(BaseModel):
    """
    Configuração de carregamento e amostragem do dataset.
    """

    split: Optional[str] = Field(
        default=None,
        description="Split principal do dataset HuggingFace a ser utilizado (ex: train, test, validation)."
    )

    hf_file: Optional[str] = Field(
        default=None,
        description="Nome do arquivo específico a ser baixado do HuggingFace Hub (ex: data.parquet)."
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
    
    remove_texts: DatasetRemovedTextsConfig = Field(
        default_factory=DatasetRemovedTextsConfig,
        description="Configurações para remoção de textos já anotados."
    )
    
    


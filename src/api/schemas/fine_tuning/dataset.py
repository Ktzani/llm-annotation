from pydantic import BaseModel, Field

class FineTuningDatasetConfig(BaseModel):
    """Configuração de localização dos dados para fine-tuning"""

    dataset_name: str = Field(..., description="Nome do dataset (ex: 'movie_review')")
    cache_dir: str = Field(..., description="Diretório de cache do HuggingFace")
    results_dir: str = Field(..., description="Diretório com resultados das anotações LLM")
    specific_date: str = Field(
        default="latest",
        description="Data específica dos resultados de anotação ou 'latest'",
    )

from pydantic import BaseModel, Field
from typing import Optional, List

class DatasetConfig(BaseModel):
    split: str = Field(default="train", description="Split do dataset")
    combine_splits: Optional[List[str]] = Field(
        default=None, 
        description="Combinar múltiplos splits"
    )
    sample_size: Optional[int] = Field(
        default=None, 
        description="Tamanho da amostra (None = dataset completo)"
    )
    random_state: int = Field(default=42, description="Seed para reprodutibilidade")
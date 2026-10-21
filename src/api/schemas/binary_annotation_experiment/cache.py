from pydantic import BaseModel, Field

class BinaryCacheConfig(BaseModel):
    """
    Diretório de cache da anotação binária (sem cache de respostas das LLMs: o intermediate.csv já guarda tudo).
    """

    dir: str = Field(
        default=r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\.cache",
        description="Diretório de cache dos datasets baixados do HuggingFace Hub."
    )

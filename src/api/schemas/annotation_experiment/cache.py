from pydantic import BaseModel, Field

class CacheConfig(BaseModel):
    """
    Configuração do sistema de cache das chamadas às LLMs.
    """

    enabled: bool = Field(
        default=True,
        description="Habilita ou desabilita o uso de cache para respostas das LLMs."
    )

    dir: str = Field(
        default=r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\.cache",
        description="Diretório onde o cache (ex: SQLite) será armazenado."
    )
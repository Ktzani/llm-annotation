from pydantic import BaseModel, Field

class ResultsConfig(BaseModel):
    """
    Configuração de persistência dos resultados do experimento.
    """
    
    save_model_metrics: bool = Field(
        default=True,
        description="Se True, salva as metricas de erro de cada modelo invidualmente"
    )

    save_intermediate: bool = Field(
        default=True,
        description="Se True, salva resultados intermediários durante a anotação."
    )

    intermediate: int = Field(
        default=100,
        description="Intervalo (em número de textos) para salvar resultados intermediários."
    )

    dir: str = Field(
        default=r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\results",
        description="Diretório base onde os resultados do experimento serão salvos."
    )
    
    
from pydantic import BaseModel, Field
from src.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy


class AnnotationConfig(BaseModel):
    """
    Configuração do processo de anotação com LLMs.
    """

    num_repetitions_per_llm: int = Field(
        default=1,
        ge=1,
        description="Número de repetições da anotação para cada LLM (usado para consenso interno)."
    )

    use_alternative_params: bool = Field(
        default=False,
        description="Se True, utiliza parâmetros alternativos de geração (ex: temperatura, top-p, etc)."
    )

    model_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução entre diferentes modelos (sequential ou parallel)."
    )

    rep_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução entre repetições do mesmo modelo."
    )

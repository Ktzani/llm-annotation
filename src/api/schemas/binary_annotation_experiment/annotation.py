from pydantic import BaseModel, Field, field_validator
from src.systems.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy
from src.systems.llm_annotation_system.core.model_variants import AlternativeParamsSelection, ModelVariantResolver


class BinaryAnnotationConfig(BaseModel):
    """
    Configuração do processo de anotação binária (uma pergunta sim/não por texto × classe × modelo, sem repetições).
    """

    use_alternative_params: AlternativeParamsSelection = Field(
        default=False,
        description=(
            "Quais `alternative_params` (src/config/llms.py) usar em cada modelo — mesmo formato da "
            "anotação multiclasse: false | \"alt2\" | [\"base\", \"alt2\"] | {\"llama3.1-8b\": \"alt2\"} | true."
        ),
        examples=[False, "alt2", ["base", "alt1"], {"llama3.1-8b": "alt2"}],
    )

    @field_validator("use_alternative_params")
    @classmethod
    def _validate_variant_names(cls, value: AlternativeParamsSelection) -> AlternativeParamsSelection:
        """Valida o formato das variações ("base"/"altN")"""
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            return {model: ModelVariantResolver.normalize_variants(spec) for model, spec in value.items()}
        return ModelVariantResolver.normalize_variants(value)

    model_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução entre diferentes modelos (sequential ou parallel)."
    )

    class_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução entre as classes de um mesmo texto (no lugar das repetições da anotação multiclasse)."
    )

    max_concurrent_classes: int = Field(
        default=5,
        ge=1,
        description=(
            "Número máximo de classes perguntadas ao mesmo tempo, por modelo (com class_strategy=parallel). "
            "Os textos vão um por vez; com model_strategy=parallel, ficam até modelos × este valor chamadas "
            "simultâneas (mantenha OLLAMA_NUM_PARALLEL ≥ esse total)."
        )
    )

    keep_alive: int | str | None = Field(
        default=None,
        description="Tempo que o Ollama mantém o modelo carregado (ex.: \"24h\")."
    )

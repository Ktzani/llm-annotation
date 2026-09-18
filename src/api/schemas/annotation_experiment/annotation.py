from pydantic import BaseModel, Field, field_validator
from src.systems.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy
from src.systems.llm_annotation_system.core.model_variants import AlternativeParamsSelection, ModelVariantResolver


class AnnotationConfig(BaseModel):
    """
    Configuração do processo de anotação com LLMs.
    """

    num_repetitions_per_llm: int = Field(
        default=1,
        ge=1,
        description="Número de repetições da anotação para cada LLM (usado para consenso interno)."
    )

    use_alternative_params: AlternativeParamsSelection = Field(
        default=False,
        description=(
            "Quais `alternative_params` (src/config/llms.py) usar em cada modelo. "
            "A variação N roda como '<modelo>_altN' (nome das colunas no CSV); 'base' são os params normais. "
            "false: só base | \"alt2\": todos os modelos com alt2 | [\"base\", \"alt2\"]: base + alt2 | "
            "{\"llama3.1-8b\": \"alt2\", \"qwen3-8b\": [\"base\", \"alt1\"]}: por modelo "
            "(os não citados rodam base) | true: base + todas as variações."
        ),
        examples=[False, "alt2", ["base", "alt1"], {"llama3.1-8b": "alt2", "qwen3-8b": ["base", "alt1"]}],
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

    rep_strategy: ExecutionStrategy = Field(
        default=ExecutionStrategy.PARALLEL,
        description="Estratégia de execução entre repetições do mesmo modelo."
    )
    
    max_concurrent_texts: int = Field(
        default=4,
        ge=1,
        description=(
            "Número máximo de textos a serem processados simultaneamente. "
            "Útil para controlar o uso de recursos e evitar sobrecarga."
        )
    )

    keep_alive: int | str | None = Field(
        default=None,
        description=(
            "Tempo em segundos para manter conexões ativas (keep-alive) com a LLM. "
            "Útil para reduzir latência em múltiplas requisições. "
            "0 significa desativado."
        )
    )

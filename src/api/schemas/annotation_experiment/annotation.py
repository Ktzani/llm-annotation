from typing import Dict, List, Optional

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

    few_shot_examples: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Exemplos rotulados do prompt_type='few_shot', na ordem em que aparecem no prompt. "
            "Cada item é {\"text\": <texto>, \"category\": <código ou nome da classe>}. "
            "Os mesmos exemplos valem para todos os modelos e textos."
        ),
        examples=[[{"text": "Cobrança duplicada no cartão", "category": "0"},
                   {"text": "O sistema não abre", "category": "suporte técnico"}]],
    )

    @field_validator("few_shot_examples")
    @classmethod
    def _validate_examples(cls, value):
        """Cada exemplo precisa de 'text' e 'category'"""
        for i, example in enumerate(value or []):
            faltando = {"text", "category"} - set(example)
            if faltando:
                raise ValueError(f"few_shot_examples[{i}]: campo(s) {sorted(faltando)} ausente(s)")
        return value

import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from src.systems.llm_annotation_system.core.model_variants import ModelVariantResolver
from src.utils.local_datasets import local_datasets_dir

from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.api.schemas.annotation_experiment.results import ResultsConfig
from src.api.schemas.binary_annotation_experiment.annotation import BinaryAnnotationConfig
from src.api.schemas.binary_annotation_experiment.cache import BinaryCacheConfig
from src.api.schemas.binary_annotation_experiment.class_prompts import ClassPrompt


class BinaryAnnotationRequest(BaseModel):
    """
    Schema principal de um experimento de anotação binária com LLMs.

    Cada texto recebe uma pergunta sim/não por classe (com o prompt daquela classe) para cada
    modelo; o voto majoritário entre os modelos define o sim/não de cada classe. O resultado é o
    conjunto de sim/não de cada texto (não há rótulo único).
    """

    dataset_name: str = Field(
        ...,
        description=(
            "Nome do dataset registrado: HuggingFace Hub (datasets_collected.py) ou local "
            "(pasta em data/datasets/<nome>/ com dataset.json)."
        )
    )

    dataset_config: DatasetConfig = Field(
        default_factory=DatasetConfig,
        description="Configurações de carregamento do dataset (split, amostragem, seed)."
    )

    models: List[str] = Field(
        default=[
            "deepseek-r1-8b",
            "qwen3-8b",
            "gemma3-4b",
            "mistral-7b",
            "llama3.1-8b"
        ],
        description="Modelos LLM que respondem às perguntas sim/não."
    )

    keep_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Colunas do dataset copiadas para o binary_labels.csv (ex.: [\"message_id\"]). Com alguma delas, "
            "a saída tem uma linha por linha do dataset; colunas que não existirem no dataset são ignoradas com aviso."
        ),
        examples=[["message_id"]],
    )

    class_prompts_file: str = Field(
        default="class_prompts.json",
        description=(
            "JSON com o prompt de cada classe ({\"<código ou nome da classe>\": <prompt>}). "
            "Caminho relativo à pasta data/datasets/<dataset_name>/ ou absoluto. "
            "Ignorado quando class_prompts vem no próprio experimento."
        )
    )

    class_prompts: Optional[Dict[str, ClassPrompt]] = Field(
        default=None,
        description=(
            "Prompts das classes no próprio experimento (mesmo formato do class_prompts_file). "
            "Cada valor é o prompt completo (string com {text}) ou {\"description\", \"examples\"}."
        )
    )

    custom_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Template próprio no lugar do BINARY_CLASS_PROMPT, para as classes descritas por description/examples. "
            "Marcadores: {text}, {class_name}, {class_code}, {class_description}, {examples}, {description}, {description_lower}; "
            "outras chaves (ex.: o JSON do formato de saída) ficam como texto."
        )
    )

    annotation: BinaryAnnotationConfig = Field(
        default_factory=BinaryAnnotationConfig,
        description="Configurações do processo de anotação (variações de parâmetros, estratégia e paralelismo)."
    )

    cache: BinaryCacheConfig = Field(
        default_factory=BinaryCacheConfig,
        description="Diretório de cache dos datasets (não há cache de respostas das LLMs)."
    )

    results: ResultsConfig = Field(
        default_factory=ResultsConfig,
        description="Persistência dos resultados: salvamento intermediário (a cada N respostas) e diretório de saída."
    )

    @model_validator(mode="after")
    def _validate_alternative_params(self) -> "BinaryAnnotationRequest":
        """Valida se as variações escolhidas existem para os modelos"""
        ModelVariantResolver(self.annotation.use_alternative_params).resolve(self.models)
        return self

    @model_validator(mode="after")
    def _load_class_prompts(self) -> "BinaryAnnotationRequest":
        """Carrega os prompts das classes do class_prompts_file quando não vêm no experimento"""
        if self.class_prompts is not None:
            return self

        path = Path(self.class_prompts_file)
        if not path.is_absolute():
            path = local_datasets_dir() / self.dataset_name / path
        if not path.is_file():
            raise ValueError(
                f"Prompts das classes não encontrados: {path}. "
                f"Crie o arquivo ou envie 'class_prompts' no experimento."
            )

        with open(path, "r", encoding="utf-8") as f:
            self.class_prompts = TypeAdapter(Dict[str, ClassPrompt]).validate_python(json.load(f))
        return self

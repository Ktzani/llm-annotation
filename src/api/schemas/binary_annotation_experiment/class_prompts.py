from typing import List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.systems.llm_annotation_system.core.binary_response_processor import BinaryResponseProcessor


class BinaryExample(BaseModel):
    """
    Exemplo rotulado do prompt de uma classe.
    """

    text: str = Field(
        ...,
        description="Texto do exemplo."
    )

    answer: str = Field(
        ...,
        description="Se o texto pertence à classe: \"yes\"/\"no\" (aceita também \"sim\"/\"não\"). Entra no prompt como escrito."
    )

    @field_validator("answer")
    @classmethod
    def _validate_answer(cls, value: str) -> str:
        """A resposta do exemplo precisa ser sim/não"""
        if BinaryResponseProcessor.parse_answer(value) == -1:
            raise ValueError(f"answer '{value}' inválido (use \"yes\"/\"no\" ou \"sim\"/\"não\")")
        return value.strip()


class ClassPrompt(BaseModel):
    """
    Prompt de uma classe na anotação binária.

    Duas formas:
    - `prompt`: prompt completo da classe (precisa do marcador {text}); ou
    - `description` (+ `examples`): entram no template padrão (BINARY_CLASS_PROMPT ou custom_prompt).
    Cada exemplo é um texto livre (situação que ilustra a classe) ou {"text", "answer"}.
    """

    prompt: Optional[str] = Field(
        default=None,
        description="Prompt completo da classe, com {text} onde entra o texto. Se definido, ignora description/examples."
    )

    description: Optional[str] = Field(
        default=None,
        description="Descrição da classe, renderizada no {class_description} do template."
    )

    examples: List[Union[str, BinaryExample]] = Field(
        default_factory=list,
        description=(
            "Exemplos da classe, um por linha no {examples} do template: texto livre vira '- <texto>' e "
            "{\"text\", \"answer\"} vira '- Mensagem: <text> → Resposta: <answer>'."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def _from_string(cls, value):
        """Aceita o prompt completo direto como string no lugar do objeto"""
        return {"prompt": value} if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_prompt(self) -> "ClassPrompt":
        """Exige o prompt completo (com {text}) ou a descrição da classe"""
        if self.prompt:
            if "{text}" not in self.prompt:
                raise ValueError("o prompt da classe precisa do marcador {text}")
        elif not self.description:
            raise ValueError("informe 'prompt' (completo) ou 'description' (+ 'examples') da classe")
        return self

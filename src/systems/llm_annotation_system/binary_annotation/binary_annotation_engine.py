"""
Binary Annotation Engine - Motor da anotação binária (uma pergunta sim/não por classe)
"""

from typing import Dict, List
from loguru import logger
import re

from src.systems.llm_annotation_system.core.llm_provider import LLMProvider
from src.systems.llm_annotation_system.core.cache_manager import CacheManager
from src.systems.llm_annotation_system.annotation.annotation_engine import AnnotationEngine
from src.systems.llm_annotation_system.core.binary_response_processor import BinaryResponseProcessor

from src.config.datasets_collected import DATASETS
from src.config.prompts import BINARY_CLASS_PROMPT


class BinaryAnnotationEngine(AnnotationEngine):
    """
    Motor de anotação binária
    Responsabilidades: montar o prompt de cada classe e perguntar à LLM se o texto pertence a ela
    """

    _PLACEHOLDER = re.compile(r"\{(\w+)\}")
    _TEXT_SLOT = "<<__TEXT_SLOT__>>"

    def __init__(
        self,
        llm_provider: LLMProvider,
        cache_manager: CacheManager,
        response_processor: BinaryResponseProcessor,
        dataset_name: str,
        class_prompts: Dict,
        prompt_template: str = BINARY_CLASS_PROMPT,
    ):
        """
        Args:
            class_prompts: {código ou nome da classe: ClassPrompt}, uma entrada por classe do dataset
            prompt_template: Template das classes descritas por description/examples
        """
        # A classe-mãe popula self._categories_indexed (o template multiclasse dela não é usado aqui)
        super().__init__(llm_provider, cache_manager, response_processor, dataset_name)

        self.class_names = {int(code): name for code, name in self._categories_indexed.items()}
        self.class_templates = self._prepare_class_templates(class_prompts, prompt_template)
        logger.info(f"Templates binários preparados: {len(self.class_templates)} classes")

    async def annotate(self, text: str, class_code: int, model: str, llm: any) -> dict:
        """Pergunta à LLM se o texto pertence à classe (uma chamada, sem cache nem repetições)"""
        template = self.class_templates[class_code]

        if isinstance(llm, dict) and llm.get("provider") == "ollama":
            chain = llm
        else:
            chain = self.llm_provider.create_chain(llm=llm, template=template)

        try:
            response = await self._ainvoke_chain(chain, text, template)
            return self.response_processor.extract_label_and_confidence(response)
        except Exception as e:
            # Degrada para resposta inválida (-1) em vez de quebrar o run inteiro
            logger.error(f"Erro em {model} classe {class_code}: {str(e)}", exc_info=True)
            return {"label": -1, "confidence": None}

    def _prepare_class_templates(self, class_prompts: Dict, prompt_template: str) -> Dict[int, str]:
        """Um template por classe do dataset (com {text} ainda como placeholder)"""
        name_to_code = {str(name).strip().lower(): code for code, name in self.class_names.items()}

        by_code = {}
        for key, class_prompt in class_prompts.items():
            key = str(key).strip()
            code = int(key) if key.lstrip("-").isdigit() and int(key) in self.class_names else name_to_code.get(key.lower())
            if code is None:
                raise ValueError(
                    f"Classe '{key}' do class_prompts não existe no dataset '{self.dataset_name}'. "
                    f"Classes: {self.class_names}"
                )
            if code in by_code:
                raise ValueError(f"Classe {code} ({self.class_names[code]}) aparece duas vezes no class_prompts")
            by_code[code] = class_prompt

        missing = [f"{code}: {name}" for code, name in self.class_names.items() if code not in by_code]
        if missing:
            raise ValueError(f"class_prompts sem prompt para {len(missing)} classe(s): {missing}")

        return {code: self._render_class_template(code, by_code[code], prompt_template) for code in self.class_names}

    def _render_class_template(self, code: int, class_prompt, prompt_template: str) -> str:
        """Preenche os marcadores com os dados da classe; o resto do texto (inclusive chaves de JSON) fica literal"""
        description = DATASETS.get(self.dataset_name, {}).get("prompt", "Text")
        values = {
            "description": description,
            "description_lower": description.lower(),
            "class_code": str(code),
            "class_name": str(self.class_names[code]),
            "class_description": class_prompt.description or "",
            "examples": self._render_binary_examples(class_prompt.examples),
        }

        def fill(match: re.Match) -> str:
            name = match.group(1)
            if name == "text":
                return self._TEXT_SLOT
            if name not in values:
                raise ValueError(
                    f"Prompt da classe {code} ({self.class_names[code]}) com marcador inválido: {{{name}}}. "
                    f"Marcadores disponíveis: {sorted(['text', *values])}"
                )
            return values[name]

        rendered = self._PLACEHOLDER.sub(fill, class_prompt.prompt or prompt_template)
        if self._TEXT_SLOT not in rendered:
            raise ValueError(f"Prompt da classe {code} ({self.class_names[code]}) sem o marcador {{text}}")

        # Só o {text} fica como variável do ChatPromptTemplate: as demais chaves viram literais
        return rendered.replace("{", "{{").replace("}", "}}").replace(self._TEXT_SLOT, "{text}")

    def _render_binary_examples(self, examples: List) -> str:
        """Um exemplo por linha: o texto como escrito ou 'Mensagem: ... → Resposta: ...'"""
        return "\n".join(
            f"- {example}" if isinstance(example, str) else f"- Mensagem: {example.text} → Resposta: {example.answer}"
            for example in examples
        )

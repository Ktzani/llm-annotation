"""
Annotation Engine - Motor de anotação
"""

import time
from typing import List, Dict, Optional
from loguru import logger

from src.llm_annotation_system.core.llm_provider import LLMProvider
from src.llm_annotation_system.core.cache_manager import CacheManager
from src.llm_annotation_system.core.response_processor import ResponseProcessor

from src.config.prompts import BASE_ANNOTATION_PROMPT, FEW_SHOT_PROMPT

class AnnotationEngine:
    """
    Motor de anotação
    Responsabilidades: anotar textos, gerenciar repetições, coordenar componentes
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        cache_manager: CacheManager,
        response_processor: ResponseProcessor
    ):
        """
        Args:
            llm_provider: Provedor de LLMs
            cache_manager: Gerenciador de cache
            response_processor: Processador de respostas
        """
        self.llm_provider = llm_provider
        self.cache = cache_manager
        self.processor = response_processor
        logger.debug("AnnotationEngine inicializado")
    
    def annotate_single(
        self,
        text: str,
        model: str,
        llm: any,
        num_repetitions: int = 1,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
        use_cache: bool = True
    ) -> List[str]:
        """
        Anota um texto com múltiplas repetições
        
        Args:
            text: Texto para anotar
            model: Nome do modelo
            llm: Instância da LLM
            num_repetitions: Número de repetições
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
            use_cache: Se True, usa cache
            
        Returns:
            Lista de classificações
        """
        classifications = []
        
        # Preparar template
        template = self._prepare_template(prompt_template, examples)
        
        # Criar chain
        chain = self.llm_provider.create_chain(
            llm=llm,
            template=template,
            variables={"text": text}
        )
        
        for rep in range(num_repetitions):
            try:
                # Verificar cache
                cache_key = self.cache.get_key(model, text, {"rep": rep})
                
                if use_cache:
                    cached = self.cache.get(cache_key)
                    if cached:
                        response = cached
                        logger.debug(f"{model} rep {rep+1}: cache hit")
                    else:
                        response = self._invoke_chain(chain, text)
                        self.cache.set(cache_key, response)
                        logger.debug(f"{model} rep {rep+1}: cache miss")
                else:
                    response = self._invoke_chain(chain, text)
                
                # Extrair categoria
                category = self.processor.extract_category(response)
                classifications.append(category)
                
                # Rate limiting
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Erro em {model} rep {rep+1}: {str(e)}")
                classifications.append("ERROR")
        
        return classifications
    
    def _prepare_template(
        self,
        prompt_template: str,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Prepara template do prompt com categorias específicas do dataset.

        Args:
            prompt_template: Template base
            examples: Exemplos para few-shot

        Returns:
            Template formatado
        """

        # Se categories vier em lista -> converter para dicionário indexado
        if isinstance(self.processor.categories, list):
            categories_indexed = {
                str(i): cat for i, cat in enumerate(self.processor.categories)
            }
        else:
            categories_indexed = self.processor.categories  # já é dict

        # Criar string numerada
        categories_str = "\n".join([
            f"- {idx}: {label}"
            for idx, label in categories_indexed.items()
        ])

        # Few-shot
        if examples and prompt_template == FEW_SHOT_PROMPT:
            examples_str = "\n\n".join([
                f"Text: {ex['text']}\nCategory: {ex['category']}"
                for ex in examples
            ])
            return prompt_template.format(
                examples=examples_str,
                text="{text}",
                categories=categories_str
            )

        # Base / COT
        return prompt_template.format(
            text="{text}",
            categories=categories_str
        )

    
    def _invoke_chain(self, chain: any, text: str) -> str:
        """
        Invoca chain e retorna resposta
        
        Args:
            chain: Chain configurada
            text: Texto para anotar
            
        Returns:
            Resposta da LLM
        """
        return chain.invoke({"text": text})

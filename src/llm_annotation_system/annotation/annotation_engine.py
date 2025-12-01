"""
Annotation Engine - Motor de anotação
"""

import time
from typing import List, Dict, Optional
from loguru import logger
import traceback

from src.llm_annotation_system.core.llm_provider import LLMProvider
from src.llm_annotation_system.core.cache_manager import CacheManager
from src.llm_annotation_system.core.response_processor import ResponseProcessor

from src.config.datasets_collected import LABEL_MEANINGS
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
        response_processor: ResponseProcessor,
        dataset_name: str,
        prompt_template: str = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
    ):
        """
        Args:
            llm_provider: Provedor de LLMs
            cache_manager: Gerenciador de cache
            response_processor: Processador de respostas
        """
        self.llm_provider = llm_provider
        self.cache_manager = cache_manager
        self.response_processor = response_processor
        self.dataset_name = dataset_name
        logger.debug(f"AnnotationEngine inicializado para dataset: {dataset_name}")
        
        # Preparar template
        self.template = self._prepare_template(prompt_template, examples)
        logger.info("Template do prompt preparado")
    
    def annotate_single(
        self,
        text: str,
        model: str,
        llm: any,
        num_repetitions: int = 1,
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
        
        # Criar chain
        chain = self.llm_provider.create_chain(
            llm=llm,
            template=self.template,
        )
        
        for rep in range(num_repetitions):
            try:
                # Verificar cache
                cache_key = self.cache_manager.get_key(model, text, {"rep": rep})
                
                if use_cache:
                    cached = self.cache_manager.get(cache_key)
                    if cached:
                        response = cached
                        logger.debug(f"{model} rep {rep+1}: cache hit")
                    else:
                        response = self._invoke_chain(chain, text)
                        self.cache_manager.set(cache_key, response)
                        logger.debug(f"{model} rep {rep+1}: cache miss")
                else:
                    response = self._invoke_chain(chain, text)
                
                # Extrair categoria
                category = self.response_processor.extract_category(response)
                classifications.append(category)
                
                # Rate limiting
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(
                    f"Erro em {model} rep {rep+1}: {str(e)}\n"
                    f"{traceback.format_exc()}"
                )
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

        # Primeiro tentamos obter a descrição a partir de LABEL_MEANINGS
        if self.dataset_name in LABEL_MEANINGS:
            categories_indexed = LABEL_MEANINGS[self.dataset_name]
    
        # Se não tiver em LABEL_MEANINGS, usamos categories do processor
        elif isinstance(self.response_processor.categories, list):
            categories_indexed = {
                str(i): cat for i, cat in enumerate(self.response_processor.categories)
            }
        else:
            categories_indexed = self.response_processor.categories
    
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

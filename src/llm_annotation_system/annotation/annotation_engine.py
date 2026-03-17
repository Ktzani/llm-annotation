"""
Annotation Engine - Motor de anotação
"""

from typing import List, Dict, Optional
from loguru import logger
import asyncio
import httpx
import time

from src.llm_annotation_system.core.llm_provider import LLMProvider
from src.llm_annotation_system.core.cache_manager import CacheManager
from src.llm_annotation_system.core.response_processor import ResponseProcessor
from src.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy

from src.config.datasets_collected import DATASETS, LABEL_MEANINGS
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
        
    async def _annotate_rep(
        self,
        chain: any,
        text: str,
        model: str,
        rep: int,
        use_cache: bool
    ) -> dict:
        try:

    
            
            cache_key = self.cache_manager.get_key(model, text, {"rep": rep})

            if use_cache:
                cached = self.cache_manager.get(cache_key)
                if cached:
                    response = cached
                    logger.debug(f"{model} rep {rep+1}: cache hit")
                else:
                    response = await self._ainvoke_chain(chain, text)
                    self.cache_manager.set(cache_key, response)
                    logger.debug(f"{model} rep {rep+1}: cache miss")
            else:
                response = await self._ainvoke_chain(chain, text)
     
            result = self.response_processor.extract_label_and_confidence(response)
            return result 

        except Exception as e:
            logger.error(
                f"Erro em {model} rep {rep+1}: {str(e)}",
                exc_info=True
            )
            return "ERROR"
    
    async def annotate(
        self,
        text: str,
        model: str,
        llm: any,
        num_repetitions: int = 1,
        use_cache: bool = True,
        rep_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
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
        if isinstance(llm, dict) and llm.get("provider") == "ollama":
            chain = llm
        else:
            chain = self.llm_provider.create_chain(
                llm=llm,
                template=self.template,
            )
        
        # ===============================
        # 🔁 SEQUENCIAL
        # ===============================
        if rep_strategy == ExecutionStrategy.SEQUENTIAL:
            for rep in range(num_repetitions):
                result = await self._annotate_rep(
                    chain, text, model, rep, use_cache
                )
                classifications.append(result)
            return classifications

        # ===============================
        # 🚀 PARALELO
        # ===============================
        elif rep_strategy == ExecutionStrategy.PARALLEL:
            tasks = [
                self._annotate_rep(chain, text, model, rep, use_cache)
                for rep in range(num_repetitions)
            ]
            return await asyncio.gather(*tasks)
        
        else: 
            raise ValueError(
                f"rep_strategy inválida: {rep_strategy}. "
                f"Use ExecutionStrategy.SEQUENTIAL ou ExecutionStrategy.PARALLEL."
            )
    
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
        

        description = DATASETS.get(self.dataset_name, {}).get("prompt", "Text")
        return prompt_template.format(
            description=description,
            description_lower=description.lower(),
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
    
    async def _ainvoke_chain(self, chain: any, text: str):
        # -----------------------------
        # OLLAMA VIA API
        # -----------------------------
        if isinstance(chain, dict) and chain.get("provider") == "ollama":

            prompt = self.template.format(text=text)

            payload = {
                "model": chain["model_name"],
                "prompt": prompt,
                "options": { **chain.get("params", {}) },
                "logprobs": chain.get("logprobs", True), 
                "stream": False,
                "keep_alive": chain.get("keep_alive", None)
            }

            try:
                
                t0 = time.perf_counter()
                
                r = await self.llm_provider.client.post(
                    f"{chain['base_url']}/api/generate",
                    json=payload
                )
                
                t1 = time.perf_counter()
    
                # levanta erro se status != 200
                r.raise_for_status()
    
                data = r.json()
                
                print(
                    chain["model_name"],
                    "http:", round(t1 - t0, 2),
                )
    
                return {
                    "content": data.get("response"),
                    "thinking": data.get("thinking"),
                    "logprobs": data.get("logprobs")
                }

            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama HTTP error: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text}")
                raise

            except httpx.RequestError as e:
                logger.error(f"Request error while calling Ollama: {e}")
                raise

            except Exception as e:
                logger.exception("Unexpected error calling Ollama")
                raise

        # -----------------------------
        # LANGCHAIN (GROQ / HF / CHATOLLAMA)
        # -----------------------------
        response = await chain.ainvoke(
            {"text": text},
        )

        return response

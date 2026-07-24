"""
Annotation Engine - Motor de anotação
"""

from typing import List, Dict, Optional
from loguru import logger
import asyncio
import httpx
import time

from src.systems.llm_annotation_system.core.llm_provider import LLMProvider
from src.systems.llm_annotation_system.core.cache_manager import CacheManager
from src.systems.llm_annotation_system.core.response_processor import ResponseProcessor
from src.systems.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy

from src.config.datasets_collected import DATASETS, LABEL_MEANINGS
from src.config.prompts import BASE_ANNOTATION_PROMPT, FEW_SHOT_PROMPT
from src.utils.get_text_id_from_text import get_text_id_from_text

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
        candidates_by_text_id: Optional[Dict[str, List[int]]] = None,
    ):
        """
        Args:
            llm_provider: Provedor de LLMs
            cache_manager: Gerenciador de cache
            response_processor: Processador de respostas
            candidates_by_text_id: Mapa opcional text_id -> lista de classes
                candidatas (índices canônicos) da Fase 1. Quando fornecido, o
                prompt de cada texto apresenta apenas essas classes (Fase 2).
                Quando None (default), o comportamento é idêntico ao baseline
                (todas as classes no prompt).
        """
        self.llm_provider = llm_provider
        self.cache_manager = cache_manager
        self.response_processor = response_processor
        self.dataset_name = dataset_name
        self.candidates_by_text_id = candidates_by_text_id
        self._prompt_template = prompt_template
        logger.debug(f"AnnotationEngine inicializado para dataset: {dataset_name}")

        # Preparar template completo (baseline). Também popula self._categories_indexed,
        # reutilizado para renderizar subconjuntos por-texto na Fase 2.
        self.template = self._prepare_template(prompt_template, examples)
        logger.info("Template do prompt preparado")
        if self.candidates_by_text_id is not None:
            logger.info(
                f"Filtro de classes ATIVO: prompts por-texto com espaço reduzido "
                f"({len(self.candidates_by_text_id)} textos mapeados)"
            )
        
    async def _annotate_rep(
        self,
        chain: any,
        text: str,
        model: str,
        rep: int,
        use_cache: bool,
        template: Optional[str] = None,
        valid_categories: Optional[List[int]] = None,
    ) -> dict:
        try:

            # NOTA: a chave de cache NÃO inclui as candidatas da Fase 1. Como o
            # cache está desligado neste fluxo, não há colisão. Se o cache for
            # reativado junto com o filtro de classes, incluir aqui uma
            # assinatura das candidatas (ex.: {"rep": rep, "cand": sorted(...)})
            # para não reusar a resposta do baseline (todas as classes).
            cache_key = self.cache_manager.get_key(model, text, {"rep": rep})

            if use_cache:
                cached = self.cache_manager.get(cache_key)
                if cached:
                    response = cached
                    logger.debug(f"{model} rep {rep+1}: cache hit")
                else:
                    response = await self._ainvoke_chain(chain, text, template)
                    self.cache_manager.set(cache_key, response)
                    logger.debug(f"{model} rep {rep+1}: cache miss")
            else:
                t0 = time.perf_counter()
                response = await self._ainvoke_chain(chain, text, template)
                t1 = time.perf_counter()
                print(f"{model}: langchain: {round(t1 - t0, 2)}")

            result = self.response_processor.extract_label_and_confidence(
                response, valid_categories=valid_categories
            )
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

        # Resolve as candidatas da Fase 1 (ou None = baseline) e o template do
        # texto. Todas as repetições deste texto compartilham ambos (as
        # candidatas não mudam entre repetições).
        candidate_indices = self._resolve_candidates(text)
        template = (
            self.template
            if candidate_indices is None
            else self._template_for_indices(candidate_indices)
        )

        # Criar chain
        if isinstance(llm, dict) and llm.get("provider") == "ollama":
            chain = llm
        else:
            chain = self.llm_provider.create_chain(
                llm=llm,
                template=template,
            )

        # ===============================
        # 🔁 SEQUENCIAL
        # ===============================
        if rep_strategy == ExecutionStrategy.SEQUENTIAL:
            for rep in range(num_repetitions):
                result = await self._annotate_rep(
                    chain, text, model, rep, use_cache, template, candidate_indices
                )
                classifications.append(result)
            return classifications

        # ===============================
        # 🚀 PARALELO
        # ===============================
        elif rep_strategy == ExecutionStrategy.PARALLEL:
            tasks = [
                self._annotate_rep(
                    chain, text, model, rep, use_cache, template, candidate_indices
                )
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
        Prepara o template do prompt com TODAS as categorias do dataset (baseline).

        Também popula ``self._categories_indexed`` (dict canônico ``{idx: label}``),
        reutilizado por ``_template_for_indices`` para renderizar subconjuntos de
        classes por-texto na Fase 2.

        Args:
            prompt_template: Template base
            examples: Exemplos para few-shot

        Returns:
            Template formatado (com ``{text}`` ainda como placeholder)
        """
        self._categories_indexed = self._get_categories_indexed()
        categories_str = self._render_categories_str()

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

    def _get_categories_indexed(self) -> Dict[str, str]:
        """
        Retorna o dict canônico ``{idx: label}`` do dataset, com a mesma
        precedência usada historicamente: primeiro ``LABEL_MEANINGS``; senão,
        enumera a lista de categorias do processor.
        """
        if self.dataset_name in LABEL_MEANINGS:
            return LABEL_MEANINGS[self.dataset_name]

        if isinstance(self.response_processor.categories, list):
            return {
                str(i): cat for i, cat in enumerate(self.response_processor.categories)
            }

        return self.response_processor.categories

    def _render_categories_str(self, indices: Optional[List[int]] = None) -> str:
        """
        Renderiza o bloco ``{categories}`` como linhas ``- idx: label``.

        Args:
            indices: Se None, renderiza TODAS as classes (baseline). Se fornecido,
                renderiza apenas esse subconjunto de índices canônicos, ordenado
                por índice ascendente — ordem neutra que preserva os índices
                originais e NÃO revela o ranking do classificador da Fase 1.
        """
        items = list(self._categories_indexed.items())

        if indices is not None:
            wanted = {str(i) for i in indices}
            items = [(idx, label) for idx, label in items if idx in wanted]
            items.sort(key=lambda kv: int(kv[0]))  # ordem canônica ascendente (neutra)

        return "\n".join([f"- {idx}: {label}" for idx, label in items])

    def _template_for_indices(self, indices: List[int]) -> str:
        """
        Monta um template por-texto apresentando apenas as classes ``indices``.

        Mantém o mesmo estilo de prompt e as mesmas descrições de classe do
        baseline — a única diferença é o tamanho da lista de categorias.
        """
        if self._prompt_template == FEW_SHOT_PROMPT:
            raise ValueError(
                "Filtro de classes por-texto é incompatível com FEW_SHOT_PROMPT: "
                "o LLM deve permanecer zero-shot em relação aos dados anotados."
            )

        categories_str = self._render_categories_str(indices)
        description = DATASETS.get(self.dataset_name, {}).get("prompt", "Text")
        return self._prompt_template.format(
            description=description,
            description_lower=description.lower(),
            text="{text}",
            categories=categories_str,
        )

    def _resolve_candidates(self, text: str) -> Optional[List[int]]:
        """
        Retorna as classes candidatas da Fase 1 para um texto, ou None quando o
        filtro está desligado ou o texto não tem candidatas mapeadas (fallback
        para todas as classes).
        """
        if self.candidates_by_text_id is None:
            return None

        text_id = get_text_id_from_text(text)
        indices = self.candidates_by_text_id.get(text_id)

        if not indices:
            logger.warning(
                f"Sem candidatas para text_id={text_id[:8]}… — usando todas as classes"
            )
            return None

        return indices

    def _resolve_template(self, text: str) -> str:
        """
        Escolhe o template para um texto: o completo (baseline) ou um reduzido
        às candidatas da Fase 1, quando ``candidates_by_text_id`` está ativo.
        """
        indices = self._resolve_candidates(text)
        if indices is None:
            return self.template
        return self._template_for_indices(indices)
    
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
    
    async def _ainvoke_chain(self, chain: any, text: str, template: Optional[str] = None) -> dict:
        # Template por-texto (Fase 2) quando fornecido; senão o completo (baseline).
        template = template if template is not None else self.template

        # -----------------------------
        # OLLAMA VIA API (httpx direto)
        # -----------------------------
        if isinstance(chain, dict) and chain.get("provider") == "ollama":

            prompt = template.format(text=text)

            payload = {
                "model": chain["model_name"],
                "prompt": prompt,
                "options": {**chain.get("params", {})},
                "logprobs": chain.get("logprobs", True),
                "stream": False,
                "keep_alive": chain.get("keep_alive", None)
            }

            try:
                r = await self.llm_provider.client.post(
                    f"{chain['base_url']}/api/generate",
                    json=payload
                )

                r.raise_for_status()

                data = r.json()

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

            except Exception:
                logger.exception("Unexpected error calling Ollama")
                raise

        # -----------------------------
        # LANGCHAIN (GROQ / HF / CHATOLLAMA)
        # -----------------------------
        response = await chain.ainvoke({"text": text})

        return {
            "content": response.content,
            "thinking": response.response_metadata.get("thinking"),
            "logprobs": response.response_metadata.get("logprobs"),
        }

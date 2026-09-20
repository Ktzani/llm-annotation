"""
LLM Annotator - Classe principal
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from collections import Counter
from loguru import logger
import asyncio
import threading
import time

from src.systems.llm_annotation_system.core.llm_provider import LLMProvider
from src.systems.llm_annotation_system.core.cache_manager import CacheManager, LangChainCacheManager
from src.systems.llm_annotation_system.core.response_processor import ResponseProcessor
from src.systems.llm_annotation_system.core.model_variants import AlternativeParamsSelection, ModelVariantResolver
from src.systems.llm_annotation_system.annotation.annotation_engine import AnnotationEngine
from src.systems.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy

from src.config.prompts import BASE_ANNOTATION_PROMPT
from src.utils.get_text_id_from_text import get_text_id_from_text

_CHECKPOINT_WRITE_LOCK = threading.Lock()

class LLMAnnotator:
    """
    Classe principal para anotações automáticas
    
    Coordena componentes:
    - LLMProvider: gerencia LLMs
    - CacheManager: gerencia cache
    - ResponseProcessor: processa respostas
    - ModelVariantResolver: resolve variações de parâmetros (alternative_params)
    - AnnotationEngine: realiza anotações
    """
    
    def __init__(
        self,
        dataset_name: str,
        models: List[str],
        categories: List[str],
        cache_dir: str,
        results_dir: str,
        prompt_template = BASE_ANNOTATION_PROMPT,
        examples: Optional[List[Dict]] = None,
        use_cache: bool = True,
        use_alternative_params: AlternativeParamsSelection = False,
        keep_alive: int | str | None = None,
        candidates_by_text_id: Optional[Dict[str, List[int]]] = None,
    ):
        """
        Args:
            models: Lista de modelos
            categories: Lista de categorias
            api_keys: Chaves de API
            cache_dir: Diretório de cache
            results_dir: Diretório de resultados
            use_langchain_cache: Se True, usa cache do LangChain
            use_alternative_params: Seleção de alternative_params (ver ModelVariantResolver)
            candidates_by_text_id: Mapa opcional text_id -> classes candidatas da
                Fase 1. Quando None (default), o LLM vê todas as classes (baseline).
        """
        self.models = models
        self.categories = categories
        self.cache_dir = Path(cache_dir)
        
        self.results_dir = Path(results_dir)
        self.results_dir = self.results_dir.joinpath(dataset_name)
        
        self.use_alternative_params = use_alternative_params
        self.use_cache = use_cache
        
        # Criar diretórios
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Inicializar componentes
        self.llm_provider = LLMProvider(keep_alive=keep_alive)
        self.cache_manager = CacheManager(cache_dir, enabled=use_cache)
        self.langchain_cache = LangChainCacheManager(cache_dir, enabled=use_cache)
        self.response_processor = ResponseProcessor(categories)
        self.variant_resolver = ModelVariantResolver(use_alternative_params)
        self.annotation_engine = AnnotationEngine(
            llm_provider=self.llm_provider,
            cache_manager=self.cache_manager,
            response_processor=self.response_processor,
            dataset_name=dataset_name,
            prompt_template=prompt_template,
            examples=examples,
            candidates_by_text_id=candidates_by_text_id,
        )
        
        # Aplicar a seleção de alternative_params (ex.: "alt2" ou {"llama3.1-8b": "alt2"})
        if use_alternative_params:
            self.models = self.variant_resolver.expand(models)
            logger.info(f"Alternative params ({use_alternative_params}): executando {self.models}")

        # Inicializar LLMs
        self.llms = self._initialize_llms()

        logger.info(f"LLMAnnotator inicializado")
        logger.info(f"Modelos: {len(self.models)} | Categorias: {len(categories)}")

    def _initialize_llms(self) -> Dict[str, Any]:
        """Inicializa todas as LLMs"""
        llms = {}
        for model in self.models:
            llms[model] = self.llm_provider.initialize_llm(model)
        return llms
        
    def get_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        return self.cache_manager.stats()
    
    def get_langchain_cache_stats(self) -> Dict:
        """Retorna estatísticas do cache do LangChain"""
        return self.langchain_cache.stats()
    
    def get_models(self) -> List[str]:
        """Retorna lista de modelos utilizados"""
        return self.models
    
    async def annotate_single(
        self,
        text: str,
        model: str,
        num_repetitions: int = 1,
        rep_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ) -> List[str]:
        """
        Anota um texto único
        
        Args:
            text: Texto para anotar
            model: Modelo a usar
            num_repetitions: Número de repetições
            prompt_template: Template do prompt
            examples: Exemplos para few-shot
            use_cache: Se True, usa cache
            rep_strategy: Rodar as repeticoes dentro de cada modelo sequencialmente ou paralelo
            
        Returns:
            Lista de classificações
        """
        return await self.annotation_engine.annotate(
            text=text,
            model=model,
            llm=self.llms[model],
            num_repetitions=num_repetitions,
            use_cache=self.use_cache,
            rep_strategy=rep_strategy
        )

    async def _annotate_model(
        self,
        text: str,
        model: str,
        num_repetitions: int,
        rep_strategy: ExecutionStrategy
    ) -> dict:
        start = time.perf_counter()

        annotations = await self.annotate_single(
            text=text,
            model=model,
            num_repetitions=num_repetitions,
            rep_strategy=rep_strategy
        )
        
        labels = [a["label"] for a in annotations]
        
        end = time.perf_counter()

        elapsed = end - start

        result = {}

        for rep_idx, annotation in enumerate(annotations):
            result[f"{model}_rep{rep_idx+1}"] = annotation["label"]
            result[f"{model}_rep{rep_idx+1}_conf"] = annotation["confidence"]

        most_common = Counter(labels).most_common(1)[0]
        
        result[f"{model}_consensus"] = int(most_common[0])
        result[f"{model}_consensus_score"] = float(
            most_common[1] / len(labels)
        )
        
        result[f"{model}_annotation_time_sec"] = elapsed

        return result
    
    async def annotate_dataset(
        self,
        texts: List[str],
        num_repetitions: Optional[int] = 1,
        intermediate: int = 10,
        model_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        rep_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        max_concurrent_texts: int = 5
    ) -> pd.DataFrame:
        """
        Anota um dataset completo de textos.

        Coordena a anotação de múltiplos textos em paralelo respeitando o
        limite de concorrência, salva resultados em lote no checkpoint
        intermediário e retorna o DataFrame final com todas as anotações.

        Args:
            texts: Lista de textos para anotar
            num_repetitions: Número de repetições por modelo
            intermediate: Salva a cada N textos processados
            model_strategy: Sequencial ou paralelo entre modelos
            rep_strategy: Sequencial ou paralelo entre repetições
            max_concurrent_texts: Limite de textos processados simultaneamente

        Returns:
            DataFrame com todas as anotações (incluindo as já existentes
            no checkpoint)
        """
        self._log_run_info(
            texts, num_repetitions, model_strategy, rep_strategy,
            intermediate, max_concurrent_texts
        )

        file_path = self.results_dir / "intermediate.csv"
        processed_ids = self._load_checkpoint(file_path)

        semaphore = asyncio.Semaphore(max_concurrent_texts)
        metrics_lock = asyncio.Lock()
        buffer_lock = asyncio.Lock()
        buffer: list[dict] = []

        # Descarta uma chamada por modelo ANTES de iniciar o cronômetro.
        await self._warmup(model_strategy)

        completed = 0
        total_time = 0.0
        start_global = time.perf_counter()

        async def process_text(text):
            """Anota um texto e adiciona ao buffer; faz flush se atingiu o lote."""
            nonlocal completed, total_time

            async with semaphore:
                start = time.perf_counter()
                text_results = await self._annotate_text(
                    text, num_repetitions, model_strategy, rep_strategy
                )
                elapsed = time.perf_counter() - start

                # Atualiza métricas agregadas (avg_time e throughput no tqdm)
                async with metrics_lock:
                    completed += 1
                    total_time += elapsed

                # Buffer + flush em lote para evitar I/O por texto
                async with buffer_lock:
                    buffer.append(text_results)
                    if len(buffer) >= intermediate:
                        await self._flush_buffer(buffer, file_path)

        # Filtra textos já presentes no checkpoint (retomada idempotente)
        # Tasks explícitas: o as_completed não cancela as tasks pendentes quando a anotação é cancelada
        tasks = [
            asyncio.create_task(process_text(text))
            for text in texts
            if get_text_id_from_text(text) not in processed_ids
        ]
        remaining = len(tasks)

        pbar = tqdm(total=remaining, desc="Anotando", smoothing=0.05)
        try:
            for coro in asyncio.as_completed(tasks):
                await coro

                avg_time = total_time / completed
                total_elapsed = time.perf_counter() - start_global
                throughput = completed / total_elapsed

                pbar.update(1)
                pbar.set_postfix({
                    "avg_s": f"{avg_time:.2f}",
                    "it/s": f"{throughput:.2f}"
                })
        except asyncio.CancelledError:
            # Para as chamadas em andamento e salva os textos já anotados para a retomada
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._save_progress(buffer, file_path)
            logger.warning(f"Anotação cancelada: {completed}/{remaining} textos anotados salvos em {file_path}")
            raise
        finally:
            pbar.close()

        total_elapsed = time.perf_counter() - start_global
        logger.info("Finalizado ✅")
        logger.info(f"Tempo total: {total_elapsed:.2f}s")
        logger.info(f"Throughput médio: {remaining / total_elapsed:.2f} textos/s")

        # Flush do que sobrou (último lote menor que `intermediate`)
        await self._save_progress(buffer, file_path)

        return pd.read_csv(file_path)

    def _log_run_info(
        self,
        texts: List[str],
        num_repetitions: int,
        model_strategy: ExecutionStrategy,
        rep_strategy: ExecutionStrategy,
        intermediate: int,
        max_concurrent_texts: int
    ) -> None:
        """Loga os parâmetros da execução antes de começar a anotação."""
        total_annotations = len(texts) * len(self.models) * num_repetitions
        logger.info(f"Textos: {len(texts)} | Modelos: {len(self.models)} | Repetições: {num_repetitions}")
        logger.info(f"Total de anotações: {total_annotations}")
        logger.info(f"Strategy - Modelos: {model_strategy.name} | Repetições: {rep_strategy.name}")
        logger.info(f"Salvamento intermediário a cada {intermediate} textos")
        logger.info(f"Máximo de textos processados simultaneamente: {max_concurrent_texts}")

    def _load_checkpoint(self, file_path: Path) -> set:
        """
        Carrega o checkpoint intermediário se existir.

        Returns:
            text_ids já processados (vazio se não há checkpoint). A necessidade
            de header é decidida na hora da escrita, em `_append_chunk`.
        """
        if not file_path.exists():
            return set()

        df_existing = pd.read_csv(file_path)
        processed_ids = set(df_existing["text_id"].tolist())
        logger.info(f"Checkpoint encontrado: {len(processed_ids)} textos já processados")
        return processed_ids

    async def _warmup(self, model_strategy: ExecutionStrategy) -> None:
        """
        Faz uma anotação descartada para carregar os modelos na VRAM.

        A primeira requisição a um modelo ainda não residente inclui o tempo de
        carregar os pesos, que contaminaria o `annotation_time_sec` do primeiro
        texto e, por consequência, o `avg_s` e o throughput da execução inteira.
        O resultado é descartado e o tempo gasto fica fora do cronômetro.

        Falha aqui não interrompe a execução: se o warm-up não funcionar, a
        anotação segue e apenas o primeiro texto sai com o tempo inflado.
        """
        logger.info(f"Warm-up: carregando {len(self.models)} modelo(s) na VRAM...")
        start = time.perf_counter()

        try:
            await self._annotate_text(
                text="warmup",
                num_repetitions=1,
                model_strategy=model_strategy,
                rep_strategy=ExecutionStrategy.SEQUENTIAL,
            )
        except Exception as e:
            logger.warning(f"Warm-up falhou ({e}); seguindo com a anotação")
            return

        logger.info(f"Warm-up concluído em {time.perf_counter() - start:.2f}s")

    async def _annotate_text(
        self,
        text: str,
        num_repetitions: int,
        model_strategy: ExecutionStrategy,
        rep_strategy: ExecutionStrategy
    ) -> dict:
        """
        Anota um único texto com todos os modelos configurados.

        Aplica a `model_strategy` para decidir entre rodar os modelos em
        sequência ou em paralelo (via `asyncio.gather`).

        Returns:
            Dict com metadados do texto (text_id, text, text_len) e os
            resultados de cada modelo (labels, confiança, consenso, tempo).
        """
        text_results = {
            "text_id": get_text_id_from_text(text),
            "text": text,
            "text_len": len(text)
        }
        
        # ===============================
        # 🔁 MODELOS SEQUENCIAL
        # ===============================
        if model_strategy == ExecutionStrategy.SEQUENTIAL:
            for model in self.models:
                result = await self._annotate_model(
                    text=text,
                    model=model,
                    num_repetitions=num_repetitions,
                    rep_strategy=rep_strategy
                )
                text_results.update(result)
                
        # ===============================
        # 🚀 MODELOS PARALELO
        # ===============================
        else:
            model_tasks = [
                self._annotate_model(
                    text=text,
                    model=model,
                    num_repetitions=num_repetitions,
                    rep_strategy=rep_strategy
                )
                for model in self.models
            ]
            for result in await asyncio.gather(*model_tasks):
                text_results.update(result)

        return text_results

    @staticmethod
    def _append_chunk(df_chunk: pd.DataFrame, file_path: Path) -> None:
        """
        Anexa o chunk ao CSV, escrevendo o header apenas se o arquivo estiver vazio.

        A decisão do header é tomada AQUI, com o arquivo já aberto em append
        (`tell() == 0` ⇒ arquivo novo/vazio), e não a partir de um flag
        capturado no início da execução. O checkpoint é compartilhado por
        dataset (`<results>/<dataset>/intermediate.csv`): duas anotações
        concorrentes do mesmo dataset carregam o checkpoint antes de qualquer
        flush e ambas se achariam "a primeira", gravando um header no meio do
        arquivo — linha que sobrevive ao `drop_duplicates(text_id)` (o
        text_id dela é a string "text_id") e contamina consenso e métricas.
        """
        # Lock: no cancelamento, a escrita de uma task cancelada ainda pode estar em andamento
        with _CHECKPOINT_WRITE_LOCK, open(file_path, "a", newline="", encoding="utf-8") as f:
            df_chunk.to_csv(f, header=f.tell() == 0, index=False)

    async def _flush_buffer(self, buffer: list, file_path: Path) -> None:
        """
        Escreve o buffer no CSV em append e limpa o buffer.

        A escrita acontece numa thread separada (`asyncio.to_thread`) para
        não bloquear o event loop.
        """
        # Copia + clear sob o lock do chamador para liberar o buffer
        # enquanto o I/O acontece em background.
        df_chunk = pd.DataFrame(buffer.copy())
        buffer.clear()

        await asyncio.to_thread(self._append_chunk, df_chunk, file_path)

    async def _save_progress(self, buffer: list, file_path: Path) -> None:
        """Grava o que restou no buffer e salva o cache (fim da anotação ou cancelamento)"""
        if buffer:
            await self._flush_buffer(buffer, file_path)
        self.cache_manager.save()

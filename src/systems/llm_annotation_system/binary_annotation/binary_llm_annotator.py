"""
Binary LLM Annotator - Anotação binária: uma pergunta sim/não por (texto, classe, modelo)
"""
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from loguru import logger
import asyncio
import time

from src.systems.llm_annotation_system.annotation.llm_annotator import LLMAnnotator, RUN_ENV_COLUMN, RUN_ENV
from src.systems.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy
from src.systems.llm_annotation_system.core.model_variants import AlternativeParamsSelection
from src.systems.llm_annotation_system.binary_annotation.binary_annotation_engine import BinaryAnnotationEngine
from src.systems.llm_annotation_system.core.binary_response_processor import BinaryResponseProcessor

from src.config.prompts import BINARY_CLASS_PROMPT
from src.utils.get_text_id_from_text import get_text_id_from_text


class BinaryLLMAnnotator(LLMAnnotator):
    """
    Classe principal da anotação binária

    Reaproveita do LLMAnnotator a inicialização das LLMs (com alternative_params) e a escrita do
    checkpoint; troca o motor e o processador pelos binários. Cada texto recebe uma pergunta por
    classe e modelo, e cada resposta vira uma linha do intermediate.csv (formato longo).
    """

    RESULTS_SUBDIR = "binary"
    KEY_COLUMNS = ["text_id", "class_code", "model"]

    def __init__(
        self,
        dataset_name: str,
        models: List[str],
        categories: List[int],
        class_prompts: Dict,
        cache_dir: str,
        results_dir: str,
        prompt_template: str = BINARY_CLASS_PROMPT,
        use_alternative_params: AlternativeParamsSelection = False,
        keep_alive: int | str | None = None,
    ):
        """
        Args:
            class_prompts: {código ou nome da classe: ClassPrompt}
            prompt_template: Template das classes descritas por description/examples
        """
        # Sem cache de respostas: o intermediate.csv já guarda todas
        super().__init__(
            dataset_name=dataset_name,
            models=models,
            categories=categories,
            cache_dir=cache_dir,
            results_dir=results_dir,
            use_cache=False,
            use_alternative_params=use_alternative_params,
            keep_alive=keep_alive,
        )

        # Checkpoint separado da anotação multiclasse (formato diferente): <results>/<dataset>/binary/
        self.results_dir = self.results_dir / self.RESULTS_SUBDIR
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Componentes binários no lugar dos multiclasse
        self.response_processor = BinaryResponseProcessor(categories)
        self.annotation_engine = BinaryAnnotationEngine(
            llm_provider=self.llm_provider,
            cache_manager=self.cache_manager,
            response_processor=self.response_processor,
            dataset_name=dataset_name,
            class_prompts=class_prompts,
            prompt_template=prompt_template,
        )
        self.class_names = self.annotation_engine.class_names

        logger.info(f"BinaryLLMAnnotator inicializado | Classes: {len(self.class_names)}")

    async def annotate_single(self, text: str, class_code: int, model: str) -> dict:
        """Pergunta a um modelo se o texto pertence à classe"""
        return await self.annotation_engine.annotate(
            text=text,
            class_code=class_code,
            model=model,
            llm=self.llms[model],
        )

    async def _annotate_answer(self, text: str, class_code: int, model: str) -> Dict[str, Any]:
        """Uma linha do checkpoint: a resposta do modelo para (texto, classe)"""
        start = time.perf_counter()
        annotation = await self.annotate_single(text, class_code, model)
        elapsed = time.perf_counter() - start

        return {
            "text_id": get_text_id_from_text(text),
            "text": text,
            "text_len": len(text),
            RUN_ENV_COLUMN: RUN_ENV,
            "class_code": class_code,
            "class_name": self.class_names[class_code],
            "model": model,
            "answer": annotation["label"],
            "confidence": annotation["confidence"],
            "p_yes": self.response_processor.p_yes(annotation["label"], annotation["confidence"]),
            "annotation_time_sec": elapsed,
        }

    async def _annotate_model(
        self,
        text: str,
        model: str,
        class_codes: List[int],
        class_strategy: ExecutionStrategy,
        max_concurrent_classes: int
    ) -> List[Dict[str, Any]]:
        """Pergunta a um modelo cada classe do texto, em sequência ou até max_concurrent_classes de uma vez"""
        if class_strategy == ExecutionStrategy.SEQUENTIAL:
            return [await self._annotate_answer(text, class_code, model) for class_code in class_codes]

        semaphore = asyncio.Semaphore(max_concurrent_classes)

        async def ask(class_code: int) -> Dict[str, Any]:
            async with semaphore:
                return await self._annotate_answer(text, class_code, model)

        return list(await asyncio.gather(*(ask(class_code) for class_code in class_codes)))

    async def _annotate_text(
        self,
        text: str,
        pending: Dict[str, List[int]],
        model_strategy: ExecutionStrategy,
        class_strategy: ExecutionStrategy,
        max_concurrent_classes: int
    ) -> List[Dict[str, Any]]:
        """
        Anota as perguntas pendentes de um texto com todos os modelos.

        Args:
            pending: {modelo: classes ainda sem resposta no checkpoint}

        Returns:
            Uma linha por (classe, modelo) respondida
        """
        if model_strategy == ExecutionStrategy.SEQUENTIAL:
            rows = []
            for model, class_codes in pending.items():
                rows.extend(await self._annotate_model(text, model, class_codes, class_strategy, max_concurrent_classes))
            return rows

        results = await asyncio.gather(*(
            self._annotate_model(text, model, class_codes, class_strategy, max_concurrent_classes)
            for model, class_codes in pending.items()
        ))
        return [row for model_rows in results for row in model_rows]

    def _pending_by_text(self, texts: List[str], processed: set) -> Dict[str, Dict[str, List[int]]]:
        """{texto: {modelo: classes sem resposta}} — só os textos com alguma pergunta pendente"""
        pending = {}
        for text in texts:
            text_id = get_text_id_from_text(text)
            by_model = {
                model: [code for code in self.class_names if (text_id, code, model) not in processed]
                for model in self.models
            }
            by_model = {model: codes for model, codes in by_model.items() if codes}
            if by_model:
                pending[text] = by_model
        return pending

    async def annotate_dataset(
        self,
        texts: List[str],
        intermediate: int = 10,
        model_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        class_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL,
        max_concurrent_classes: int = 5
    ) -> pd.DataFrame:
        """
        Anota um dataset completo: cada texto com todas as classes e modelos.

        Mesmo fluxo do LLMAnnotator.annotate_dataset (buffer com flush em lote no intermediate.csv e
        retomada pelo checkpoint), com as classes no lugar das repetições. Os textos vão um por vez:
        o paralelismo é entre as classes de cada texto.

        Args:
            intermediate: Salva a cada N textos processados
            class_strategy: Classes de um texto em sequência ou em paralelo
            max_concurrent_classes: Classes perguntadas ao mesmo tempo, por modelo (com class_strategy=parallel)

        Returns:
            DataFrame longo com todas as respostas (incluindo as já existentes no checkpoint)
        """
        file_path = self.results_dir / "intermediate.csv"
        processed = self._load_checkpoint(file_path)
        pending = self._pending_by_text(texts, processed)

        self._log_binary_run_info(texts, pending, model_strategy, class_strategy, intermediate, max_concurrent_classes)

        # Um texto por vez: o paralelismo é entre as classes (max_concurrent_classes)
        semaphore = asyncio.Semaphore(1)
        metrics_lock = asyncio.Lock()
        buffer_lock = asyncio.Lock()
        buffer: list[list[dict]] = []

        # Descarta uma chamada por modelo ANTES de iniciar o cronômetro.
        if pending:
            await self._warmup(model_strategy)

        completed = 0
        total_time = 0.0
        start_global = time.perf_counter()

        async def process_text(text, text_pending):
            """Anota um texto e adiciona ao buffer; faz flush se atingiu o lote."""
            nonlocal completed, total_time

            async with semaphore:
                start = time.perf_counter()
                text_rows = await self._annotate_text(text, text_pending, model_strategy, class_strategy, max_concurrent_classes)
                elapsed = time.perf_counter() - start

                async with metrics_lock:
                    completed += 1
                    total_time += elapsed

                async with buffer_lock:
                    buffer.append(text_rows)
                    if len(buffer) >= intermediate:
                        await self._flush_buffer(buffer, file_path)

        # Tasks explícitas: o as_completed não cancela as tasks pendentes quando a anotação é cancelada
        tasks = [asyncio.create_task(process_text(text, text_pending)) for text, text_pending in pending.items()]
        remaining = len(tasks)

        pbar = tqdm(total=remaining, desc="Anotando (binário)", smoothing=0.05)
        try:
            for coro in asyncio.as_completed(tasks):
                await coro

                pbar.update(1)
                pbar.set_postfix({
                    "avg_s": f"{total_time / completed:.2f}",
                    "it/s": f"{completed / (time.perf_counter() - start_global):.2f}"
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
        if remaining:
            logger.info(f"Throughput médio: {remaining / total_elapsed:.2f} textos/s")

        # Flush do que sobrou (último lote menor que `intermediate`)
        await self._save_progress(buffer, file_path)

        return pd.read_csv(file_path)

    async def _flush_buffer(self, buffer: list, file_path: Path) -> None:
        """Escreve as linhas dos textos do buffer no CSV em append e limpa o buffer"""
        df_chunk = pd.DataFrame([row for text_rows in buffer for row in text_rows])
        buffer.clear()

        await asyncio.to_thread(self._append_chunk, df_chunk, file_path)

    def _log_binary_run_info(
        self,
        texts: List[str],
        pending: Dict[str, Dict[str, List[int]]],
        model_strategy: ExecutionStrategy,
        class_strategy: ExecutionStrategy,
        intermediate: int,
        max_concurrent_classes: int
    ) -> None:
        """Loga os parâmetros da execução antes de começar a anotação."""
        # Textos repetidos viram uma pergunta só (mesmo text_id)
        unique_texts = len({get_text_id_from_text(text) for text in texts})
        total = unique_texts * len(self.class_names) * len(self.models)
        to_do = sum(len(codes) for by_model in pending.values() for codes in by_model.values())
        logger.info(f"Textos: {len(texts)} ({unique_texts} únicos) | Classes: {len(self.class_names)} | Modelos: {len(self.models)}")
        logger.info(f"Total de perguntas: {total} | Já no checkpoint: {total - to_do} | A fazer: {to_do} ({len(pending)} textos)")
        logger.info(f"Strategy - Modelos: {model_strategy.name} | Classes: {class_strategy.name}")
        logger.info(f"Salvamento intermediário a cada {intermediate} textos")
        logger.info(f"Máximo de classes perguntadas simultaneamente (por modelo): {max_concurrent_classes}")

    def _load_checkpoint(self, file_path: Path) -> set:
        """Perguntas (text_id, class_code, model) já respondidas no checkpoint"""
        if not file_path.exists():
            return set()

        df_existing = pd.read_csv(file_path, usecols=self.KEY_COLUMNS)
        processed = set(zip(df_existing["text_id"], df_existing["class_code"].astype(int), df_existing["model"]))
        logger.info(f"Checkpoint encontrado: {len(processed)} respostas já obtidas | Ambiente atual: {RUN_ENV}")
        return processed

    async def _warmup(self, model_strategy: ExecutionStrategy) -> None:
        """Uma pergunta descartada por modelo para carregar os pesos na VRAM antes do cronômetro"""
        logger.info(f"Warm-up: carregando {len(self.models)} modelo(s) na VRAM...")
        start = time.perf_counter()
        first_class = next(iter(self.class_names))

        try:
            await self._annotate_text(
                text="warmup",
                pending={model: [first_class] for model in self.models},
                model_strategy=model_strategy,
                class_strategy=ExecutionStrategy.SEQUENTIAL,
                max_concurrent_classes=1,
            )
        except Exception as e:
            logger.warning(f"Warm-up falhou ({e}); seguindo com a anotação")
            return

        logger.info(f"Warm-up concluído em {time.perf_counter() - start:.2f}s")

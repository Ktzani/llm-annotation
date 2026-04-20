"""
Pipeline de anotação com LLMs
"""

import sys
import json
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

from src.api.schemas.annotation_experiment.experiment import ExperimentRequest
from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.api.schemas.annotation_experiment.annotation import AnnotationConfig as AnnotationSchemaConfig
from src.api.schemas.annotation_experiment.prompt_enum import PromptType
from src.api.services.prompt_factory import get_prompt_template
from src.utils.data_loader import load_hf_dataset
from src.utils.get_text_id_from_text import get_text_id_from_text
from src.llm_annotation_system.annotation.llm_annotator import LLMAnnotator
from src.llm_annotation_system.annotation.execution_estrategy import ExecutionStrategy
from src.llm_annotation_system.core.evaluate_model_metrics import evaluate_model_metrics


class AnnotationConfig:
    """
    Configurações do pipeline de anotação.

    O parâmetro `experiment_config` aceita duas formas de entrada, escolhidas
    conforme o contexto de execução:

    1) **Path para JSON** (`str`): usado em execuções via CLI/script, quando o
       experimento está versionado em disco. O arquivo é lido e validado como
       `ExperimentRequest`.

       Exemplo:
           config = AnnotationConfig(
               dataset_name="meu_dataset",
               experiment_config="experiments/experimento.json",
           )

    2) **Objeto `ExperimentRequest`**: usado quando o experimento já foi
       construído em memória — tipicamente pela API (FastAPI) após validar o
       payload do request. Evita round-trip desnecessário pelo disco.

       Exemplo:
           config = AnnotationConfig(
               dataset_name=request.dataset_name,
               experiment_config=request,  # instância de ExperimentRequest
           )

    Em ambos os casos o método interno `_apply_experiment` popula os mesmos
    atributos (models, prompt_type, dataset_config, etc.), então o
    `AnnotationPipeline` opera de forma idêntica independente da origem.
    """

    def __init__(
        self,
        dataset_name: str,
        experiment_config: Optional[Union[str, ExperimentRequest]] = None,
    ):
        self.dataset_name = dataset_name
        self.experiment_config_path = experiment_config if isinstance(experiment_config, str) else None

        if isinstance(experiment_config, ExperimentRequest):
            self._apply_experiment(experiment_config)
        elif isinstance(experiment_config, str):
            self._load_from_experiment(experiment_config)

    def _load_from_experiment(self, config_path: str) -> None:
        """Sobrescreve configurações a partir de um arquivo de experimento JSON"""
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        exp = ExperimentRequest(**config_dict)
        self._apply_experiment(exp)
        logger.info(f"Configurações carregadas de: {config_path}")

    def _apply_experiment(self, exp: ExperimentRequest) -> None:
        self.models = exp.models
        self.prompt_type = exp.prompt_type
        self.custom_prompt = exp.custom_prompt
        self.dataset_config = exp.dataset_config
        self.num_repetitions = exp.annotation.num_repetitions_per_llm
        self.use_alternative_params = exp.annotation.use_alternative_params
        self.model_strategy = exp.annotation.model_strategy
        self.rep_strategy = exp.annotation.rep_strategy
        self.max_concurrent_texts = exp.annotation.max_concurrent_texts
        self.keep_alive = exp.annotation.keep_alive
        self.cache_dir = exp.cache.dir
        self.intermediate = exp.results.intermediate
        self.results_dir = exp.results.dir
        
        logger.info(f"Configurações aplicadas do experimento: {exp.dataset_name} | Modelos: {len(self.models)} | Prompt: {self.prompt_type}")


class AnnotationPipeline:
    """Pipeline de anotação com LLMs"""

    def __init__(self, config: AnnotationConfig):
        self.config = config
        self.prompt_template = get_prompt_template(
            config.prompt_type,
            config.custom_prompt,
        )
        self._annotator: Optional[LLMAnnotator] = None
        logger.success("✓ Setup completo")

    def _get_annotator(self, categories: list) -> LLMAnnotator:
        if self._annotator is None:
            self._annotator = LLMAnnotator(
                dataset_name=self.config.dataset_name,
                models=self.config.models,
                categories=categories,
                cache_dir=self.config.cache_dir,
                results_dir=self.config.results_dir,
                prompt_template=self.prompt_template,
                use_cache=True,
                use_alternative_params=self.config.use_alternative_params,
                keep_alive=self.config.keep_alive,
            )
            logger.success(f"✓ Annotator inicializado com {len(self._annotator.models)} modelos")
        return self._annotator

    def load_texts(self, remove_annotated: bool = False) -> tuple:
        """Carrega textos, categorias e ground truth do HF"""
        texts, categories, ground_truth = load_hf_dataset(
            dataset_name=self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            dataset_global_config=self.config.dataset_config,
        )

        if remove_annotated:
            annotated_path = (
                Path(self.config.results_dir)
                / self.config.dataset_name
                / "intermediate.csv"
            )
            if annotated_path.exists():
                df_existing = pd.read_csv(annotated_path)
                annotated_texts = set(df_existing["text"])
                before = len(texts)
                texts = [t for t in texts if t not in annotated_texts]
                logger.info(f"Removidos {before - len(texts)} textos já anotados")
            else:
                logger.warning(f"Checkpoint não encontrado: {annotated_path}")

        logger.info(f"Textos: {len(texts)} | Categorias: {categories} | GT: {'Sim' if ground_truth else 'Não'}")
        return texts, categories, ground_truth

    async def run_single(
        self,
        texts: list,
        ground_truth: Optional[list],
        categories: list,
        index: int = 0,
    ) -> None:
        """Anota um único texto (modo debug)"""
        if index >= len(texts):
            logger.error(f"Índice {index} fora do range ({len(texts)} textos)")
            return

        annotator = self._get_annotator(categories)
        model = annotator.models[0]
        text = texts[index]

        logger.warning(f"Modo: anotação única | Índice: {index} | Modelo: {model}")
        logger.warning(f"Texto ({len(text)} chars): {text[:100]}...")

        annotations = await annotator.annotate_single(
            text=text,
            model=model,
            num_repetitions=3,
            rep_strategy=self.config.rep_strategy,
        )

        logger.success("✓ Anotação única completa")
        logger.info(f"Resultado: {annotations}")
        if ground_truth:
            logger.info(f"Ground truth: {ground_truth[index]}")

    async def run_dataset(
        self,
        texts: list,
        ground_truth: Optional[list],
        categories: list,
    ) -> Path:
        """Anota dataset completo, salva CSV e calcula métricas"""
        annotator = self._get_annotator(categories)

        df_annotations = await annotator.annotate_dataset(
            texts=texts,
            num_repetitions=self.config.num_repetitions,
            intermediate=self.config.intermediate,
            model_strategy=self.config.model_strategy,
            rep_strategy=self.config.rep_strategy,
            max_concurrent_texts=self.config.max_concurrent_texts,
        )

        df_annotations = df_annotations.drop_duplicates(subset=["text_id"])

        if len(texts) == len(ground_truth):
            df_gt = pd.DataFrame({"text": texts, "ground_truth": ground_truth})
            df_gt["text_id"] = df_gt["text"].apply(get_text_id_from_text)
            df_annotations = df_annotations.merge(
                df_gt[["text_id", "ground_truth"]], on="text_id", how="left"
            )

        date_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path(self.config.results_dir)
            / self.config.dataset_name
            / date_path
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        df_annotations.to_csv(output_dir / "annotations.csv", index=False)
        logger.success(f"✓ Anotações salvas em: {output_dir}")

        evaluate_model_metrics(
            df_annotations,
            models=annotator.models,
            ground_truth_col="ground_truth",
            output_dir=output_dir,
        )
        logger.success(f"✓ Métricas salvas em: {output_dir}")

        return output_dir

    async def run(self, debug_single: bool = False, debug_index: int = 0) -> Optional[Path]:
        """Executa pipeline completo"""
        logger.info("=" * 60)
        logger.info("Iniciando pipeline de anotação")
        logger.info("=" * 60)

        texts, categories, ground_truth = self.load_texts(
            remove_annotated=self.config.dataset_config.remove_texts or False
        )

        if debug_single:
            await self.run_single(texts, ground_truth, categories, index=debug_index)
            return None

        output_dir = await self.run_dataset(texts, ground_truth, categories)

        logger.info("=" * 60)
        logger.success("Pipeline de anotação finalizado!")
        logger.info(f"Arquivos em: {output_dir}")
        logger.info("=" * 60)

        return output_dir

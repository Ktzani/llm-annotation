"""
Pipeline de anotação binária com LLMs (uma pergunta sim/não por classe)
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

from loguru import logger

from src.api.schemas.binary_annotation_experiment.experiment import BinaryAnnotationRequest
from src.config.prompts import BINARY_CLASS_PROMPT
from src.utils.data_loader import load_hf_dataset_with_columns
from src.utils.get_text_id_from_text import get_text_id_from_text
from src.systems.llm_annotation_system.pipeline import AnnotationPipeline
from src.systems.llm_annotation_system.binary_annotation.binary_llm_annotator import BinaryLLMAnnotator
from src.systems.llm_annotation_system.binary_annotation.binary_labels_builder import BinaryLabelsBuilder
from src.systems.llm_annotation_system.core.evaluate_class_metrics import evaluate_class_metrics


class BinaryAnnotationConfig:
    """
    Configurações do pipeline de anotação binária.

    Como o AnnotationConfig, aceita o path do JSON do experimento (CLI/script) ou o
    `BinaryAnnotationRequest` já validado (API).
    """

    def __init__(
        self,
        experiment_config: Optional[Union[str, BinaryAnnotationRequest]] = None,
    ):
        self.experiment_config_path = experiment_config if isinstance(experiment_config, str) else None

        if isinstance(experiment_config, BinaryAnnotationRequest):
            self._apply_experiment(experiment_config)
        elif isinstance(experiment_config, str):
            self._load_from_experiment(experiment_config)

    def _load_from_experiment(self, config_path: str) -> None:
        """Sobrescreve configurações a partir de um arquivo de experimento JSON"""
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        exp = BinaryAnnotationRequest(**config_dict)
        self._apply_experiment(exp)
        logger.info(f"Configurações carregadas de: {config_path}")

    def _apply_experiment(self, exp: BinaryAnnotationRequest) -> None:
        self.request = exp
        self.dataset_name = exp.dataset_name
        self.models = exp.models
        self.class_prompts = exp.class_prompts
        self.custom_prompt = exp.custom_prompt
        self.dataset_config = exp.dataset_config
        self.use_alternative_params = exp.annotation.use_alternative_params
        self.model_strategy = exp.annotation.model_strategy
        self.class_strategy = exp.annotation.class_strategy
        self.max_concurrent_classes = exp.annotation.max_concurrent_classes
        self.keep_alive = exp.annotation.keep_alive
        self.keep_columns = exp.keep_columns
        self.cache_dir = exp.cache.dir
        self.intermediate = exp.results.intermediate
        self.results_dir = exp.results.dir
        logger.info(f"Configurações aplicadas do experimento binário: {exp.dataset_name} | Modelos: {len(self.models)} | Classes com prompt: {len(self.class_prompts)}")


class BinaryAnnotationPipeline(AnnotationPipeline):
    """Pipeline de anotação binária com LLMs (reaproveita o carregamento de textos do AnnotationPipeline)"""

    def __init__(self, config: BinaryAnnotationConfig):
        self.config = config
        self.prompt_template = config.custom_prompt or BINARY_CLASS_PROMPT
        self.labels_builder = BinaryLabelsBuilder()
        self._annotator: Optional[BinaryLLMAnnotator] = None
        self.df_rows: Optional[pd.DataFrame] = None
        logger.success("✓ Setup completo")

    def load_texts(self, remove_annotated: bool = False, run_type: str = "dataset") -> tuple:
        """Carrega textos, categorias e df_ground_truth como o AnnotationPipeline e guarda as keep_columns de cada linha"""
        texts, categories, ground_truth, df_extra = load_hf_dataset_with_columns(
            dataset_name=self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            dataset_global_config=self.config.dataset_config,
            extra_columns=self.config.keep_columns,
        )
        text_ids = [get_text_id_from_text(t) for t in texts]

        df_ground_truth = None
        if ground_truth is not None and len(ground_truth) == len(texts):
            df_ground_truth = pd.DataFrame({
                "text_id": text_ids,
                "ground_truth": ground_truth,
            }).drop_duplicates(subset=["text_id"])

        # Uma linha por linha do dataset (ex.: por message_id): textos repetidos são anotados uma vez só
        df_rows = df_extra if df_extra is not None else pd.DataFrame(index=range(len(texts)))
        self.df_rows = df_rows.assign(text_id=text_ids)

        if run_type == "single_text":
            logger.warning(f"Modo: anotação única | Textos carregados: {len(texts)} | Categorias: {categories} | GT: {'Sim' if df_ground_truth is not None else 'Não'}")
            return texts, categories, df_ground_truth

        # As planilhas continuam com todas as linhas (df_rows): só a lista a anotar perde os textos já anotados
        if remove_annotated:
            annotated_path = (
                Path(self.config.results_dir)
                / self.config.dataset_name
                / BinaryLLMAnnotator.RESULTS_SUBDIR
                / "intermediate.csv"
            )
            if annotated_path.exists():
                annotated_ids = set(pd.read_csv(annotated_path, usecols=["text_id"])["text_id"])
                before = len(texts)
                texts = [t for t, text_id in zip(texts, text_ids) if text_id not in annotated_ids]
                logger.info(f"Removidos {before - len(texts)} textos já anotados (as planilhas mantêm as {len(self.df_rows)} linhas do dataset)")
            else:
                logger.warning(f"Checkpoint não encontrado: {annotated_path}")

        logger.info(f"Textos a anotar: {len(texts)} ({len({get_text_id_from_text(t) for t in texts})} únicos) | Categorias: {categories} | GT: {'Sim' if df_ground_truth is not None else 'Não'}")
        return texts, categories, df_ground_truth

    def _get_annotator(self, categories: list) -> BinaryLLMAnnotator:
        if self._annotator is None:
            self._annotator = BinaryLLMAnnotator(
                dataset_name=self.config.dataset_name,
                models=self.config.models,
                categories=categories,
                class_prompts=self.config.class_prompts,
                cache_dir=self.config.cache_dir,
                results_dir=self.config.results_dir,
                prompt_template=self.prompt_template,
                use_alternative_params=self.config.use_alternative_params,
                keep_alive=self.config.keep_alive,
            )
            logger.success(f"✓ Annotator inicializado com {len(self._annotator.models)} modelos")
        return self._annotator

    async def run_single_text(
        self,
        texts: list,
        ground_truth: pd.DataFrame,
        categories: list,
        index: int = 0,
    ) -> None:
        """Anota um único texto em todas as classes com o primeiro modelo (modo debug dos prompts)"""
        if index >= len(texts):
            logger.error(f"Índice {index} fora do range ({len(texts)} textos)")
            return

        annotator = self._get_annotator(categories)
        model = annotator.models[0]
        text = texts[index]

        logger.warning(f"Modo: anotação única | Índice: {index} | Modelo: {model}")
        logger.warning(f"Texto ({len(text)} chars): {text[:100]}...")

        for class_code, class_name in annotator.class_names.items():
            annotation = await annotator.annotate_single(text=text, class_code=class_code, model=model)
            logger.info(f"  {class_code}: {class_name} → {annotation}")

        logger.success("✓ Anotação única completa")
        if ground_truth is not None:
            logger.info(f"Ground truth: {ground_truth.iloc[index]['ground_truth']}")

    async def run_dataset(
        self,
        texts: list,
        categories: list,
        df_ground_truth: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Anota o dataset e salva as respostas de cada modelo; o consenso roda depois (BinaryConsensusPipeline)"""
        annotator = self._get_annotator(categories)

        df_answers = await annotator.annotate_dataset(
            texts=texts,
            intermediate=self.config.intermediate,
            model_strategy=self.config.model_strategy,
            class_strategy=self.config.class_strategy,
            max_concurrent_classes=self.config.max_concurrent_classes,
        )
        df_answers = df_answers.drop_duplicates(subset=BinaryLLMAnnotator.KEY_COLUMNS)

        df_items = self.labels_builder.annotations_by_class(df_answers, annotator.models)

        if df_ground_truth is not None:
            # Ground truth do item (texto, classe): 1 se a classe verdadeira do texto é a do item
            truth = df_items[BinaryLabelsBuilder.TEXT_KEY].map(df_ground_truth.set_index("text_id")["ground_truth"])
            df_items["ground_truth"] = (truth == df_items["class_code"]).astype("Int64").where(truth.notna())
        else:
            logger.warning("df_ground_truth=None — merge de ground_truth pulado")

        date_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = (
            Path(self.config.results_dir)
            / self.config.dataset_name
            / BinaryLLMAnnotator.RESULTS_SUBDIR
            / date_path
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        df_items.to_csv(output_dir / "annotations.csv", index=False)
        for model in annotator.models:
            df_model = self.labels_builder.labels_by_text(df_items, column=f"{model}_consensus")
            self.labels_builder.expand_to_rows(df_model, self.df_rows).to_csv(output_dir / f"binary_labels_{model}.csv", index=False)
        self._save_prompts(annotator, output_dir)
        logger.success(f"✓ Anotações salvas em: {output_dir}")

        if "ground_truth" not in df_items.columns:
            logger.warning("Coluna ground_truth ausente — pulando cálculo de métricas")
            return output_dir

        evaluate_class_metrics(df_items, models=annotator.models, output_dir=output_dir)
        logger.success(f"✓ Métricas salvas em: {output_dir}")

        return output_dir

    @staticmethod
    def _save_prompts(annotator: BinaryLLMAnnotator, output_dir: Path) -> None:
        """Salva o prompt de cada classe como o modelo recebe ({text} no lugar da mensagem), para rastrear a versão usada"""
        prompts = {
            str(code): {"class_name": annotator.class_names[code], "template": template.format(text="{text}")}
            for code, template in annotator.annotation_engine.class_templates.items()
        }
        with open(output_dir / "prompts.json", "w", encoding="utf-8") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)

    async def run(self, run_type: str = "dataset", single_index: int = 0) -> Optional[tuple]:
        """Executa pipeline completo"""
        logger.info("=" * 60)
        logger.info("Iniciando pipeline de anotação binária")
        logger.info("=" * 60)

        texts, categories, df_ground_truth = self.load_texts(
            remove_annotated=self.config.dataset_config.remove_texts or False,
            run_type=run_type
        )

        if run_type == "single_text":
            await self.run_single_text(texts, df_ground_truth, categories, index=single_index)
            return None

        elif run_type == "dataset":
            output_dir = await self.run_dataset(texts, categories, df_ground_truth=df_ground_truth)

        else:
            logger.error(f"Run type desconhecido: {run_type}")
            return None

        logger.info("=" * 60)
        logger.success("Pipeline de anotação binária finalizado!")
        logger.info(f"Arquivos em: {output_dir}")
        logger.info("=" * 60)

        return output_dir, texts, categories, df_ground_truth

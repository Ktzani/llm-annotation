"""
Fine-tuning controlado de RoBERTa (GT vs Consenso LLM)

Este script realiza fine-tuning de modelos RoBERTa comparando resultados
entre ground truth e consenso de anotações LLM.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Union
import json

import pandas as pd
from datasets import Dataset
from loguru import logger
from transformers import TrainingArguments


# Configuração do logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# Imports do projeto

from src.utils.data_loader import load_hf_dataset_as_dataframe, add_label_description
from src.utils.get_text_id_from_text import get_text_id_from_text
from src.fine_tune_system.fine_tune.supervised_fine_tuner import SupervisedFineTuner
from src.fine_tune_system.fine_tune.fine_tune_factory import FineTunerFactory

from src.fine_tune_system.core.hf_tokenizer import HFTokenizer
from src.fine_tune_system.core.model_factory import ModelFactory

from src.fine_tune_system.training.trainer_builder import TrainerBuilder
from src.fine_tune_system.training.metrics import MetricsComputer
from src.fine_tune_system.training.label_schema import LabelSchema
from src.fine_tune_system.training.splits_aligner import CVSplitAligner
from src.fine_tune_system.training.cross_validator import CrossValidator


from src.utils.get_latest_results_date import get_latest_results_date
from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.api.schemas.fine_tuning.fine_tuning import FineTuningRequest
from src.is_system.filtering.annotation_filter import AnnotationFilter, save_filter_result



class FineTuningConfig:
    """
    Configurações do experimento de fine-tuning.

    O parâmetro `experiment_config` aceita duas formas de entrada, escolhidas
    conforme o contexto de execução (mesmo padrão de `AnnotationConfig`):

    1) **Path para JSON** (`str`): usado em execuções via CLI/script. O arquivo
       é lido e validado como `FineTuningRequest`.

       Exemplo:
           config = FineTuningConfig(
               experiment_config="src/api/experiments/fine_tuning/local_job.json",
           )

    2) **Objeto `FineTuningRequest`**: usado quando o request já foi construído
       em memória — tipicamente pela API (FastAPI) após validar o payload.

       Exemplo:
           config = FineTuningConfig(experiment_config=request)

    Em ambos os casos `_apply_experiment` popula os mesmos atributos, então o
    `FineTuningPipeline` opera de forma idêntica independente da origem.
    """

    def __init__(
        self,
        experiment_config: Optional[Union[str, FineTuningRequest]] = None,
    ):
        self.experiment_config_path = experiment_config if isinstance(experiment_config, str) else None

        if isinstance(experiment_config, FineTuningRequest):
            self._apply_experiment(experiment_config)
        elif isinstance(experiment_config, str):
            self._load_from_experiment(experiment_config)

    def _load_from_experiment(self, config_path: str) -> None:
        """Carrega as configurações a partir de um arquivo de experimento JSON."""
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        req = FineTuningRequest(**config_dict)
        self._apply_experiment(req)
        logger.info(f"Configurações carregadas de: {config_path}")

    def _apply_experiment(self, req: FineTuningRequest) -> None:
        # Dataset
        self.dataset_name = req.dataset.dataset_name
        self.cache_dir = req.dataset.cache_dir
        self.results_dir = Path(req.dataset.results_dir)
        self.specific_date = req.dataset.specific_date
        # Modelo / execução
        self.model_name = req.model_name
        self.run_type = req.run_type
        self.max_parallel_folds = req.max_parallel_folds
        # Hiperparâmetros
        self.learning_rate = req.hyperparams.learning_rate
        self.num_epochs = req.hyperparams.num_epochs
        self.train_batch_size = req.hyperparams.train_batch_size
        self.eval_batch_size = req.hyperparams.eval_batch_size
        self.weight_decay = req.hyperparams.weight_decay
        self.max_length = req.hyperparams.max_length
        self.seed = req.hyperparams.seed
        # Seleção de instâncias (biO-IS)
        self.use_instance_selection = req.instance_selection.enabled
        self.is_method = req.instance_selection.method
        self.is_params = req.instance_selection.params
        logger.info(
            f"Configurações aplicadas: {self.dataset_name} | model={self.model_name} | run_type={self.run_type} | IS={self.use_instance_selection} ({self.is_method})"
        )



class FineTuningPipeline:
    """Pipeline principal para fine-tuning"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.results_dataset_path = self._get_results_path()
        self.fine_tune_output_dir = self.results_dataset_path / "finetuning"
        
        self.fine_tune_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.success("✓ Setup completo")
    
    def _get_results_path(self) -> Path:
        """Obtém o caminho dos resultados"""
        specific_date = self.config.specific_date
        
        if specific_date == "latest":
            specific_date = get_latest_results_date(
                self.config.results_dir,
                self.config.dataset_name
            )
        
        return self.config.results_dir / self.config.dataset_name / specific_date
    
    def remove_problematic_instances(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """Remove instâncias problemáticas identificadas no consenso"""
        problematic_cases_path = (
            self.results_dataset_path / "consensus" / "problematic_cases.csv"
        )
        
        if problematic_cases_path.exists():
            df_problematic = pd.read_csv(problematic_cases_path)
            df_annotations = df_annotations[
                    ~df_annotations["text_id"].isin(df_problematic["text_id"])
            ].reset_index(drop=True)
            
            logger.info(f"Removidas {len(df_problematic)} instâncias problemáticas")
        
        return df_annotations
    
    def remove_invalid_instances(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """Remove instâncias com rótulos inválidos (-1)"""
        df_invalid = df_annotations.loc[
            df_annotations["label"] == -1 
        ]

        df_annotations = df_annotations[
            ~df_annotations["text_id"].isin(df_invalid["text_id"])
        ].reset_index(drop=True)
        
        logger.info(f"Removidas {len(df_invalid)} instâncias inválidas")

        return df_annotations

    def apply_instance_selection(self, df_annotations: pd.DataFrame) -> pd.DataFrame:
        """
        Mantém apenas as instâncias selecionadas pela filtragem biO-IS.

        Prefere o resultado pré-computado em
        `instance_selection/dataset_filtrado.csv` (gerado por
        `src/run_instance_selection.py`). Se ausente, executa a seleção e SALVA
        os artefatos com TODAS as colunas originais (para análises futuras), no
        mesmo formato do pipeline dedicado.

        Para não re-filtrar um conjunto diferente, a seleção roda sobre as
        colunas completas do CSV anotado RESTRITAS exatamente às mesmas linhas
        de `df_annotations` (alinhadas pela chave canônica). Filtra uma única
        vez e deriva o conjunto de treino da mesma seleção.

        O join (caminho pré-computado) usa a chave canônica `md5(text.strip())`
        — a mesma do alinhamento com os splits do HuggingFace —, pois o
        `text_id` salvo no CSV difere do recalculado pelo fine-tuning.
        """
        if not self.config.use_instance_selection:
            logger.info("Seleção de instâncias desativada (use_instance_selection=False).")
            return df_annotations

        filtered_path = (
            self.results_dataset_path / "instance_selection" / "dataset_filtrado.csv"
        )
        before = len(df_annotations)

        if filtered_path.exists():
            df_filtered = pd.read_csv(filtered_path)
            selected_ids = set(df_filtered["text"].apply(get_text_id_from_text))

            df_annotations = df_annotations[
                df_annotations["text_id"].isin(selected_ids)
            ].reset_index(drop=True)

            logger.info(
                f"Seleção de instâncias (biO-IS, pré-computado): "
                f"{before} → {len(df_annotations)} (removidas {before - len(df_annotations)})"
            )
            return df_annotations

        # Sem filtragem salva: executa e salva os artefatos com TODAS as colunas
        # originais (para análises futuras), restringindo o CSV anotado completo
        # exatamente às mesmas linhas já carregadas em df_annotations.
        logger.warning(
            f"'{filtered_path.name}' não encontrado — executando e salvando a filtragem..."
        )

        full_df = pd.read_csv(
            self.results_dataset_path / "summary" / "dataset_anotado_completo.csv"
        )
        canon_key = full_df["text"].apply(get_text_id_from_text)
        full_aligned = full_df[
            canon_key.isin(set(df_annotations["text_id"]))
        ].reset_index(drop=True)

        is_overrides = self.config.is_params or {}
        annotation_filter = AnnotationFilter(
            method=self.config.is_method,
            label_column="resolved_annotation",
            random_state=self.config.seed,
            **is_overrides,
        )
        result = annotation_filter.filter(full_aligned)
        save_filter_result(result, self.results_dataset_path / "instance_selection")

        # Conjunto de treino: deriva da MESMA seleção (mantém as 4 colunas).
        selected_ids = set(result.filtered_df["text"].apply(get_text_id_from_text))
        df_annotations = df_annotations[
            df_annotations["text_id"].isin(selected_ids)
        ].reset_index(drop=True)

        logger.info(
            f"Seleção de instâncias ({self.config.is_method}, computado e salvo): "
            f"{before} → {len(df_annotations)} (removidas {before - len(df_annotations)})"
        )
        return df_annotations

    def load_annotated_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados anotados"""
        logger.info("Carregando dados anotados...")
        
        df = pd.read_csv(
            self.results_dataset_path / "summary" / "dataset_anotado_completo.csv"
        )
        logger.info(f"Anotado: {len(df)} exemplos")

        df["text"] = df["text"].str.strip()
        df["text_id"] = df["text"].apply(get_text_id_from_text)

        # Dataset anotado (consenso)
        df_annotations = (
            df[["text_id", "text", "resolved_annotation"]]
            .rename(columns={"resolved_annotation": "label"})
        )
        df_annotations = add_label_description(
            df_annotations,
            dataset_name=self.config.dataset_name
        )
        
        df_annotations = self.remove_invalid_instances(df_annotations)
        df_annotations = self.remove_problematic_instances(df_annotations)
        df_annotations = self.apply_instance_selection(df_annotations)

        return df_annotations
    
    def load_hf_splits_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados de validação e teste"""
        logger.info("Carregando dados de avaliação (folds)...")
    
        splits = {}
        fold = 0

        while True:
            try:
                logger.info(f"Carregando fold {fold}...")
                train_config = DatasetConfig(
                    hf_file=f"train_fold_{fold}.parquet",
                )

                test_config = DatasetConfig(
                    hf_file=f"test_fold_{fold}.parquet",
                )

                df_train, _ = load_hf_dataset_as_dataframe(
                    dataset_name=self.config.dataset_name,
                    cache_dir=self.config.cache_dir,
                    dataset_global_config=train_config,
                )

                df_val, _ = load_hf_dataset_as_dataframe(
                    dataset_name=self.config.dataset_name,
                    cache_dir=self.config.cache_dir,
                    dataset_global_config=test_config,
                )

                splits[fold] = {
                    "train": df_train,
                    "val": df_val
                }

                logger.info(
                    f"Fold {fold}: train={len(df_train)}, val={len(df_val)} \n"
                )

                fold += 1

            except Exception as e:
                logger.info(f"Parando em fold {fold} (não encontrado).")
                break
        
        logger.info(f"Total de folds carregados: {len(splits)}")
        
        return splits
    
    def align_datasets_splits(
        self,
        splits: dict,
        annotated_df: pd.DataFrame
    ) -> Dict[int, Dict[str, Dataset]]:
        """Alinha splits do HF com dados anotados"""
        aligner = CVSplitAligner(splits, id_column="text_id")
        aligned_splits = aligner.align_datasets_splits(annotated_df)
        
        return aligned_splits
    
    def create_training_args(self) -> TrainingArguments:
        """Cria argumentos de treinamento"""
        return TrainingArguments(
            output_dir=self.fine_tune_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_strategy="epoch",
            save_total_limit=2,
            seed=self.config.seed,
        )
    
    def run_fine_tuning(
        self,
        train_ds: Dataset,
        eval_ds: Dataset,
        label_schema: LabelSchema,
        experiment_name: str
    ) -> dict:
        """Executa fine-tuning"""
        logger.info(f"Iniciando fine-tuning: {experiment_name}")
        
        training_args = self.create_training_args()
        
        fine_tuner = SupervisedFineTuner(
            model_name=self.config.model_name,
            training_args=training_args,
            label_schema=label_schema,
            tokenizer=HFTokenizer(
                model_name=self.config.model_name,
                max_length=self.config.max_length
            ),
            model_factory=ModelFactory,
            trainer_builder=TrainerBuilder,
            metrics_computer=MetricsComputer(),
        )

        fine_tuner.fit(
            train_ds=train_ds,
            eval_ds=eval_ds,
        )

        result = fine_tuner.best_val_metrics()
        
        logger.success(f"Fine-tuning concluído: {experiment_name}")
        logger.info(f"Accuracy: {result['eval_accuracy']:.4f}, F1 Macro: {result['eval_f1_macro']:.4f}")
        
        return result
    
    def run_cross_validation(
        self,
        cv_splits: pd.DataFrame,
        label_schema: LabelSchema,
        experiment_name: str,
        max_parallel_folds: int = 4,
    ):
        """Executa cross-validation"""
        logger.info(f"\n🚀 Cross-validation: {experiment_name}")
        
        training_args = self.create_training_args()

        # factory
        factory = FineTunerFactory(
            model_name=self.config.model_name,
            training_args=training_args,
            label_schema=label_schema,
            tokenizer=HFTokenizer(
                model_name=self.config.model_name,
                max_length=self.config.max_length
            ),
            model_factory=ModelFactory,
            trainer_builder=TrainerBuilder,
            metrics_computer=MetricsComputer(),
        )

        # cross-validator
        cv = CrossValidator(fine_tuner_factory=factory, max_parallel_folds=max_parallel_folds)

        results = cv.run(
            cv_splits=cv_splits,
            fine_tune_type="supervised",
        )

        logger.success(f"CV finalizado: {experiment_name}")
        logger.info(results)

        return results
    
    def run(self, run_type: str = "single", max_parallel_folds: int = 4) -> dict:
        """Executa pipeline completo"""
        logger.info("=" * 60)
        logger.info("Iniciando pipeline de fine-tuning")
        logger.info("=" * 60)
        
        # Carregar dados
        df_annotations = self.load_annotated_data()
        
        # Criar label schema
        label_schema = LabelSchema.from_dataframe(df_annotations)
        logger.info(f"Labels: {label_schema.id2label}")
        
        cv_splits_hf = self.load_hf_splits_data()
        cv_aligned_annotaded_splits = self.align_datasets_splits(cv_splits_hf, df_annotations)
        
        # Executar fine-tuning com consenso
        logger.info("\n" + "=" * 60)
        logger.info("Fine-tuning com CONSENSO LLM")
        logger.info("=" * 60)
        
        if run_type == "single":
            logger.warning("⚠️ Modo 'single' é apenas para teste rápido. Use 'cross-validation' para avaliação robusta.")
            results = self.run_fine_tuning(
                train_ds=cv_aligned_annotaded_splits[0]["train"],
                eval_ds=cv_aligned_annotaded_splits[0]["val"],
                label_schema=label_schema,
                experiment_name="consensus_llm_single"
            )
            
        elif run_type == "cross-validation":
            results = self.run_cross_validation(
                cv_splits=cv_aligned_annotaded_splits,
                label_schema=label_schema,
                experiment_name="consensus_llm_cv",
                max_parallel_folds=max_parallel_folds
            )
        
        else:
            raise ValueError(f"Tipo de execução desconhecido: {run_type}")
        
        # Salvar resultados
        output_path = self.fine_tune_output_dir / f"{self.config.model_name}_fine_tuning_results.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        logger.success(f"\nResultados salvos em: {output_path}")
        
        return results
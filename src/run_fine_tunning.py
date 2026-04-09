"""
Fine-tuning controlado de RoBERTa (GT vs Consenso LLM)

Este script realiza fine-tuning de modelos RoBERTa comparando resultados
entre ground truth e consenso de anotações LLM.
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset
from loguru import logger
from sklearn.model_selection import train_test_split
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
from src.fine_tune_system.fine_tune.supervised_fine_tuner import SupervisedFineTuner
from src.fine_tune_system.core.hf_tokenizer import HFTokenizer
from src.fine_tune_system.core.model_factory import ModelFactory
from src.fine_tune_system.training.trainer_builder import TrainerBuilder
from src.fine_tune_system.training.metrics import MetricsComputer
from src.fine_tune_system.training.label_schema import LabelSchema
from src.utils.dataset_aligner import DatasetAligner
from src.utils.get_latest_results_date import get_latest_results_date
from src.api.schemas.dataset import DatasetConfig


class FineTuningConfig:
    """Configurações do experimento de fine-tuning"""
    
    def __init__(
        self,
        dataset_name: str = "yelp_reviews",
        cache_dir: str = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\.cache",
        results_dir: str = "C:\\Users\\gabri\\Documents\\GitHub\\llm-annotation\\data\\results",
        model_name: str = "roberta-base",
        specific_date: str = "latest",
        learning_rate: float = 5e-5,
        num_epochs: int = 20,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        weight_decay: float = 0.01,
        max_length: int = 256,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        self.specific_date = specific_date
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.weight_decay = weight_decay
        self.max_length = max_length
        self.seed = seed
        
        # Configurações de dataset
        self.dataset_train_config = DatasetConfig(
            split="train",
            combine_splits=[],
            sample_size=None,
            random_state=seed,
        )
        
        self.dataset_eval_config = DatasetConfig(
            split="test",
            combine_splits=[],
            sample_size=5000,
            random_state=seed,
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
    
    def load_annotated_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados anotados"""
        logger.info("Carregando dados anotados...")
        
        df = pd.read_csv(
            self.results_dataset_path / "summary" / "dataset_anotado_completo.csv"
        )
        logger.info(f"Anotado: {len(df)} exemplos")
        
        # Dataset ground truth
        df_gt_train = (
            df[["text_id", "text", "ground_truth"]]
            .rename(columns={"ground_truth": "label"})
        )
        df_gt_train = add_label_description(
            df_gt_train,
            dataset_name=self.config.dataset_name
        )
        
        # Dataset anotado (consenso)
        df_annotations = (
            df[["text_id", "text", "resolved_annotation"]]
            .rename(columns={"resolved_annotation": "label"})
        )
        df_annotations = add_label_description(
            df_annotations,
            dataset_name=self.config.dataset_name
        )
        
        # Remover instâncias problemáticas
        problematic_cases_path = (
            self.results_dataset_path / "consensus" / "problematic_cases.csv"
        )
        
        if problematic_cases_path.exists():
            df_problematic = pd.read_csv(problematic_cases_path)
            df_annotations = df_annotations[
                ~df_annotations["text_id"].isin(df_problematic["text_id"])
            ].reset_index(drop=True)
            logger.info(f"Removidas {len(df_problematic)} instâncias problemáticas")
        
        return df_gt_train, df_annotations
    
    def load_eval_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega dados de validação e teste"""
        logger.info("Carregando dados de avaliação...")
        
        df_eval, _ = load_hf_dataset_as_dataframe(
            dataset_name=self.config.dataset_name,
            cache_dir=self.config.cache_dir,
            dataset_global_config=self.config.dataset_eval_config,
        )
        logger.info(f"Avaliação HF: {len(df_eval)} exemplos")
        
        # Dividir em validação e teste
        df_gt_val, df_gt_test = train_test_split(
            df_eval,
            test_size=0.5,
            random_state=self.config.seed,
            stratify=df_eval["label"]
        )
        
        logger.info(f"Validação: {len(df_gt_val)}, Teste: {len(df_gt_test)}")
        
        return df_gt_val, df_gt_test
    
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
            save_total_limit=1,
            seed=self.config.seed,
        )
    
    def run_fine_tuning(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        eval_dataset: Optional[Dataset],
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
        
        fine_tuner.fit(train_dataset, eval_dataset)
        metrics = fine_tuner.evaluate(test_dataset)
        metrics["source"] = experiment_name
        
        logger.success(f"Fine-tuning concluído: {experiment_name}")
        logger.info(f"Accuracy: {metrics['eval_accuracy']:.4f}, F1 Macro: {metrics['eval_f1_macro']:.4f}")
        
        return metrics
    
    def run(self):
        """Executa pipeline completo"""
        logger.info("=" * 60)
        logger.info("Iniciando pipeline de fine-tuning")
        logger.info("=" * 60)
        
        # Carregar dados
        df_gt_train, df_annotations = self.load_annotated_data()
        df_gt_val, df_gt_test = self.load_eval_data()
        
        # Criar label schema
        label_schema = LabelSchema.from_dataframe(df_gt_train)
        logger.info(f"Labels: {label_schema.id2label}")
        
        # Converter para HuggingFace Dataset
        train_gt = Dataset.from_pandas(df_gt_train)
        train_consensus = Dataset.from_pandas(df_annotations)
        test_dataset = Dataset.from_pandas(df_gt_test)
        eval_dataset = Dataset.from_pandas(df_gt_val)
        
        # Executar fine-tuning com consenso
        logger.info("\n" + "=" * 60)
        logger.info("Fine-tuning com CONSENSO LLM")
        logger.info("=" * 60)
        metrics_consensus = self.run_fine_tuning(
            train_dataset=train_consensus,
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            label_schema=label_schema,
            experiment_name="consensus_llm"
        )
        
        # Executar fine-tuning com ground truth
        logger.info("\n" + "=" * 60)
        logger.info("Fine-tuning com GROUND TRUTH")
        logger.info("=" * 60)
        metrics_gt = self.run_fine_tuning(
            train_dataset=train_gt,
            test_dataset=test_dataset,
            eval_dataset=eval_dataset,
            label_schema=label_schema,
            experiment_name="ground_truth"
        )
        
        # Salvar resultados
        results = pd.DataFrame([metrics_consensus, metrics_gt])
        output_path = self.fine_tune_output_dir / f"{self.config.model_name}_fine_tuning_results.csv"
        results.to_csv(output_path, index=False)
        
        logger.success(f"\nResultados salvos em: {output_path}")
        logger.info("\n" + "=" * 60)
        logger.info("RESULTADOS FINAIS")
        logger.info("=" * 60)
        logger.info(f"\n{results[['source', 'eval_accuracy', 'eval_f1_macro']].to_string(index=False)}")
        
        return results


def main():
    """Função principal"""
    dataset_name = "movie_review"
    model_name = "roberta-base"
    
    # Configuração
    config = FineTuningConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        num_epochs=20,
        learning_rate=5e-5
    )
    
    # Executar pipeline
    pipeline = FineTuningPipeline(config)
    results = pipeline.run()
    
    return results


if __name__ == "__main__":
    main()
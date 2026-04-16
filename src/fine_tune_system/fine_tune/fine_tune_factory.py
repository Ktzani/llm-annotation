from transformers import TrainingArguments

from src.fine_tune_system.core.model_factory import ModelFactory
from src.fine_tune_system.training.metrics import MetricsComputer
from src.fine_tune_system.training.trainer_builder import TrainerBuilder
from src.fine_tune_system.fine_tune.supervised_fine_tuner import SupervisedFineTuner
from src.fine_tune_system.core.tokenizer import Tokenizer
from src.fine_tune_system.training.label_schema import LabelSchema

from copy import deepcopy
from pathlib import Path


class FineTunerFactory:
    def __init__(
        self,
        model_name: str,
        training_args: TrainingArguments,
        label_schema: LabelSchema,
        tokenizer: Tokenizer,
        model_factory: ModelFactory,
        trainer_builder: TrainerBuilder,
        metrics_computer: MetricsComputer,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.label_schema = label_schema
        self.tokenizer = tokenizer
        self.model_factory = model_factory
        self.trainer_builder = trainer_builder
        self.metrics_computer = metrics_computer

    def create(
        self,
        type: str = "supervised",
        fold: int = 0
    ):

        # cópia independente para cada fold
        training_args = deepcopy(self.training_args)

        # pasta específica
        training_args.output_dir = str(
            Path(self.training_args.output_dir, self.model_name, f"fold_{fold}")
        )

        if type == "supervised":
            return SupervisedFineTuner(
                model_name=self.model_name,
                training_args=training_args,
                label_schema=self.label_schema,
                tokenizer=self.tokenizer,
                model_factory=self.model_factory,
                trainer_builder=self.trainer_builder,
                metrics_computer=self.metrics_computer,
            )

        else:
            raise ValueError(
                f"Tipo de fine-tuning '{type}' não suportado."
            )
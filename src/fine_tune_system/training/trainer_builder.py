from transformers import Trainer, TrainingArguments
from datasets import Dataset

from src.fine_tune_system.training.metrics import MetricsComputer


class TrainerBuilder:
    def __init__(self, training_args: TrainingArguments, metrics_computer: MetricsComputer):
        self.training_args = training_args
        self.metrics_computer = metrics_computer

    def build(self, model: str, train_ds: Dataset, eval_ds: Dataset) -> Trainer:
        return Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=self.metrics_computer
        )

import torch
from transformers import TrainingArguments
from datasets import Dataset

from src.fine_tune_system.core.tokenizer import Tokenizer
from src.fine_tune_system.core.model_factory import ModelFactory

from src.fine_tune_system.training.trainer_builder import TrainerBuilder
from src.fine_tune_system.training.metrics import MetricsComputer
from src.fine_tune_system.training.label_schema import LabelSchema

from src.fine_tune_system.fine_tune.fine_tuner import FineTuner

class SupervisedFineTuner(FineTuner):
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

        self._trainer = None
        
    
    def best_val_metrics(self) -> dict:
        if self._trainer is None:
            raise RuntimeError("Fine-tuner must be fitted before evaluation")
    
        best_checkpoint = self._trainer.state.best_model_checkpoint
        best_step = int(best_checkpoint.split("-")[-1])

        for entry in self._trainer.state.log_history:
            if entry.get("step") == best_step and "eval_loss" in entry:
                return {k: v for k, v in entry.items() if k.startswith("eval_") or k == "epoch"}

    def fit(self, train_ds: Dataset, eval_ds: Dataset = None):
        train_ds = self.tokenizer.encode(train_ds)
        
        if eval_ds is not None:
            eval_ds = self.tokenizer.encode(eval_ds)  
              
        model_factory: ModelFactory = self.model_factory(
            self.model_name,
            self.label_schema
        )
        
        model = model_factory.create()

        trainer_builder: TrainerBuilder = self.trainer_builder(
            self.training_args,
            self.metrics_computer
        )

        self._trainer = trainer_builder.build(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds
        )

        self._trainer.train()

    def evaluate(self, test_ds: Dataset) -> dict:
        if self._trainer is None:
            raise RuntimeError("Fine-tuner must be fitted before evaluation")

        test_ds = self.tokenizer.encode(test_ds)
        return self._trainer.evaluate(test_ds)

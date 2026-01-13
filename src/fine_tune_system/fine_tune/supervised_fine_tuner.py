from src.fine_tune_system.core.tokenizer import Tokenizer
from src.fine_tune_system.core.model_factory import ModelFactory

from src.fine_tune_system.training.trainer_builder import TrainerBuilder
from src.fine_tune_system.training.metrics import MetricsComputer
from src.fine_tune_system.training.label_schema import LabelSchema

from src.fine_tune_system.fine_tune.fine_tuner import FineTuner
import torch

class SupervisedFineTuner(FineTuner):
    def __init__(
        self,
        model_name: str,
        training_args,
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

    def fit(self, train_ds, eval_ds = None):
        train_ds = self.tokenizer.encode(train_ds)
        
        if eval_ds is not None:
            eval_ds = self.tokenizer.encode(eval_ds)    
        model = self.model_factory(
            self.model_name,
            self.label_schema
        ).create()

        self._trainer = self.trainer_builder(
            self.training_args,
            self.metrics_computer
        ).build(
            model=model,
            train_ds=train_ds,
            eval_ds=eval_ds
        )

        self._trainer.train()

    def evaluate(self, test_ds) -> dict:
        if self._trainer is None:
            raise RuntimeError("Fine-tuner must be fitted before evaluation")

        test_ds = self.tokenizer.encode(test_ds)
        return self._trainer.evaluate(test_ds)

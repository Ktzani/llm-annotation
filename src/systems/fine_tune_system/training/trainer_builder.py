import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from src.systems.fine_tune_system.training.metrics import MetricsComputer


class SoftLabelTrainer(Trainer):
    """
    Trainer para rótulos em distribuição (soft labels)
    Responsabilidades: entropia cruzada suave quando o rótulo é uma distribuição; entropia cruzada padrão com rótulo inteiro (avaliação)
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is None or not labels.is_floating_point():
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        loss = -(labels.to(log_probs.dtype) * log_probs).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


class TrainerBuilder:
    trainer_class = Trainer

    def __init__(self, training_args: TrainingArguments, metrics_computer: MetricsComputer):
        self.training_args = training_args
        self.metrics_computer = metrics_computer

    def build(self, model: str, train_ds: Dataset, eval_ds: Dataset) -> Trainer:
        return self.trainer_class(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=self.metrics_computer
        )


class SoftLabelTrainerBuilder(TrainerBuilder):
    """Monta o SoftLabelTrainer (regime de soft labels)"""

    trainer_class = SoftLabelTrainer

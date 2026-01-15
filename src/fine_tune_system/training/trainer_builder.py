from transformers import Trainer

class TrainerBuilder:
    def __init__(self, training_args, metrics_computer):
        self.training_args = training_args
        self.metrics_computer = metrics_computer

    def build(self, model, train_ds, eval_ds):
        return Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=self.metrics_computer
        )

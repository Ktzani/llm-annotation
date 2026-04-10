from src.fine_tune_system.fine_tune.supervised_fine_tuner import SupervisedFineTuner

class FineTunerFactory:
    def __init__(
        self,
        model_name,
        training_args,
        label_schema,
        tokenizer,
        model_factory,
        trainer_builder,
        metrics_computer,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.label_schema = label_schema
        self.tokenizer = tokenizer
        self.model_factory = model_factory
        self.trainer_builder = trainer_builder
        self.metrics_computer = metrics_computer

    def create(self, type="supervised"):
        if type == "supervised":
            return SupervisedFineTuner(
                model_name=self.model_name,
                training_args=self.training_args,
                label_schema=self.label_schema,
                tokenizer=self.tokenizer,
                model_factory=self.model_factory,
                trainer_builder=self.trainer_builder,
                metrics_computer=self.metrics_computer,
            )
            
        else: 
            raise ValueError(f"Tipo de fine-tuning '{type}' não suportado.")
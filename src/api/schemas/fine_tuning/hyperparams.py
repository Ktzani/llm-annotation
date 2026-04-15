from pydantic import BaseModel, Field

class FineTuningHyperparams(BaseModel):
    """Hiperparâmetros de treinamento"""

    learning_rate: float = Field(default=5e-5)
    num_epochs: int = Field(default=20)
    train_batch_size: int = Field(default=32)
    eval_batch_size: int = Field(default=64)
    weight_decay: float = Field(default=0.01)
    max_length: int = Field(default=256)
    seed: int = Field(default=42)
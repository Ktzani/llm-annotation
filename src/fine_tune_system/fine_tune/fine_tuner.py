from abc import ABC, abstractmethod

class FineTuner(ABC):

    @abstractmethod
    def fit(self, train_ds, eval_ds=None):
        pass

    @abstractmethod
    def evaluate(self, test_ds) -> dict:
        pass
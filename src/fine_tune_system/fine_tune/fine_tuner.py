from abc import ABC, abstractmethod

class FineTuner(ABC):

    @abstractmethod
    def fit(self, train_ds: list, eval_ds: list = None):
        pass

    @abstractmethod
    def evaluate(self, test_ds: list) -> dict:
        pass
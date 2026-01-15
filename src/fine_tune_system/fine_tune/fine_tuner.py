from abc import ABC, abstractmethod

class FineTuner(ABC):

    @abstractmethod
    def fit(self, train_ds):
        pass

    @abstractmethod
    def evaluate(self, test_ds) -> dict:
        pass
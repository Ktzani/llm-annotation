from abc import ABC, abstractmethod

class Tokenizer(ABC):

    @abstractmethod
    def encode(self, dataset: list) -> list:
        pass
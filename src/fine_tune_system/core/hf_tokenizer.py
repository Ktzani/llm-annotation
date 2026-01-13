from transformers import AutoTokenizer
from src.fine_tune_system.core.tokenizer import Tokenizer

class HFTokenizer(Tokenizer):
    def __init__(self, model_name: str, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            do_lower_case=False
        )
        self.max_length = max_length

    def encode(self, dataset):
        # 1. Garantir labels
        if "label" in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")

        # 2. Tokenizar
        def tokenize_fn(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        dataset = dataset.map(tokenize_fn, batched=True)

        # 3. Remover texto cru
        if "text" in dataset.column_names:
            dataset = dataset.remove_columns(["text"])

        # 4. Setar formato corretamente
        dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

        return dataset

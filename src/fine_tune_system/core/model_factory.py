from transformers import AutoModelForSequenceClassification

class ModelFactory:
    def __init__(self, model_name: str, label_schema):
        self.model_name = model_name
        self.label_schema = label_schema

    def create(self):
        tokenizer = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.label_schema.num_labels(),
            id2label=self.label_schema.id2label,
            label2id={v: k for k, v in self.label_schema.id2label.items()},
            dtype="auto", 
            device_map="auto"
        )
        
        if 'roberta' in self.model_name:
            tokenizer.add_prefix_space = True
            
        return tokenizer

# training/metrics.py
import numpy as np
import evaluate

class MetricsComputer:
    def __init__(self):
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def __call__(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        return {
            "accuracy": self.accuracy.compute(
                predictions=preds,
                references=labels
            )["accuracy"],
            "f1_macro": self.f1.compute(
                predictions=preds,
                references=labels,
                average="macro"
            )["f1"]
        }

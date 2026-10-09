# training/metrics.py
import numpy as np
import evaluate

from datasets import Dataset

from src.systems.fine_tune_system.training.calibration import brier_score, expected_calibration_error, softmax

class MetricsComputer:
    def __init__(self):
        self.accuracy = evaluate.load("accuracy")
        self.f1 = evaluate.load("f1")

    def __call__(self, eval_pred: Dataset):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        probs = softmax(logits)

        return {
            "accuracy": self.accuracy.compute(
                predictions=preds,
                references=labels
            )["accuracy"],
            "f1_macro": self.f1.compute(
                predictions=preds,
                references=labels,
                average="macro"
            )["f1"],
            "ece": expected_calibration_error(probs, labels),
            "brier": brier_score(probs, labels),
        }

from typing import List, Dict, Optional
import numpy as np
from datasets import Dataset

from src.fine_tune_system.fine_tune.fine_tune_factory import FineTunerFactory

class CrossValidator:
    def __init__(self, fine_tuner_factory: FineTunerFactory):
        self.fine_tuner_factory = fine_tuner_factory

    def run(
        self,
        cv_splits: Dict[int, Dict[str, Dataset]],
        test_ds: Optional[Dataset] = None,
        fine_tune_type: Optional[str] = "supervised"
    ) -> Dict:

        fold_metrics = []
        fold_metrics_dict = {}  # ✅ guarda por fold
        test_metrics_all = []

        for fold, split in cv_splits.items():
            print(f"\n🚀 Fold {fold}")

            fine_tuner = self.fine_tuner_factory.create(type=fine_tune_type, fold=fold)

            eval_split = "val" if "val" in split else "test"

            train_ds = split["train"]
            val_ds = split[eval_split]

            # treino
            fine_tuner.fit(
                train_ds=train_ds,
                eval_ds=val_ds
            )

            # validação
            val_metrics = fine_tuner.evaluate(val_ds)

            fold_metrics.append(val_metrics)
            fold_metrics_dict[fold] = val_metrics  # ✅ chave do fold

            print(f"📊 Val metrics (fold {fold}): {val_metrics}")

            # teste opcional
            if test_ds is not None:
                test_metrics = fine_tuner.evaluate(test_ds)
                test_metrics_all.append(test_metrics)

                print(f"🧪 Test metrics: {test_metrics}")

        results = {
            "folds": fold_metrics_dict,  # ✅ métricas individuais
            "cv": self._aggregate_metrics(fold_metrics)
        }

        if test_metrics_all:
            results["test"] = self._aggregate_metrics(test_metrics_all)

        return results

    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        aggregated = {}

        keys = metrics_list[0].keys()

        for key in keys:
            values = [m[key] for m in metrics_list if key in m]

            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

        return aggregated
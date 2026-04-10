from typing import List, Dict, Optional
import numpy as np

from src.fine_tune_system.fine_tune.fine_tune_factory import FineTunerFactory


class CrossValidator:
    def __init__(self, fine_tuner_factory: FineTunerFactory):
        """
        fine_tuner_factory: instância de FineTunerFactory
        """
        self.fine_tuner_factory = fine_tuner_factory

    def run(
        self,
        cv_splits: List[Dict[str, object]],
        test_ds: Optional[object] = None,
        fine_tune_type: Optional[str] = "supervised"
    ) -> Dict:

        fold_metrics = []
        test_metrics_all = []

        for i, split in enumerate(cv_splits):
            print(f"\n🚀 Fold {i+1}/{len(cv_splits)}")

            # 🔥 cria um NOVO fine-tuner a cada fold
            fine_tuner = self.fine_tuner_factory.create(type=fine_tune_type)

            # treino
            fine_tuner.fit(
                train_ds=split["train"],
                eval_ds=split["val"]
            )

            # validação
            val_metrics = fine_tuner.evaluate(split["val"])
            fold_metrics.append(val_metrics)

            print(f"📊 Val metrics: {val_metrics}")

            # test opcional
            if test_ds is not None:
                test_metrics = fine_tuner.evaluate(test_ds)
                test_metrics_all.append(test_metrics)

                print(f"🧪 Test metrics: {test_metrics}")

        results = {
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
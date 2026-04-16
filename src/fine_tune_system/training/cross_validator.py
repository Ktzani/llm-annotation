import os
from typing import List, Dict, Optional
import numpy as np
from datasets import Dataset
from loguru import logger

import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.fine_tune_system.fine_tune.fine_tune_factory import FineTunerFactory

class CrossValidator:
    def __init__(self, fine_tuner_factory: FineTunerFactory, max_parallel_folds: int = 4):
        self.fine_tuner_factory = fine_tuner_factory
        self.max_parallel_folds = max_parallel_folds

    def run(
        self,
        cv_splits: Dict[int, Dict[str, Dataset]],
        fine_tune_type: Optional[str] = "supervised",
    ) -> Dict:

        fold_metrics = []
        fold_metrics_dict = {}
        test_metrics_all = []
        
        if torch.cuda.is_available():
            print('CUDA:', torch.cuda.is_available())
            print('Device:', torch.cuda.get_device_name(0))
            print('Current:', torch.cuda.current_device())
            
        else:
            logger.warning("⚠️ CUDA não disponível. Rodando na CPU (muito lento!)")
            raise RuntimeError("CUDA não disponível")

        with ProcessPoolExecutor(
            max_workers=self.max_parallel_folds
        ) as executor:

            futures = {
                executor.submit(
                    self._run_single_fold,
                    fold,
                    split,
                    fine_tune_type,
                ): fold
                for fold, split in cv_splits.items()
            }

            for future in as_completed(futures):
                result = future.result()

                fold = result["fold"]
                val_metrics = result["val_metrics"]

                fold_metrics.append(val_metrics)
                fold_metrics_dict[fold] = val_metrics

                print(f"📊 Fold {fold} finalizado")
                print(val_metrics)

                if "test_metrics" in result:
                    test_metrics_all.append(result["test_metrics"])

        results = {
            "folds": fold_metrics_dict,
            "cv": self._aggregate_metrics(fold_metrics),
        }

        if test_metrics_all:
            results["test"] = self._aggregate_metrics(
                test_metrics_all
            )

        return results
    
    def _run_single_fold(
        self,
        fold: int,
        split: Dict[str, Dataset],
        fine_tune_type: str,
    ) -> Dict:
        logger.info(f"\n🚀 Iniciando Fold {fold}")

        fine_tuner = self.fine_tuner_factory.create(
            type=fine_tune_type,
            fold=fold,
        )

        eval_split = "val" if "val" in split else "test"

        train_ds = split["train"]
        val_ds = split[eval_split]

        fine_tuner.fit(
            train_ds=train_ds,
            eval_ds=val_ds,
        )

        result = {
            "fold": fold,
            "val_metrics": fine_tuner.best_val_metrics(),
        }

        logger.info(f"✅ Fold {fold} concluído")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _aggregate_metrics(
        self,
        metrics_list: List[Dict],
    ) -> Dict:

        aggregated = {}

        keys = metrics_list[0].keys()

        for key in keys:
            values = [
                m[key]
                for m in metrics_list
                if key in m
            ]

            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        return aggregated
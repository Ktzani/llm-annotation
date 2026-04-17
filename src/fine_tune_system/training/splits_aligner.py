import pandas as pd
from typing import Dict, List
from loguru import logger
from datasets import Dataset

from src.utils.get_text_id_from_text import get_text_id_from_text


class CVSplitAligner:
    def __init__(
        self,
        dataset: Dict[int, Dict[str, pd.DataFrame]],
        id_column: str = "text_id",
    ):
        """
        dataset: dataset carregado do HF -> {fold: {"train": df, "val": df}}
        id_column: coluna única que identifica cada texto
        """
        self.dataset = dataset
        self.id_column = id_column
        
        self._add_ids()
        

    # -------------------------
    # 🧩 Adicionar IDs ao dataset do HF
    # -------------------------
    def _add_ids(self):
        """
        Adiciona text_id baseado na ordem dos dados (caso HF não tenha)
        """
        for split in self.dataset.keys():
            # TRAIN
            if "text_id" not in self.dataset[split]["train"].columns:
                self.dataset[split]["train"]["text_id"] = self.dataset[split]["train"]["text"].apply(get_text_id_from_text)

            # VAL OU TEST
            target_split = "val" if "val" in self.dataset[split] else "test"

            if "text_id" not in self.dataset[split][target_split].columns:
                self.dataset[split][target_split]["text_id"] = self.dataset[split][target_split]["text"].apply(get_text_id_from_text)
            
    # -------------------------
    # 🔗 Alinhar split
    # -------------------------
    def align_split(
        self,
        annotated_df: pd.DataFrame,
        annotated_ids: set,
        fold: int,
        fold_data: Dict[str, pd.DataFrame],
        aligned_splits: dict[int, dict[str, Dataset]],
    ) -> None:
        aligned_splits[fold] = {}

        for split_name, hf_df in fold_data.items():

            if not isinstance(hf_df, pd.DataFrame):
                hf_df = hf_df.to_pandas()

            split_ids = set(hf_df[self.id_column])

            aligned_df = annotated_df[
                annotated_df[self.id_column].isin(split_ids)
            ].copy()

            # 🔍 cobertura
            coverage = len(aligned_df) / len(hf_df)

            missing = split_ids - annotated_ids

            logger.warning(
                f"HF: {len(hf_df)} | Annotated: {len(aligned_df)} "
                f"| Coverage: {coverage:.4f} | Missing: {len(missing)}"
            )

            aligned_ds = Dataset.from_pandas(aligned_df)
            aligned_splits[fold][split_name] = aligned_ds

            logger.info(
                f"Fold {fold} | {split_name}: {len(aligned_df)} exemplos anotados"
            )

    # -------------------------
    # 📦 Alinhar todos splits
    # -------------------------
    def align_datasets_splits(
        self,
        annotated_df: pd.DataFrame,
    ) -> Dict[int, Dict[str, Dataset]]:
        """Alinha splits usando apenas dados anotados (HF só define os IDs)"""

        # 🔴 sanity check crítico
        assert annotated_df[self.id_column].is_unique, \
            "❌ IDs duplicados no dataset anotado!"

        aligned_splits = {}
        annotated_ids = set(annotated_df[self.id_column])

        for fold, fold_data in self.dataset.items():
            logger.info(f"🔄 Alinhando fold {fold}...")

            self.align_split(
                annotated_df,
                annotated_ids,
                fold,
                fold_data,
                aligned_splits
            )

        # -------------------------
        # 🚨 Data leakage check
        # -------------------------
        for fold, data in aligned_splits.items():

            train_ids = set(data["train"][self.id_column])

            # suporta "val" OU "test"
            eval_split = "val" if "val" in data else "test"
            test_ids = set(data[eval_split][self.id_column])

            intersection = train_ids & test_ids

            assert len(intersection) == 0, \
                f"❌ Data leakage no fold {fold}! ({len(intersection)} exemplos)"

        return aligned_splits
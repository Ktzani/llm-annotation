from langchain_huggingface import data
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
        allow_duplicate_ids: bool = False,
    ):
        """
        dataset: dataset carregado do HF -> {fold: {"train": df, "val": df}}
        id_column: coluna única que identifica cada texto
        allow_duplicate_ids: se True, permite `id_column` repetido no conjunto
            anotado (modo perspectivismo: uma linha por LLM por texto). Quando
            False (default), mantém a checagem de unicidade do modo agregado.
        """
        self.dataset = dataset
        self.id_column = id_column
        self.allow_duplicate_ids = allow_duplicate_ids

        self._add_ids()
        

    # -------------------------
    # Adicionar IDs ao dataset do HF
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
    # Alinhar split
    # -------------------------
    def align_split(
        self,
        annotated_df: pd.DataFrame,
        fold: int,
        fold_data: Dict[str, pd.DataFrame],
        aligned_splits: dict[int, dict[str, Dataset]],
    ) -> None:
        aligned_splits[fold] = {}

        for split_name, hf_df in fold_data.items():

            if not isinstance(hf_df, pd.DataFrame):
                hf_df = hf_df.to_pandas()

            if split_name == "train":
                # Treino: labels do consenso LLM (annotated_df)
                split_ids = set(hf_df[self.id_column])

                aligned_df = annotated_df[
                    annotated_df[self.id_column].isin(split_ids)
                ].copy()

                coverage = len(aligned_df) / len(hf_df)
                missing = split_ids - set(annotated_df[self.id_column])

                logger.warning(
                    f"[train|consenso] HF: {len(hf_df)} | Annotated: {len(aligned_df)} "
                    f"| Coverage: {coverage:.4f} | Missing: {len(missing)}"
                )
            else:
                # Val/test: labels do GT original do HuggingFace (hf_df)
                aligned_df = hf_df.copy()

                logger.warning(
                    f"[{split_name}|GT] usando {len(aligned_df)} exemplos com labels do HuggingFace (ground-truth)"
                )

            aligned_ds = Dataset.from_pandas(aligned_df)
            aligned_splits[fold][split_name] = aligned_ds

            logger.info(
                f"Fold {fold} | {split_name}: {len(aligned_df)} exemplos"
            )

    # -------------------------
    # Alinhar todos splits
    # -------------------------
    def align_datasets_splits(
        self,
        annotated_df: pd.DataFrame,
    ) -> Dict[int, Dict[str, Dataset]]:
        """Alinha splits usando apenas dados anotados (HF só define os IDs)"""

        # 🔴 sanity check crítico (relaxado no modo perspectivismo, onde o mesmo
        # texto aparece uma vez por LLM)
        if not self.allow_duplicate_ids:
            assert annotated_df[self.id_column].is_unique, \
                "❌ IDs duplicados no dataset anotado!"
        elif not annotated_df[self.id_column].is_unique:
            n_rows = len(annotated_df)
            n_texts = annotated_df[self.id_column].nunique()
            logger.info(
                f"[perspectivismo] IDs repetidos permitidos: "
                f"{n_rows} linhas para {n_texts} textos únicos."
            )

        aligned_splits = {}

        for fold, fold_data in self.dataset.items():
            logger.info(f"🔄 Alinhando fold {fold}...")

            self.align_split(
                annotated_df,
                fold,
                fold_data,
                aligned_splits
            )

        # -------------------------
        # Data leakage check + correção (remove do val/test o que está no train)
        # -------------------------
        aligned_splits = self._remove_data_leakage(aligned_splits)

        return aligned_splits
    
    def _remove_data_leakage(self, aligned_splits):

        for fold, data in aligned_splits.items():
        
            train_ds = data["train"]
            eval_split = "val" if "val" in data else "test"
            eval_ds = data[eval_split]
    
            leaked_ids = set(train_ds[self.id_column]) & set(eval_ds[self.id_column])
    
            if leaked_ids:
                logger.warning(
                    f"Leakage no fold {fold}: removendo {len(leaked_ids)} exemplos de {eval_split}"
                )
    
                aligned_splits[fold][eval_split] = eval_ds.filter(
                    lambda x: x[self.id_column] not in leaked_ids
                )
    
        return aligned_splits
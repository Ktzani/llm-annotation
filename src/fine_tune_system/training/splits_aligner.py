import pandas as pd
from typing import Dict, List


class CVSplitAligner:
    def __init__(
        self,
        hf_dataset: Dict,
        id_column: str = "text_id",
    ):
        """
        hf_dataset: dataset carregado do HF
        id_column: coluna única que identifica cada texto
        """
        self.hf_dataset = hf_dataset
        self.id_column = id_column

    # -------------------------
    # 🧩 Adicionar IDs ao HF (se necessário)
    # -------------------------
    def add_ids_to_hf(self):
        """
        Adiciona text_id baseado na ordem dos dados (caso HF não tenha)
        """
        for split in self.hf_dataset.keys():
            df = self.hf_dataset[split].to_pandas().reset_index(drop=True)
            df[self.id_column] = df.index

            self.hf_dataset[split] = df

    # -------------------------
    # 🔗 Alinhar split
    # -------------------------
    def align_split(
        self,
        annotated_df: pd.DataFrame,
        split_name: str,
    ) -> pd.DataFrame:

        hf_df = self.hf_dataset[split_name]

        if not isinstance(hf_df, pd.DataFrame):
            hf_df = hf_df.to_pandas()

        merged = hf_df.merge(
            annotated_df,
            on=self.id_column,
            suffixes=("_hf", "_annot"),
        )

        # 🔍 sanity checks fortes (importante pra paper)
        if len(merged) != len(hf_df):
            missing = set(hf_df[self.id_column]) - set(annotated_df[self.id_column])
            print(f"[WARN] Split '{split_name}' não alinhou 100%")
            print(f"HF: {len(hf_df)} | Merged: {len(merged)}")
            print(f"Missing IDs: {len(missing)}")

        return merged

    # -------------------------
    # 📦 Alinhar todos splits
    # -------------------------
    def align_all_splits(
        self,
        annotated_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:

        aligned = {}

        for split in self.hf_dataset.keys():
            aligned[split] = self.align_split(annotated_df, split)

        return aligned

    # -------------------------
    # 🔁 Cross-validation via coluna fold
    # -------------------------
    def build_cv_from_column(
        self,
        annotated_df: pd.DataFrame,
        fold_column: str,
    ) -> List[Dict[str, pd.DataFrame]]:

        df = self.align_split(annotated_df, "train")

        folds = sorted(df[fold_column].unique())
        splits = []

        for fold in folds:
            train_fold = df[df[fold_column] != fold]
            val_fold = df[df[fold_column] == fold]

            splits.append({
                "train": train_fold,
                "val": val_fold,
            })

        return splits

    # -------------------------
    # 🔁 KFold fallback
    # -------------------------
    def build_kfold(
        self,
        annotated_df: pd.DataFrame,
        n_splits: int = 5,
        shuffle: bool = False,
    ) -> List[Dict[str, pd.DataFrame]]:

        from sklearn.model_selection import KFold

        df = self.align_split(annotated_df, "train")

        kf = KFold(n_splits=n_splits, shuffle=shuffle)

        splits = []

        for train_idx, val_idx in kf.split(df):
            splits.append({
                "train": df.iloc[train_idx],
                "val": df.iloc[val_idx],
            })

        return splits
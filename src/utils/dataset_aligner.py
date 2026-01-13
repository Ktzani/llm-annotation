import pandas as pd

class DatasetAligner:
    def __init__(self, id_column: str, gt_label: str, ann_label: str):
        self.id_column = id_column
        self.gt_label = gt_label
        self.ann_label = ann_label

    def align(self, df_gt: pd.DataFrame, df_ann: pd.DataFrame):
        gt = df_gt[[self.id_column, "text", self.gt_label]]
        ann = df_ann[[self.id_column, self.ann_label]]

        merged = gt.merge(
            ann,
            on=self.id_column,
            how="inner",
            validate="one_to_one"
        )
    

        gt_aligned = merged[["text", self.gt_label]].rename(
            columns={self.gt_label: "label"}
        )

        ann_aligned = merged[["text", self.ann_label]].rename(
            columns={self.ann_label: "label"}
        )

        return gt_aligned, ann_aligned

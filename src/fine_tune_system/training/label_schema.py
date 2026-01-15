class LabelSchema:
    def __init__(self, id2label: dict[int, str]):
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}

    @classmethod
    def from_dataframe(cls, df):
        mapping = (
            df[["label", "label_description"]]
            .drop_duplicates()
            .sort_values("label")
        )
        id2label = dict(zip(mapping["label"], mapping["label_description"]))
        return cls(id2label)

    def num_labels(self):
        return len(self.id2label)

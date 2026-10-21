"""
Binary Labels Builder - Monta as tabelas da anotação binária (itens para o consenso e planilhas texto × classe)
"""

from typing import List
import pandas as pd
from loguru import logger


class BinaryLabelsBuilder:
    """
    Monta as tabelas da anotação binária
    Responsabilidades: annotations.csv no formato do ConsensusPipeline e as planilhas texto × classe
    """

    # O item do consenso é o par (texto, classe): o text_id do annotations.csv vira "<text_id>:<classe>"
    TEXT_KEY = "source_text_id"

    def __init__(self):
        logger.debug("BinaryLabelsBuilder inicializado")

    def annotations_by_class(self, df_answers: pd.DataFrame, models: List[str]) -> pd.DataFrame:
        """Uma linha por (texto, classe) com a resposta de cada modelo em `{modelo}_consensus` (1, 0 ou -1)"""
        df_answers = df_answers[df_answers["model"].isin(models)]

        keys = ["text_id", "class_code"]
        df_items = df_answers.drop_duplicates(keys)[keys + ["text", "text_len", "class_name"]].set_index(keys)

        # Pergunta sem resposta no checkpoint (ex.: modelo novo ainda não rodado) conta como inválida
        answers = df_answers.pivot_table(index=keys, columns="model", values="answer", aggfunc="first")
        answers = answers.reindex(index=df_items.index, columns=models).fillna(-1).astype(int)
        invalid = int((answers == -1).sum().sum())
        if invalid:
            logger.warning(f"{invalid} respostas inválidas ou ausentes no checkpoint (ficam -1)")

        for model in models:
            df_items[f"{model}_consensus"] = answers[model]

        df_items = df_items.reset_index().rename(columns={"text_id": self.TEXT_KEY})
        df_items.insert(0, "text_id", df_items[self.TEXT_KEY] + ":" + df_items["class_code"].astype(str))
        return df_items

    def labels_by_text(self, df_items: pd.DataFrame, column: str) -> pd.DataFrame:
        """Uma linha por texto e uma coluna por classe com `column` (1, 0 ou -1): a resposta de um modelo ou o consenso"""
        classes = df_items.drop_duplicates("class_code").sort_values("class_code")
        class_columns = dict(zip(classes["class_code"], classes["class_name"].astype(str)))

        values = pd.to_numeric(df_items[column], errors="coerce").fillna(-1).astype(int)
        labels = df_items.assign(**{column: values}).pivot(index=self.TEXT_KEY, columns="class_code", values=column)
        labels = labels[list(class_columns)].rename(columns=class_columns)

        texts = df_items.drop_duplicates(self.TEXT_KEY).set_index(self.TEXT_KEY)[["text", "text_len"]]
        df_labels = texts.join(labels)
        df_labels["num_yes"] = (labels == 1).sum(axis=1)
        df_labels["num_invalid"] = (labels == -1).sum(axis=1)

        logger.info(
            f"Planilha ({column}): {len(df_labels)} textos | sim por texto (média): {df_labels['num_yes'].mean():.2f} | "
            f"textos sem nenhum sim: {(df_labels['num_yes'] == 0).sum()} | "
            f"textos com alguma classe inválida: {(df_labels['num_invalid'] > 0).sum()}"
        )
        return df_labels.rename_axis("text_id").reset_index()

    @staticmethod
    def expand_to_rows(df_labels: pd.DataFrame, df_rows: pd.DataFrame) -> pd.DataFrame:
        """Uma linha por linha do dataset (keep_columns na frente): textos repetidos recebem o mesmo conjunto"""
        return df_rows.merge(df_labels, on="text_id", how="left")

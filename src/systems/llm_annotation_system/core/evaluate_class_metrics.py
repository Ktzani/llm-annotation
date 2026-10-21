from sklearn.metrics import f1_score, precision_score, recall_score

from pathlib import Path
import pandas as pd
from loguru import logger

def evaluate_class_metrics(
        df_items: pd.DataFrame,
        models: list,
        output_dir: Path,
        ground_truth_col: str = "ground_truth",
    ) -> pd.DataFrame:
        """
        Precisão, recall e F1 de cada classe (um contra todos) para cada modelo; ground_truth do item é 1/0.
        Resposta inválida (-1) conta como "não" no F1; a taxa de inválidas sai em invalid_rate.
        """

        logger.info("Calculando métricas por classe...")

        df = df_items[df_items[ground_truth_col].notna()]
        if df.empty:
            logger.warning("Sem ground truth nas respostas — métricas por classe puladas")
            return pd.DataFrame()

        is_true_class = df[ground_truth_col].astype(int)

        answers_by_annotator = {model: df[f"{model}_consensus"] for model in models}

        results = []

        for annotator, answers in answers_by_annotator.items():
            predicted = (answers == 1).astype(int)
            invalid = answers == -1

            for class_code, index in df.groupby("class_code").groups.items():
                y_true = is_true_class.loc[index]
                y_pred = predicted.loc[index]

                results.append({
                    "annotator": annotator,
                    "class_code": class_code,
                    "class_name": df.loc[index, "class_name"].iloc[0],
                    "support": int(y_true.sum()),
                    "predicted_yes": int(y_pred.sum()),
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "invalid_rate": float(invalid.loc[index].mean()),
                })

        df_metrics = pd.DataFrame(results)

        for annotator, group in df_metrics.groupby("annotator", sort=False):
            logger.info(f"Métricas por classe {annotator}: F1 macro={group['f1'].mean():.4f} | Recall macro={group['recall'].mean():.4f}")

        output_path = output_dir / "class_metrics.csv"
        df_metrics.to_csv(output_path, index=False)
        logger.success(f"Métricas por classe salvas em: {output_path}")

        return df_metrics

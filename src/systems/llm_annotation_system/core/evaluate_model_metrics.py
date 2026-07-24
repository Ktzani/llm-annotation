from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pathlib import Path
import pandas as pd
from loguru import logger

def evaluate_model_metrics(
        df: pd.DataFrame,
        models: list,
        output_dir: Path,
        ground_truth_col: str = "ground_truth",
    ) -> pd.DataFrame:
        """
        Calcula métricas por modelo, considerando -1 como classe de erro válida.
        Não remove as linhas com -1, pois isso faz parte da avaliação.
        """

        logger.info("Calculando métricas por modelo...")

        model_consensus_cols = {
            model: f"{model}_consensus"
            for model in models
            if f"{model}_consensus" in df.columns
        }

        if len(model_consensus_cols) == 0:
            logger.error("Nenhuma coluna *_consensus encontrada no DataFrame.")
            return pd.DataFrame()

        df_clean = df.copy()

        for col in model_consensus_cols.values():
            df_clean[col] = df_clean[col].replace(
                {"ERROR": -1, None: -1, "": -1, "N/A": -1}
            )

        df_clean = df_clean[df_clean[ground_truth_col].notna()]

        for col in model_consensus_cols.values():
            df_clean[col] = df_clean[col].astype(int)

        df_clean[ground_truth_col] = df_clean[ground_truth_col].astype(int)

        logger.info(f"Total de linhas avaliadas: {len(df_clean)}")

        results = []

        for model_name, col in model_consensus_cols.items():

            y_true = df_clean[ground_truth_col]
            y_pred = df_clean[col]
            
            # Filtrar instâncias válidas (remover -1)
            valid_mask = (y_true != -1) & (y_pred != -1)
            
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            # Métricas apenas nas predições válidas
            if len(y_true_valid) > 0:
                acc = accuracy_score(y_true_valid, y_pred_valid)
                f1 = f1_score(y_true_valid, y_pred_valid, average="macro")
                prec = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
                rec = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
            else:
                acc, f1, prec, rec = 0, 0, 0, 0

            # Coverage: % de predições != -1
            coverage = (y_pred != -1).mean()

            results.append({
                "model": model_name,
                "accuracy": acc,
                "f1_macro": f1,
                "precision_macro": prec,
                "recall_macro": rec,
                "coverage": coverage,
                "error_rate": 1 - acc,
                "invalid_predictions_rate": 1 - coverage
            })
            
            logger.info(f"Métricas para {model_name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, Cov={coverage:.4f}")

        df_metrics = pd.DataFrame(results)
        df_metrics = df_metrics.sort_values("f1_macro", ascending=False)
        
        output_path = output_dir / "model_metrics.csv"
        df_metrics.to_csv(output_path, index=False)
        logger.success(f"Métricas por modelo salvas em: {output_path}")

        logger.success("✓ Métricas calculadas com sucesso")

        return df_metrics
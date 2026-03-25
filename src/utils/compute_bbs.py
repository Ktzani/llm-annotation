import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def compute_bbs(df, conf_col, label_col, gt_col="ground_truth", n_bins=10, seed=42):
    np.random.seed(seed)

    df = df[[conf_col, label_col, gt_col]].dropna().copy()

    # Confiança máxima (ĉ)
    df["conf"] = df[conf_col]

    # Correção (1 ou 0)
    df["correct"] = (df[label_col] == df[gt_col]).astype(int)

    # Criar bins
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=False, include_lowest=True)

    balanced_rows = []

    for b in df["bin"].dropna().unique():
        df_bin = df[df["bin"] == b]

        C_m = df_bin[df_bin["correct"] == 1]
        I_m = df_bin[df_bin["correct"] == 0]

        if len(C_m) == 0 or len(I_m) == 0:
            continue

        k = min(len(C_m), len(I_m))

        C_sample = C_m.sample(n=k, random_state=seed)
        I_sample = I_m.sample(n=k, random_state=seed)

        balanced_rows.append(pd.concat([C_sample, I_sample]))

    if len(balanced_rows) == 0:
        return np.nan

    D_balanced = pd.concat(balanced_rows)

    # Brier Score: (p - y)^2
    return brier_score_loss(D_balanced["correct"], D_balanced["conf"])

def compute_bbs_per_bin(df, conf_col, label_col, gt_col="ground_truth", n_bins=10, seed=42):
    np.random.seed(seed)

    df = df[[conf_col, label_col, gt_col]].dropna().copy()
    df["conf"] = df[conf_col]
    df["correct"] = (df[label_col] == df[gt_col]).astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["conf"], bins=bins, labels=False, include_lowest=True)

    results = []

    for b in sorted(df["bin"].dropna().unique()):
        df_bin = df[df["bin"] == b]

        C_m = df_bin[df_bin["correct"] == 1]
        I_m = df_bin[df_bin["correct"] == 0]

        if len(C_m) == 0 or len(I_m) == 0:
            continue

        k = min(len(C_m), len(I_m))

        C_sample = C_m.sample(n=k, random_state=seed)
        I_sample = I_m.sample(n=k, random_state=seed)

        df_bal = pd.concat([C_sample, I_sample])

        brier = np.mean((df_bal["conf"] - df_bal["correct"]) ** 2)

        results.append({
            "bin": b,
            "mean_conf": df_bin["conf"].mean(),
            "bbs_bin": brier
        })

    return pd.DataFrame(results)
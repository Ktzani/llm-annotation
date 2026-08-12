"""
(Re)computa o recall@k de um run de 2 fases com MICRO e MACRO.

Reexecuta a Fase 1 (LR sobre TF-IDF) por fold — determinístico, sem LLM — e
salva `recall_at_k_all_folds.csv` e `recall_at_k_aggregated.csv` já com as duas
métricas: `recall_at_k` (micro, teto da accuracy) e `recall_at_k_macro` (macro,
teto do f1-macro — coerente com dataset desbalanceado).

Útil para runs antigos que só tinham o micro. Runs novos já saem com as duas.

Uso:
    python -m scripts.recompute_recall --dataset dblp
    python -m scripts.recompute_recall --dataset books --run <ts_dir>
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.data_loader import load_hf_dataset_as_dataframe
from src.api.schemas.annotation_experiment.dataset import DatasetConfig
from src.systems.class_filter_system.folds.inner_split import inner_split
from src.systems.class_filter_system.classifiers.factory import get_class_filter
from src.systems.class_filter_system.validation.recall_at_k import recall_at_k_sweep
from src.utils.get_text_id_from_text import get_text_id_from_text
from scripts.compare_filtered_vs_baseline import latest_two_phase_run

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description="Recomputa recall@k (micro+macro) de um run de 2 fases.")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", default=None, help="Run de 2 fases (default: mais recente).")
    ap.add_argument("--cache-dir", default="data/.cache")
    ap.add_argument("--n-inner", type=int, default=5)
    ap.add_argument("--holdout", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold-pattern", default="train_fold_{fold}.parquet")
    args = ap.parse_args()

    ds_dir = Path(args.results_root) / args.dataset
    run_dir = Path(args.run) if args.run else latest_two_phase_run(ds_dir)
    if run_dir is None:
        print(f"Sem run de 2 fases em {ds_dir}.")
        return

    # k's: reaproveita os do run (se houver), senão [2,3,4]
    allf = run_dir / "recall_at_k_all_folds.csv"
    k_values = sorted(pd.read_csv(allf)["k"].unique().tolist()) if allf.exists() else [2, 3, 4]

    print(f"Dataset {args.dataset} | run {run_dir} | k={k_values}")
    frames = []
    fold = 0
    while True:
        cfg = DatasetConfig(hf_file=args.fold_pattern.format(fold=fold), random_state=args.seed)
        try:
            df, _ = load_hf_dataset_as_dataframe(args.dataset, args.cache_dir, cfg)
        except Exception:
            break
        df = df.copy()
        df["text"] = df["text"].astype(str)
        df["text_id"] = df["text"].apply(get_text_id_from_text)
        df = df.drop_duplicates("text_id")

        df_fit, df_ho = inner_split(df, args.n_inner, args.holdout, args.seed, "label")
        filt = get_class_filter("logistic_regression", random_state=args.seed).fit(
            df_fit["text"].tolist(), df_fit["label"].tolist()
        )
        proba = filt.predict_proba(df_ho["text"].tolist())
        dfr = recall_at_k_sweep(df_ho["label"].tolist(), proba, filt.classes_, k_values)
        dfr.insert(0, "fold", fold)
        frames.append(dfr)
        fold += 1

    if not frames:
        print("Nenhum fold encontrado.")
        return

    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(run_dir / "recall_at_k_all_folds.csv", index=False)

    metric_cols = [c for c in ("recall_at_k", "recall_at_k_macro") if c in df_all.columns]
    df_agg = df_all.groupby("k")[metric_cols].agg(["mean", "std", "count"]).reset_index()
    df_agg.columns = ["k" if c[0] == "k" else f"{c[0]}_{c[1]}" for c in df_agg.columns]
    df_agg.to_csv(run_dir / "recall_at_k_aggregated.csv", index=False)

    show = df_agg[["k", "recall_at_k_mean", "recall_at_k_macro_mean"]].round(4)
    show.columns = ["k", "micro (accuracy)", "macro (f1-macro)"]
    print(show.to_string(index=False))
    print(f"[OK] salvo em {run_dir}")


if __name__ == "__main__":
    main()

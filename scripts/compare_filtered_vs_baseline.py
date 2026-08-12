"""
Comparação: anotação FILTERED (2 fases) vs BASELINE já rodado, nos MESMOS textos.

Ideia
-----
O baseline (todas as classes) já foi anotado sobre o dataset inteiro. Os held-out
da 2 fases são um subconjunto desse dataset. Como o LLM anota texto a texto
(zero-shot), a predição baseline de cada texto já existe — basta cruzar pelo
`text_id`. Assim comparamos filtered vs baseline nos MESMOS textos, isolando só o
efeito de reduzir a lista de classes, SEM re-rodar o baseline.

Para cada dataset com um run de 2 fases:
  1. Junta os `fold_*/filtered/annotations.csv` (held-out anotados), por fold.
  2. Pega o ÚLTIMO diretório de baseline do dataset com `annotations.csv`.
  3. Cruza por `text_id` e calcula accuracy / f1_macro (filtered vs baseline) por
     modelo e POR FOLD, tratando -1 como inválido (igual ao evaluate).
  4. Agrega entre folds pela MÉDIA (± desvio).
  5. Gera CSVs (por-fold e agregado) e um relatório em Markdown.

Uso
---
    python -m scripts.compare_filtered_vs_baseline
    python -m scripts.compare_filtered_vs_baseline --datasets books
    python -m scripts.compare_filtered_vs_baseline --run <path/two_phase/<ts>> --baseline <dir_ou_csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

try:
    from src.config.datasets_collected import LABEL_MEANINGS
except Exception:
    LABEL_MEANINGS = {}

INVALID = -1


# ----------------------------------------------------------------------
# Descoberta de diretórios
# ----------------------------------------------------------------------
def latest_two_phase_run(dataset_dir: Path) -> Path | None:
    """Último run de 2 fases com ao menos um fold_*/filtered/annotations.csv."""
    tp = dataset_dir / "two_phase"
    if not tp.is_dir():
        return None
    for run in sorted([d for d in tp.iterdir() if d.is_dir()], reverse=True):
        if any(run.glob("fold_*/filtered/annotations.csv")):
            return run
    return None


def latest_baseline_dir(dataset_dir: Path) -> Path | None:
    """Último diretório de baseline do dataset que contenha `annotations.csv`."""
    for d in sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name != "two_phase"],
        reverse=True,
    ):
        if (d / "annotations.csv").exists():
            return d
    return None


# ----------------------------------------------------------------------
# Carregamento / métricas
# ----------------------------------------------------------------------
def consensus_models(df: pd.DataFrame) -> list[str]:
    return [c[: -len("_consensus")] for c in df.columns if c.endswith("_consensus")]


def coerce_int(series: pd.Series) -> pd.Series:
    """ERROR/None/''/N/A → -1, depois int (igual ao evaluate_model_metrics)."""
    return (
        series.replace({"ERROR": INVALID, None: INVALID, "": INVALID, "N/A": INVALID})
        .fillna(INVALID)
        .astype(float)
        .astype(int)
    )


def load_filtered(run_dir: Path) -> pd.DataFrame:
    """Concatena todos os held-out filtered do run, marcando o fold de origem."""
    frames = []
    for f in sorted(run_dir.glob("fold_*/filtered/annotations.csv")):
        df = pd.read_csv(f)
        df["fold"] = int(f.parts[-3].split("_")[1])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum fold_*/filtered/annotations.csv em {run_dir}")
    return pd.concat(frames, ignore_index=True)


def _acc_f1(y_true, y_pred) -> tuple[float, float]:
    if len(y_true) == 0:
        return 0.0, 0.0
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def load_recall_ceiling(run_dir: Path) -> pd.DataFrame | None:
    """
    Teto do filtro por k. Devolve `recall@k` MACRO (teto do f1-macro, primário) e
    MICRO (teto da accuracy). Lida com o formato novo (colunas achatadas) e o
    antigo (só micro, colunas mean/std/count).
    """
    p = run_dir / "recall_at_k_aggregated.csv"
    if p.exists():
        agg = pd.read_csv(p)
    else:
        p2 = run_dir / "recall_at_k_all_folds.csv"
        if not p2.exists():
            return None
        df = pd.read_csv(p2)
        cols = [c for c in ("recall_at_k", "recall_at_k_macro") if c in df.columns]
        agg = df.groupby("k")[cols].mean().reset_index()

    out = pd.DataFrame({"k": agg["k"].astype(int)})
    if "recall_at_k_macro_mean" in agg.columns:          # novo formato agregado
        out["recall_macro (f1)"] = agg["recall_at_k_macro_mean"]
        out["recall_micro (acc)"] = agg["recall_at_k_mean"]
    elif "recall_at_k_macro" in agg.columns:             # média do all_folds
        out["recall_macro (f1)"] = agg["recall_at_k_macro"]
        out["recall_micro (acc)"] = agg["recall_at_k"]
    elif "mean" in agg.columns:                          # formato antigo (micro só)
        out["recall_micro (acc)"] = agg["mean"]
    elif "recall_at_k" in agg.columns:
        out["recall_micro (acc)"] = agg["recall_at_k"]
    return out


# ----------------------------------------------------------------------
# Comparação
# ----------------------------------------------------------------------
def compare_per_fold(dataset: str, run_dir: Path, baseline_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Métricas filtered vs baseline por (fold, modelo)."""
    filtered = load_filtered(run_dir)
    baseline = pd.read_csv(baseline_dir / "annotations.csv")

    models_f = consensus_models(filtered)
    models_b = consensus_models(baseline)
    common = [m for m in models_f if m in models_b]
    if not common:
        raise ValueError(f"Sem modelos em comum: filtered {models_f} vs baseline {models_b}.")

    base_small = baseline[["text_id"] + [f"{m}_consensus" for m in common]].rename(
        columns={f"{m}_consensus": f"{m}__baseline" for m in common}
    ).drop_duplicates(subset=["text_id"])

    filt_small = filtered[["text_id", "ground_truth", "fold"] + [f"{m}_consensus" for m in common]].rename(
        columns={f"{m}_consensus": f"{m}__filtered" for m in common}
    )

    merged = filt_small.merge(base_small, on="text_id", how="left")

    rows = []
    for fold, g in merged.groupby("fold"):
        n_holdout = len(g)
        gt = coerce_int(g["ground_truth"])
        for m in common:
            covered = g[f"{m}__baseline"].notna()
            fp = coerce_int(g[f"{m}__filtered"])
            bp = coerce_int(g[f"{m}__baseline"])

            # Comparação PAREADA: as MESMAS linhas para os dois lados — GT,
            # filtered e baseline todos válidos (≠ -1). Isola o efeito do espaço
            # de classes, sem viés de -1 (ex.: overflow de contexto) diferente
            # entre as condições.
            valid = covered & (gt != INVALID) & (fp != INVALID) & (bp != INVALID)
            f_acc, f_f1 = _acc_f1(gt[valid], fp[valid])
            b_acc, b_f1 = _acc_f1(gt[valid], bp[valid])

            # Taxas de inválido (sobre o held-out coberto) — mostram se as
            # condições diferem em quantos -1 produziram.
            cov_n = int(covered.sum())
            filt_inv = float((fp[covered] == INVALID).mean()) if cov_n else 0.0
            base_inv = float((bp[covered] == INVALID).mean()) if cov_n else 0.0

            rows.append({
                "dataset": dataset,
                "fold": int(fold),
                "model": m,
                "n_holdout": n_holdout,
                "n_paired": int(valid.sum()),
                "filtered_acc": f_acc,
                "baseline_acc": b_acc,
                "delta_acc": f_acc - b_acc,
                "filtered_f1": f_f1,
                "baseline_f1": b_f1,
                "delta_f1": f_f1 - b_f1,
                "filtered_invalid_rate": filt_inv,
                "baseline_invalid_rate": base_inv,
                "join_coverage": float(cov_n / n_holdout) if n_holdout else 0.0,
            })
    return pd.DataFrame(rows), common


def load_consensus_filtered(run_dir: Path) -> pd.DataFrame | None:
    """Concatena os `fold_*/filtered/consensus/dataset_consenso.csv`, marcando o fold."""
    frames = []
    for f in sorted(run_dir.glob("fold_*/filtered/consensus/dataset_consenso.csv")):
        df = pd.read_csv(f)
        df["fold"] = int(f.parts[-4].split("_")[1])  # .../fold_N/filtered/consensus/...
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def compare_consensus_per_fold(dataset: str, run_dir: Path, baseline_dir: Path) -> pd.DataFrame | None:
    """
    Compara o CONSENSO (`resolved_annotation`) filtered vs baseline, por fold.
    Retorna None se os datasets de consenso ainda não existirem (rode
    `scripts.run_consensus_two_phase` antes). Mesma schema de `compare_per_fold`,
    com `model="consensus"`.
    """
    filt = load_consensus_filtered(run_dir)
    base_path = baseline_dir / "consensus" / "dataset_consenso.csv"
    if filt is None or not base_path.exists():
        return None

    base = pd.read_csv(base_path)[["text_id", "resolved_annotation"]].rename(
        columns={"resolved_annotation": "resolved_baseline"}
    ).drop_duplicates(subset=["text_id"])

    filt = filt[["text_id", "ground_truth", "fold", "resolved_annotation"]].rename(
        columns={"resolved_annotation": "resolved_filtered"}
    )
    merged = filt.merge(base, on="text_id", how="left")

    rows = []
    for fold, g in merged.groupby("fold"):
        n_holdout = len(g)
        gt = coerce_int(g["ground_truth"])
        fp = coerce_int(g["resolved_filtered"])
        bp = coerce_int(g["resolved_baseline"])
        covered = g["resolved_baseline"].notna()
        valid = covered & (gt != INVALID) & (fp != INVALID) & (bp != INVALID)
        f_acc, f_f1 = _acc_f1(gt[valid], fp[valid])
        b_acc, b_f1 = _acc_f1(gt[valid], bp[valid])
        cov_n = int(covered.sum())
        rows.append({
            "dataset": dataset,
            "fold": int(fold),
            "model": "consensus",
            "n_holdout": n_holdout,
            "n_paired": int(valid.sum()),
            "filtered_acc": f_acc,
            "baseline_acc": b_acc,
            "delta_acc": f_acc - b_acc,
            "filtered_f1": f_f1,
            "baseline_f1": b_f1,
            "delta_f1": f_f1 - b_f1,
            "filtered_invalid_rate": float((fp[covered] == INVALID).mean()) if cov_n else 0.0,
            "baseline_invalid_rate": float((bp[covered] == INVALID).mean()) if cov_n else 0.0,
            "join_coverage": float(cov_n / n_holdout) if n_holdout else 0.0,
        })
    return pd.DataFrame(rows)


def aggregate_by_mean(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Agrega as métricas por-fold pela MÉDIA (± desvio) entre folds, por modelo."""
    g = per_fold.groupby(["dataset", "model"])
    agg = g.agg(
        folds=("fold", "nunique"),
        n_paired_total=("n_paired", "sum"),
        filtered_acc_mean=("filtered_acc", "mean"),
        filtered_acc_std=("filtered_acc", "std"),
        baseline_acc_mean=("baseline_acc", "mean"),
        baseline_acc_std=("baseline_acc", "std"),
        delta_acc_mean=("delta_acc", "mean"),
        delta_acc_std=("delta_acc", "std"),
        filtered_f1_mean=("filtered_f1", "mean"),
        filtered_f1_std=("filtered_f1", "std"),
        baseline_f1_mean=("baseline_f1", "mean"),
        baseline_f1_std=("baseline_f1", "std"),
        delta_f1_mean=("delta_f1", "mean"),
        delta_f1_std=("delta_f1", "std"),
        filtered_invalid_rate_mean=("filtered_invalid_rate", "mean"),
        baseline_invalid_rate_mean=("baseline_invalid_rate", "mean"),
    ).reset_index()
    return agg


# ----------------------------------------------------------------------
# Relatório
# ----------------------------------------------------------------------
def _md_table(df: pd.DataFrame) -> str:
    """Renderiza um DataFrame como tabela Markdown (sem depender de tabulate)."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _confusion_md(y_true, y_pred, labels, names) -> str:
    """Matriz de confusão normalizada por linha (fração da classe verdadeira), em Markdown."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rowsum = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, rowsum, out=np.zeros_like(cm, dtype=float), where=rowsum != 0)

    def lab(l):
        return f"{l}:{str(names.get(str(l), l))[:10]}"

    head = ["true \\ pred"] + [lab(l) for l in labels]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for i, l in enumerate(labels):
        cells = [lab(l)] + [f"{frac[i, j]:.2f}" for j in range(len(labels))]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_confusion_section(run_dir: Path, dataset: str, models: list[str]) -> str:
    """Matrizes de confusão das predições FILTERED (por modelo + consenso)."""
    names = LABEL_MEANINGS.get(dataset, {})
    filtered = load_filtered(run_dir).drop_duplicates("text_id")
    gt = coerce_int(filtered["ground_truth"])
    labels = sorted(x for x in set(gt.tolist()) if x != INVALID)
    if not labels:
        return ""

    cons = load_consensus_filtered(run_dir)
    L = ["## Matriz de confusão — filtered\n",
         "Linha = classe verdadeira, coluna = predita; valores = fração da linha "
         "(diagonal = acerto). Fora da diagonal alto = confusão sistemática entre gêneros.\n"]
    for m in models:
        if m == "consensus":
            if cons is None:
                continue
            c = cons.drop_duplicates("text_id")
            yt, yp = coerce_int(c["ground_truth"]), coerce_int(c["resolved_annotation"])
        else:
            yt, yp = gt, coerce_int(filtered[f"{m}_consensus"])
        valid = (yt != INVALID) & (yp != INVALID)
        L.append(f"### {m}\n")
        L.append(_confusion_md(yt[valid], yp[valid], labels, names))
        L.append("")
    return "\n".join(L)


def build_report(dataset, run_dir, baseline_dir, models, per_fold, agg, ceiling) -> str:
    L = []
    L.append(f"# Comparação Filtered (2 fases) vs Baseline — `{dataset}`\n")
    L.append(f"- **Run 2 fases:** `{run_dir}`")
    L.append(f"- **Baseline:** `{baseline_dir}`")
    L.append(f"- **Modelos comparados:** {', '.join(models)}")
    L.append(f"- **Folds:** {per_fold['fold'].nunique()}")
    L.append("- **Métrica:** comparação PAREADA — filtered e baseline medidos exatamente nas "
             "mesmas linhas (GT, filtered e baseline todos válidos, ≠ -1), cruzadas por `text_id`. "
             "Isola só a redução do espaço de classes. `*_invalid_rate` mostra quantos -1 cada "
             "condição produziu (ex.: overflow de contexto).\n")

    if ceiling is not None:
        L.append("## Teto da Fase 2 — recall@k (média entre folds)\n")
        L.append("> **macro** = teto do f1-macro (métrica primária, cada classe pesa igual); "
                 "**micro** = teto da accuracy (ponderado por instância, favorece majoritárias).\n")
        L.append(ceiling.round(4).pipe(_md_table))
        L.append("")

    L.append("## Agregado (média entre folds) — métrica primária: f1-macro\n")
    L.append("> Datasets desbalanceados → **f1-macro** é a métrica de referência; accuracy vem como apoio.\n")
    show = agg.sort_values("delta_f1_mean", ascending=False).copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(4)
    cols = ["model", "folds", "filtered_f1_mean", "baseline_f1_mean", "delta_f1_mean",
            "delta_f1_std", "filtered_acc_mean", "baseline_acc_mean", "delta_acc_mean",
            "filtered_invalid_rate_mean", "baseline_invalid_rate_mean"]
    L.append(show[cols].pipe(_md_table))
    L.append("")
    L.append("> `delta = filtered - baseline`. Negativo = filtered perdeu; positivo = filtered ganhou.\n")

    # Matrizes de confusão (filtered) — onde o erro se concentra.
    L.append(build_confusion_section(run_dir, dataset, models))

    L.append("## Por fold (f1-macro primeiro)\n")
    for m in models:
        L.append(f"### {m}\n")
        sub = per_fold[per_fold["model"] == m].sort_values("fold")
        cols = ["fold", "n_paired", "filtered_f1", "baseline_f1", "delta_f1",
                "filtered_acc", "baseline_acc", "delta_acc"]
        L.append(sub[cols].round(4).pipe(_md_table))
        L.append("")

    return "\n".join(L)


def discover_datasets(results_root: Path) -> list[Path]:
    return sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and (d / "two_phase").is_dir()
    )


def main():
    ap = argparse.ArgumentParser(description="Compara filtered (2 fases) vs baseline por fold e agregado.")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--datasets", default=None, help="Lista separada por vírgula. Default: auto-descobre.")
    ap.add_argument("--run", default=None, help="Run de 2 fases específico (só com 1 dataset).")
    ap.add_argument("--baseline", default=None, help="Dir ou CSV de baseline específico (só com 1 dataset).")
    args = ap.parse_args()

    root = Path(args.results_root)
    dataset_dirs = (
        [root / d.strip() for d in args.datasets.split(",")]
        if args.datasets else discover_datasets(root)
    )
    if not dataset_dirs:
        print(f"Nenhum dataset com run de 2 fases em {root}.")
        return

    all_per_fold, all_agg = [], []
    for ds_dir in dataset_dirs:
        if not ds_dir.is_dir():
            print(f"[skip] {ds_dir} não existe.")
            continue

        run_dir = Path(args.run) if args.run else latest_two_phase_run(ds_dir)
        if run_dir is None or not run_dir.is_dir():
            print(f"[skip] {ds_dir.name}: sem run de 2 fases.")
            continue

        if args.baseline:
            bpath = Path(args.baseline)
            baseline_dir = bpath.parent if bpath.suffix == ".csv" else bpath
        else:
            baseline_dir = latest_baseline_dir(ds_dir)
        if baseline_dir is None:
            print(f"[skip] {ds_dir.name}: sem baseline com annotations.csv.")
            continue

        print(f"\n=== {ds_dir.name} ===")
        print(f"  run 2 fases: {run_dir}")
        print(f"  baseline   : {baseline_dir}")

        per_fold, models = compare_per_fold(ds_dir.name, run_dir, baseline_dir)

        # Consenso (resolved_annotation) entra como um "modelo" a mais, se já
        # calculado (scripts.run_consensus_two_phase). Aparece no relatório junto.
        per_fold_cons = compare_consensus_per_fold(ds_dir.name, run_dir, baseline_dir)
        if per_fold_cons is not None:
            per_fold = pd.concat([per_fold, per_fold_cons], ignore_index=True)
            models = models + ["consensus"]
            print("  (consenso incluído)")
        else:
            print("  (consenso ausente — rode scripts.run_consensus_two_phase para incluí-lo)")

        agg = aggregate_by_mean(per_fold)
        ceiling = load_recall_ceiling(run_dir)

        # persiste os artefatos DENTRO do run de 2 fases
        per_fold.round(6).to_csv(run_dir / "comparison_per_fold.csv", index=False)
        agg.round(6).to_csv(run_dir / "comparison_aggregated.csv", index=False)
        report = build_report(ds_dir.name, run_dir, baseline_dir, models, per_fold, agg, ceiling)
        (run_dir / "comparison_report.md").write_text(report, encoding="utf-8")

        # eco no terminal (agregado, f1-macro primeiro — datasets desbalanceados)
        show = agg.sort_values("delta_f1_mean", ascending=False).copy()
        for c in show.columns:
            if show[c].dtype.kind == "f":
                show[c] = show[c].round(4)
        print(show[["model", "folds", "filtered_f1_mean", "baseline_f1_mean", "delta_f1_mean",
                    "filtered_acc_mean", "baseline_acc_mean", "delta_acc_mean"]].to_string(index=False))
        print(f"  -> {run_dir / 'comparison_report.md'}")
        print(f"  -> {run_dir / 'comparison_per_fold.csv'}")
        print(f"  -> {run_dir / 'comparison_aggregated.csv'}")

        all_per_fold.append(per_fold)
        all_agg.append(agg)

    if all_per_fold:
        pd.concat(all_per_fold, ignore_index=True).round(6).to_csv(
            root / "comparison_per_fold.csv", index=False)
        pd.concat(all_agg, ignore_index=True).round(6).to_csv(
            root / "comparison_aggregated.csv", index=False)
        print(f"\n[OK] Consolidado em {root / 'comparison_aggregated.csv'}")


if __name__ == "__main__":
    main()

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
  1. Junta todos os `fold_*/filtered/annotations.csv` do run (held-out anotados).
  2. Pega o ÚLTIMO diretório de baseline do dataset que tenha `annotations.csv`
     (ex.: `data/results/books/2026-05-26_18-26-27/annotations.csv`).
  3. Cruza por `text_id` e calcula accuracy / f1_macro (filtered vs baseline) por
     modelo, nas mesmas linhas, tratando -1 como inválido (igual ao evaluate).
  4. Mostra o recall@k agregado (teto da Fase 2) como referência.

Uso
---
    python -m scripts.compare_filtered_vs_baseline
    python -m scripts.compare_filtered_vs_baseline --datasets books,agnews
    python -m scripts.compare_filtered_vs_baseline --results-root data/results
    python -m scripts.compare_filtered_vs_baseline --run <path/two_phase/<ts>> --baseline <dir_ou_csv>
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

INVALID = -1


# ----------------------------------------------------------------------
# Descoberta de diretórios
# ----------------------------------------------------------------------
def latest_two_phase_run(dataset_dir: Path) -> Path | None:
    """
    Último run de 2 fases (<dataset>/two_phase/<timestamp>/) que contenha ao menos
    um `fold_*/filtered/annotations.csv`. Pula runs vazios/interrompidos — mesma
    guarda do 'latest results folder gotcha'.
    """
    tp = dataset_dir / "two_phase"
    if not tp.is_dir():
        return None
    runs = sorted([d for d in tp.iterdir() if d.is_dir()], reverse=True)
    for run in runs:
        if any(run.glob("fold_*/filtered/annotations.csv")):
            return run
    return None


def latest_baseline_dir(dataset_dir: Path) -> Path | None:
    """
    Último diretório de baseline do dataset que contenha `annotations.csv`.

    Ordena por nome desc (os dirs começam com data ISO → ordem cronológica) e
    pula os incompletos — guarda contra o 'latest results folder gotcha'.
    """
    candidates = sorted(
        [d for d in dataset_dir.iterdir() if d.is_dir() and d.name != "two_phase"],
        reverse=True,
    )
    for d in candidates:
        if (d / "annotations.csv").exists():
            return d
    return None


# ----------------------------------------------------------------------
# Carregamento / métricas
# ----------------------------------------------------------------------
def consensus_models(df: pd.DataFrame) -> list[str]:
    """Modelos presentes via colunas `{model}_consensus`."""
    return [c[: -len("_consensus")] for c in df.columns if c.endswith("_consensus")]


def coerce_int(series: pd.Series) -> pd.Series:
    """Normaliza rótulos: ERROR/None/''/N/A → -1, depois int (igual ao evaluate)."""
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
        df["fold"] = int(f.parts[-3].split("_")[1])  # .../fold_N/filtered/annotations.csv
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum fold_*/filtered/annotations.csv em {run_dir}")
    return pd.concat(frames, ignore_index=True)


def scores(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """accuracy e f1_macro nas linhas válidas (y_true != -1 e y_pred != -1)."""
    yt, yp = coerce_int(y_true), coerce_int(y_pred)
    mask = (yt != INVALID) & (yp != INVALID)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "n_valid": 0}
    return {
        "accuracy": accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro"),
        "n_valid": int(len(yt)),
    }


def load_recall_ceiling(run_dir: Path) -> pd.DataFrame | None:
    p = run_dir / "recall_at_k_aggregated.csv"
    if p.exists():
        return pd.read_csv(p)
    # fallback: agrega o por-fold se o agregado não existir (run interrompido)
    p2 = run_dir / "recall_at_k_all_folds.csv"
    if p2.exists():
        df = pd.read_csv(p2)
        return df.groupby("k")["recall_at_k"].agg(["mean", "std", "count"]).reset_index()
    return None


# ----------------------------------------------------------------------
# Comparação por dataset
# ----------------------------------------------------------------------
def compare_dataset(dataset_dir: Path, run_dir: Path, baseline_dir: Path) -> pd.DataFrame:
    filtered = load_filtered(run_dir)
    baseline = pd.read_csv(baseline_dir / "annotations.csv")

    models_f = consensus_models(filtered)
    models_b = consensus_models(baseline)
    common = [m for m in models_f if m in models_b]
    if not common:
        raise ValueError(
            f"Sem modelos em comum entre filtered {models_f} e baseline {models_b}."
        )

    # baseline: só text_id + consensus (renomeado para não colidir)
    base_small = baseline[["text_id"] + [f"{m}_consensus" for m in common]].rename(
        columns={f"{m}_consensus": f"{m}__baseline" for m in common}
    ).drop_duplicates(subset=["text_id"])

    filt_small = filtered[["text_id", "ground_truth", "fold"] + [f"{m}_consensus" for m in common]].rename(
        columns={f"{m}_consensus": f"{m}__filtered" for m in common}
    )

    merged = filt_small.merge(base_small, on="text_id", how="left")

    # cobertura do join: held-out encontrados no baseline
    any_base_col = f"{common[0]}__baseline"
    covered = merged[any_base_col].notna()
    coverage = float(covered.mean()) if len(merged) else 0.0

    rows = []
    for m in common:
        sub = merged[merged[f"{m}__baseline"].notna()]
        sf = scores(sub["ground_truth"], sub[f"{m}__filtered"])
        sb = scores(sub["ground_truth"], sub[f"{m}__baseline"])
        rows.append({
            "dataset": dataset_dir.name,
            "model": m,
            "n_pairs": int(len(sub)),
            "filtered_acc": round(sf["accuracy"], 4),
            "baseline_acc": round(sb["accuracy"], 4),
            "delta_acc": round(sf["accuracy"] - sb["accuracy"], 4),
            "filtered_f1": round(sf["f1_macro"], 4),
            "baseline_f1": round(sb["f1_macro"], 4),
            "delta_f1": round(sf["f1_macro"] - sb["f1_macro"], 4),
            "join_coverage": round(coverage, 4),
        })
    return pd.DataFrame(rows)


def discover_datasets(results_root: Path) -> list[Path]:
    """Datasets que têm um run de 2 fases."""
    return sorted(
        d for d in results_root.iterdir()
        if d.is_dir() and (d / "two_phase").is_dir()
    )


def main():
    ap = argparse.ArgumentParser(description="Compara filtered (2 fases) vs baseline por text_id.")
    ap.add_argument("--results-root", default="data/results", help="Raiz dos resultados.")
    ap.add_argument("--datasets", default=None, help="Lista separada por vírgula. Default: auto-descobre.")
    ap.add_argument("--run", default=None, help="Run de 2 fases específico (sobrepõe auto). Só com 1 dataset.")
    ap.add_argument("--baseline", default=None, help="Dir ou CSV de baseline específico. Só com 1 dataset.")
    ap.add_argument("--out", default=None, help="CSV de saída consolidado. Default: <results-root>/comparison_filtered_vs_baseline.csv")
    args = ap.parse_args()

    root = Path(args.results_root)

    if args.datasets:
        dataset_dirs = [root / d.strip() for d in args.datasets.split(",")]
    else:
        dataset_dirs = discover_datasets(root)

    if not dataset_dirs:
        print(f"Nenhum dataset com run de 2 fases em {root}.")
        return

    all_rows = []
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
            print(f"[skip] {ds_dir.name}: sem diretório de baseline com annotations.csv.")
            continue

        print(f"\n=== {ds_dir.name} ===")
        print(f"  filtered (2 fases): {run_dir}")
        print(f"  baseline          : {baseline_dir}")

        df = compare_dataset(ds_dir, run_dir, baseline_dir)
        all_rows.append(df)

        ceiling = load_recall_ceiling(run_dir)
        if ceiling is not None:
            print("  recall@k (teto Fase 2, média entre folds):")
            print("   ", ceiling.to_string(index=False).replace("\n", "\n    "))

        cov = df["join_coverage"].iloc[0] if len(df) else 0.0
        if cov < 0.999:
            print(f"  [AVISO] cobertura do join = {cov:.1%} — nem todo held-out foi achado no baseline "
                  f"(text_ids ausentes ficam de fora da comparacao).")
        print(df.to_string(index=False))

    if not all_rows:
        print("\nNada comparado.")
        return

    consolidated = pd.concat(all_rows, ignore_index=True)
    out = Path(args.out) if args.out else root / "comparison_filtered_vs_baseline.csv"
    consolidated.to_csv(out, index=False)
    print(f"\n[OK] Comparacao consolidada salva em: {out}")


if __name__ == "__main__":
    main()

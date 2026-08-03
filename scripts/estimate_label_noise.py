"""
Estima um LIMITE INFERIOR do ruído de rótulo (ground-truth) por classe.

Sinal: concordância UNÂNIME dos modelos independentes contra o GT. Usa as
anotações do BASELINE (espaço completo de classes) — ali cada modelo viu TODAS
as classes, inclusive a do GT, e ainda assim os N modelos escolheram, por
unanimidade, a MESMA outra classe. A chance de N modelos de arquiteturas
diferentes errarem juntos, na mesma classe, é baixa → forte indício de que o
rótulo é que está errado.

É um LIMITE INFERIOR: casos de rótulo errado onde os modelos discordam entre si
não são contados.

Uso:
    python -m scripts.estimate_label_noise --dataset books
    python -m scripts.estimate_label_noise --dataset books --baseline <dir>
    python -m scripts.estimate_label_noise --dataset books --show history:romance:15
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.compare_filtered_vs_baseline import (
    latest_baseline_dir,
    consensus_models,
    coerce_int,
    LABEL_MEANINGS,
)


def main():
    ap = argparse.ArgumentParser(description="Estima limite inferior do ruído de rótulo via unanimidade dos modelos.")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline", default=None, help="Dir de baseline (default: último completo).")
    ap.add_argument("--show", default=None, help="Ex.: 'history:romance:15' — imprime 15 textos GT=history unânime=romance.")
    args = ap.parse_args()

    ds_dir = Path(args.results_root) / args.dataset
    baseline_dir = Path(args.baseline) if args.baseline else latest_baseline_dir(ds_dir)
    if baseline_dir is None:
        print(f"Sem baseline com annotations.csv em {ds_dir}.")
        return

    names = LABEL_MEANINGS.get(args.dataset, {})
    name = lambda i: names.get(str(i), str(i))

    df = pd.read_csv(baseline_dir / "annotations.csv").drop_duplicates("text_id")
    models = consensus_models(df)
    cons_cols = [f"{m}_consensus" for m in models]
    for c in cons_cols + ["ground_truth"]:
        df[c] = coerce_int(df[c])

    print(f"Baseline: {baseline_dir}")
    print(f"Modelos ({len(models)}): {models} | textos: {len(df)}\n")

    # unânime = todos os modelos concordam na mesma classe (!= -1)
    preds = df[cons_cols].to_numpy()
    unanime_cls, is_unanime = [], []
    for row in preds:
        s = set(int(x) for x in row)
        if len(s) == 1 and -1 not in s:
            is_unanime.append(True); unanime_cls.append(row[0])
        else:
            is_unanime.append(False); unanime_cls.append(-1)
    df["_unanime"] = is_unanime
    df["_unanime_cls"] = [int(x) for x in unanime_cls]

    gt = df["ground_truth"]
    # discordância unânime = unânime numa classe != GT (e GT válido)
    disagree = df["_unanime"] & (df["_unanime_cls"] != gt) & (gt != -1)

    rows = []
    for c in sorted(x for x in set(gt.tolist()) if x != -1):
        m = gt == c
        n = int(m.sum())
        dis = int((disagree & m).sum())
        top = Counter(df.loc[disagree & m, "_unanime_cls"]).most_common(1)
        top_txt = f"{name(top[0][0])} ({top[0][1]})" if top else "-"
        rows.append({
            "classe": f"{c}:{name(c)}",
            "n": n,
            "ruido_min": dis,
            "taxa_min": round(dis / n, 4) if n else 0.0,
            "principal_alvo_unanime": top_txt,
        })
    out = pd.DataFrame(rows).sort_values("taxa_min", ascending=False)
    print("Ruído de rótulo (LIMITE INFERIOR) por classe — via unanimidade dos modelos:\n")
    print(out.to_string(index=False))
    overall = int(disagree.sum())
    print(f"\nGeral: {overall}/{int((gt!=-1).sum())} = "
          f"{overall/max(1,int((gt!=-1).sum())):.2%} dos textos têm os {len(models)} modelos "
          f"UNÂNIMES numa classe != GT (limite inferior do ruído global).")

    if args.show:
        parts = args.show.split(":")
        tgt_gt = next(i for i, v in names.items() if parts[0].lower() in v.lower())
        tgt_pred = next(i for i, v in names.items() if parts[1].lower() in v.lower())
        k = int(parts[2]) if len(parts) > 2 else 10
        sub = df[(gt == int(tgt_gt)) & df["_unanime"] & (df["_unanime_cls"] == int(tgt_pred))]
        print(f"\n{len(sub)} casos GT={name(int(tgt_gt))} com os modelos UNÂNIMES em {name(int(tgt_pred))}. Amostra:\n")
        for i, (_, r) in enumerate(sub.head(k).iterrows()):
            print(f"[{i+1}] {str(r['text'])[:260]}".replace("\n", " "))


if __name__ == "__main__":
    main()

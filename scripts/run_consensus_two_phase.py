"""
Roda o CONSENSO (resolved_annotation, combinando os modelos) para um run de 2 fases.

Aplica o `ConsensusPipeline` existente em:
  1. cada `fold_*/filtered/annotations.csv` do run de 2 fases, e
  2. o diretório de baseline correspondente (o consenso do baseline completo).

O `ConsensusPipeline` espera as anotações em `<results_dir>/<dataset_name>/<date>/
annotations.csv` e escreve `consensus/dataset_consenso.csv` (com `resolved_annotation`)
lá dentro. Mapeamos os componentes de caminho para reusá-lo sem alterar nada:

  fold : results_dir=<run_dir>, dataset_name="fold_N", date="filtered"
         → <run_dir>/fold_N/filtered/
  base : results_dir=<results_root>, dataset_name=<dataset>, date=<baseline_ts>
         → <results_root>/<dataset>/<baseline_ts>/

Uso
---
    python -m scripts.run_consensus_two_phase --datasets books
    python -m scripts.run_consensus_two_phase --datasets books --force
    python -m scripts.run_consensus_two_phase --datasets books --skip-baseline
"""
import argparse
from pathlib import Path

from loguru import logger

from src.systems.llm_annotation_system.consensus.pipeline import (
    ConsensusConfig,
    ConsensusPipeline,
)
from scripts.compare_filtered_vs_baseline import (
    latest_two_phase_run,
    latest_baseline_dir,
    discover_datasets,
)


def run_consensus(results_dir: Path, dataset_name: str, date: str, force: bool) -> bool:
    """Roda o consenso num diretório <results_dir>/<dataset_name>/<date>. Idempotente."""
    target = Path(results_dir) / dataset_name / date
    consenso = ConsensusPipeline.dataset_path(target)
    if consenso.exists() and not force:
        logger.info(f"[skip] consenso já existe: {consenso} (use --force para refazer)")
        return False
    if not (target / "annotations.csv").exists():
        logger.warning(f"[skip] sem annotations.csv em {target}")
        return False

    cfg = ConsensusConfig(
        dataset_name=dataset_name,
        results_dir=str(results_dir),
        specific_date=date,
    )
    ConsensusPipeline(cfg).run()
    return True


def main():
    ap = argparse.ArgumentParser(description="Roda consenso por fold (filtered) + baseline de um run de 2 fases.")
    ap.add_argument("--results-root", default="data/results")
    ap.add_argument("--datasets", default=None, help="Lista separada por vírgula. Default: auto-descobre.")
    ap.add_argument("--run", default=None, help="Run de 2 fases específico (só com 1 dataset).")
    ap.add_argument("--baseline", default=None, help="Dir de baseline específico (só com 1 dataset).")
    ap.add_argument("--skip-baseline", action="store_true", help="Não roda o consenso do baseline.")
    ap.add_argument("--force", action="store_true", help="Refaz o consenso mesmo se já existir.")
    args = ap.parse_args()

    root = Path(args.results_root)
    dataset_dirs = (
        [root / d.strip() for d in args.datasets.split(",")]
        if args.datasets else discover_datasets(root)
    )
    if not dataset_dirs:
        print(f"Nenhum dataset com run de 2 fases em {root}.")
        return

    for ds_dir in dataset_dirs:
        run_dir = Path(args.run) if args.run else latest_two_phase_run(ds_dir)
        if run_dir is None:
            print(f"[skip] {ds_dir.name}: sem run de 2 fases.")
            continue

        print(f"\n=== {ds_dir.name} ===")
        print(f"  run 2 fases: {run_dir}")

        # 1) Consenso por fold (filtered)
        folds = sorted(run_dir.glob("fold_*/filtered/annotations.csv"))
        for f in folds:
            fold_name = f.parts[-3]  # fold_N
            run_consensus(run_dir, fold_name, "filtered", args.force)

        # 2) Consenso do baseline
        if not args.skip_baseline:
            baseline_dir = Path(args.baseline) if args.baseline else latest_baseline_dir(ds_dir)
            if baseline_dir is None:
                print(f"  [aviso] sem baseline com annotations.csv — pulando consenso do baseline.")
            else:
                print(f"  baseline: {baseline_dir}")
                # baseline_dir == <root>/<dataset>/<date> → results_dir é a raiz.
                run_consensus(root, ds_dir.name, baseline_dir.name, args.force)

    print("\n[OK] Consenso concluído.")


if __name__ == "__main__":
    main()

from pathlib import Path

def get_latest_results_date(results_dir: str, dataset_name: str) -> str:
    results_dir = Path(results_dir)
    results_dataset_path = results_dir.joinpath(dataset_name)
    # Só pastas de anotação: ignora arquivos (intermediate.csv), 2 fases e checkpoints
    candidates = [
        p for p in results_dataset_path.iterdir()
        if p.is_dir() and not p.name.startswith(("two_phase", "_"))
    ]
    latest_date = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest_date.name

from pathlib import Path

def get_latest_results_date(results_dir: str, dataset_name: str) -> str:
    results_dir = Path(results_dir)
    results_dataset_path = results_dir.joinpath(dataset_name)
    latest_date = max(results_dataset_path.iterdir(), key=lambda p: p.stat().st_mtime)
    return latest_date.name
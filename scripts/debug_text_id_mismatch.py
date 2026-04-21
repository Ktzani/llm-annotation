"""Diagnóstico do mismatch de text_id entre CSV anotado e parquet do HF."""
import hashlib
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src.utils.get_latest_results_date import get_latest_results_date

DATASET_NAME = "agnews"
HF_REPO = "waashk/agnews"
HF_FILE = "train_fold_0.parquet"
RESULTS_DIR = Path(r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\results")
CACHE_DIR = Path(r"C:\Users\gabri\Documents\GitHub\llm-annotation\data\.cache\hf")


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def main():
    date = get_latest_results_date(str(RESULTS_DIR), DATASET_NAME)
    csv_path = RESULTS_DIR / DATASET_NAME / date / "summary" / "dataset_anotado_completo.csv"
    df_csv = pd.read_csv(csv_path)

    hf_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename=HF_FILE, cache_dir=str(CACHE_DIR),
    )
    df_hf = pd.read_parquet(hf_path)

    print(f"CSV linhas: {len(df_csv)}, HF linhas: {len(df_hf)}")

    # Analise 1: todos os textos do CSV terminam com \n?
    ends_newline = df_csv["text"].str.endswith("\n", na=False).sum()
    print(f"CSV textos terminando com \\n: {ends_newline}/{len(df_csv)}")
    starts_newline = df_csv["text"].str.startswith("\n", na=False).sum()
    print(f"CSV textos comecando com \\n: {starts_newline}/{len(df_csv)}")

    ends_newline_hf = df_hf["text"].str.endswith("\n", na=False).sum()
    print(f"HF  textos terminando com \\n: {ends_newline_hf}/{len(df_hf)}")

    # Analise 2: se tirar strip do CSV, quantos batem com HF?
    df_csv["_text_stripped"] = df_csv["text"].str.strip()
    df_csv["_hash_stripped"] = df_csv["_text_stripped"].apply(md5)
    df_hf["_hash"] = df_hf["text"].apply(md5)

    inter = set(df_csv["_hash_stripped"]) & set(df_hf["_hash"])
    print(
        f"\nApos strip() no CSV: {len(inter)} IDs em comum "
        f"({100*len(inter)/len(df_hf):.2f}% do HF)"
    )

    # Analise 3: text_id armazenado == hash do text atual?
    df_csv["_hash_stored"] = df_csv["text"].apply(md5)
    match_stored = (df_csv["_hash_stored"] == df_csv["text_id"]).sum()
    print(
        f"text_id armazenado confere com md5(text atual): "
        f"{match_stored}/{len(df_csv)}"
    )


if __name__ == "__main__":
    main()

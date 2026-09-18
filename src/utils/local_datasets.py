"""
Datasets locais / proprietários (fora do HuggingFace Hub)

Um dataset local é registrado por um manifesto `dataset.json` dentro de uma
pasta em `LOCAL_DATASETS_DIR` (default: `<projeto>/data/datasets`, que fica fora
do git e é montada no Docker em `/app/data/datasets`). O nome da pasta é o
`dataset_name` usado nos experimentos (anotação, consenso, fine-tuning):

    data/datasets/
      meu_dataset/
        dataset.json          ← manifesto
        data.csv              ← textos a anotar (hf_file/split resolvem aqui)
        train_fold_0.csv      ← (opcional) folds p/ fine-tuning e 2 fases
        test_fold_0.csv

Manifesto (`dataset.json`):

    {
      "path": "data.csv",              # opcional: arquivo ou pasta, relativo ao manifesto (default: a pasta)
      "text_column": "texto",
      "label_column": "classe",        # ou null se não houver ground truth
      "label_meanings": {"0": "financeiro", "1": "suporte técnico"},
      "prompt": "Customer ticket",     # descrição do tipo de texto usada no prompt
      "description": "Tickets internos de suporte",
      "read_kwargs": {"sep": ";"}      # opcional: repassado ao pandas.read_csv (CSV/TSV)
    }

Os rótulos podem vir como inteiros (0, 1, ...) ou como os nomes presentes em
`label_meanings` ("financeiro", ...): são convertidos para os códigos inteiros
usados pelo resto do framework.

Datasets locais também podem ser declarados direto em `DATASETS`
(`src/config/datasets_collected.py`) com `"source": "local"` e `"path"` absoluto,
junto de uma entrada em `LABEL_MEANINGS`.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from src.config.datasets_collected import DATASETS, LABEL_MEANINGS

MANIFEST_NAME = "dataset.json"
DEFAULT_DATA_STEM = "data"

# Ordem de preferência ao resolver um arquivo pelo nome sem extensão.
SUPPORTED_EXTENSIONS = (".parquet", ".csv", ".tsv", ".jsonl", ".json", ".xlsx", ".xls")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def local_datasets_dir() -> Path:
    """Pasta raiz dos datasets locais (sobrescrevível por `LOCAL_DATASETS_DIR`)."""
    return Path(os.getenv("LOCAL_DATASETS_DIR", _PROJECT_ROOT / "data" / "datasets"))


# =============================================================================
# REGISTRO
# =============================================================================

def refresh_local_datasets() -> List[str]:
    """
    Registra (ou atualiza) em `DATASETS` / `LABEL_MEANINGS` os datasets locais
    descritos por manifestos. Idempotente e barato: chamado antes de cada
    consulta ao registro, então pastas novas valem sem reiniciar a API.

    Um manifesto inválido é ignorado com erro no log, sem afetar os demais.
    Não sobrescreve datasets declarados em código (ex.: os do HF).

    Returns:
        Nomes dos datasets locais registrados.
    """
    root = local_datasets_dir()
    if not root.is_dir():
        return []

    registered = []
    for manifest_path in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        name = manifest_path.parent.name

        existing = DATASETS.get(name)
        if existing is not None and "manifest" not in existing:
            _log_once(
                manifest_path, "warning",
                f"Dataset local '{name}' ignorado: já existe um dataset com esse nome "
                f"em datasets_collected.py. Renomeie a pasta {manifest_path.parent}."
            )
            continue

        try:
            spec, label_meanings = _read_manifest(manifest_path)
        except Exception as e:
            _log_once(manifest_path, "error", f"Manifesto inválido em {manifest_path}: {e}")
            continue

        DATASETS[name] = spec
        LABEL_MEANINGS[name] = label_meanings
        registered.append(name)

    return registered


_logged_problems: set = set()


def _log_once(manifest_path: Path, level: str, message: str) -> None:
    """O registro é reescaneado a cada consulta: loga cada problema uma vez por versão do manifesto."""
    key = (str(manifest_path), manifest_path.stat().st_mtime, message)
    if key not in _logged_problems:
        _logged_problems.add(key)
        getattr(logger, level)(message)


def _read_manifest(manifest_path: Path) -> tuple[Dict, Dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError("o manifesto deve ser um objeto JSON")

    if not spec.get("text_column"):
        raise ValueError("campo obrigatório 'text_column' ausente")

    label_meanings = spec.pop("label_meanings", None)
    if not isinstance(label_meanings, dict) or not label_meanings:
        raise ValueError(
            "campo obrigatório 'label_meanings' ausente ou vazio "
            "(ex.: {\"0\": \"classe A\", \"1\": \"classe B\"})"
        )

    for key in label_meanings:
        if not re.fullmatch(r"-?\d+", str(key).strip()):
            raise ValueError(f"chave '{key}' de label_meanings deve ser um inteiro (ex.: \"0\")")

    data_path = Path(spec.get("path", "."))
    if not data_path.is_absolute():
        data_path = manifest_path.parent / data_path

    spec["source"] = "local"
    spec["path"] = str(data_path.resolve())
    spec["manifest"] = str(manifest_path)
    spec.setdefault("label_column", None)
    spec.setdefault("categories", None)

    return spec, {str(k).strip(): v for k, v in label_meanings.items()}


# =============================================================================
# CARREGAMENTO
# =============================================================================

def load_local_dataframe(
    dataset_name: str,
    spec: Dict,
    hf_file: Optional[str] = None,
    split: Optional[str] = None,
    combine_splits: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Carrega um dataset local como DataFrame, espelhando as opções do HF:

    - `hf_file`: arquivo dentro da pasta do dataset (ex.: "train_fold_0.parquet").
    - `combine_splits`: concatena os arquivos cujo nome é o split (train.csv + test.csv).
    - `split`: um único arquivo com esse nome.
    - nenhum: o próprio arquivo de `path`, ou `data.*` / o único arquivo da pasta.

    Nomes são resolvidos pela extensão exata e, se não existir, pelo mesmo nome
    com qualquer extensão suportada — "train_fold_0.parquet" encontra
    "train_fold_0.csv". Assim os folds do fine-tuning/2 fases funcionam sem
    mudar a convenção de nomes.

    Os rótulos são validados e convertidos para os códigos inteiros de
    `LABEL_MEANINGS[dataset_name]`.
    """
    root = Path(spec["path"])
    if not root.exists():
        raise FileNotFoundError(f"Caminho do dataset local '{dataset_name}' não existe: {root}")

    base_dir = root if root.is_dir() else root.parent
    read_kwargs = spec.get("read_kwargs") or {}

    if hf_file:
        files = [_resolve_file(base_dir, hf_file)]
    elif combine_splits:
        files = []
        for sp in combine_splits:
            try:
                files.append(_resolve_file(base_dir, sp))
            except FileNotFoundError as e:
                logger.warning(f"  ⚠️  Split {sp} indisponível ({e})")
        if not files:
            raise ValueError("Nenhum split disponível para combinar")
    elif split:
        files = [_resolve_file(base_dir, split)]
    elif root.is_file():
        files = [root]
    else:
        files = [_find_default_file(base_dir)]

    frames = []
    for file in files:
        df_part = _read_file(file, read_kwargs)
        logger.info(f"  ✓ {file.name}: {len(df_part)} exemplos")
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    text_column = spec["text_column"]
    if text_column not in df.columns:
        raise ValueError(
            f"Coluna de texto '{text_column}' não encontrada.\n"
            f"Colunas disponíveis: {list(df.columns)}"
        )

    # Textos vazios não têm o que anotar e quebrariam o text_id.
    empty = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")
    if empty.any():
        logger.warning(f"Removidas {int(empty.sum())} linhas com texto vazio")
        df = df.loc[~empty].reset_index(drop=True)
    df[text_column] = df[text_column].astype(str)

    label_column = spec.get("label_column")
    if label_column:
        if label_column not in df.columns:
            raise ValueError(
                f"Coluna de rótulo '{label_column}' não encontrada.\n"
                f"Colunas disponíveis: {list(df.columns)}\n"
                f"Use \"label_column\": null se o dataset não tiver ground truth."
            )
        df[label_column] = encode_labels(
            df[label_column], LABEL_MEANINGS.get(dataset_name, {}), label_column
        )

    return df


def encode_labels(labels: pd.Series, label_meanings: Dict[str, str], label_column: str) -> pd.Series:
    """
    Converte rótulos (inteiros, strings numéricas ou nomes de classe) para os
    códigos inteiros de `label_meanings`. Falha com a lista dos valores
    desconhecidos em vez de gerar ground truth silenciosamente errado.
    """
    if labels.isna().any():
        raise ValueError(
            f"{int(labels.isna().sum())} linhas sem rótulo na coluna '{label_column}'. "
            f"Remova-as ou use \"label_column\": null para anotar sem ground truth."
        )

    valid_codes = {int(k) for k in label_meanings}
    name_to_code = {
        str(name).strip().lower(): int(code)
        for code, name in label_meanings.items()
        if name is not None
    }

    def to_code(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and float(value).is_integer():
            return int(value)
        text = str(value).strip()
        if re.fullmatch(r"-?\d+(\.0+)?", text):
            return int(float(text))
        return name_to_code.get(text.lower())

    codes = labels.map(to_code)

    unknown = sorted({str(v) for v, c in zip(labels, codes) if c is None or c not in valid_codes})
    if unknown:
        raise ValueError(
            f"Rótulos da coluna '{label_column}' fora de label_meanings: {unknown[:20]}. "
            f"Classes aceitas: {label_meanings}"
        )

    return codes.astype("int64")


def _resolve_file(base_dir: Path, name: str) -> Path:
    candidate = base_dir / name
    if candidate.is_file() and candidate.name != MANIFEST_NAME:
        return candidate

    stem = Path(name).stem if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS else name
    for ext in SUPPORTED_EXTENSIONS:
        candidate = base_dir / f"{stem}{ext}"
        if candidate.is_file() and candidate.name != MANIFEST_NAME:
            return candidate

    raise FileNotFoundError(
        f"'{name}' não encontrado em {base_dir}. Arquivos disponíveis: {_data_files(base_dir)}"
    )


def _find_default_file(base_dir: Path) -> Path:
    """`data.*` se existir; senão o único arquivo de dados da pasta."""
    try:
        return _resolve_file(base_dir, DEFAULT_DATA_STEM)
    except FileNotFoundError:
        pass

    files = _data_files(base_dir)
    if len(files) == 1:
        return base_dir / files[0]

    raise FileNotFoundError(
        f"Não foi possível escolher o arquivo de dados em {base_dir} ({files or 'nenhum arquivo suportado'}). "
        f"Nomeie-o como 'data.<ext>' ou informe hf_file/split no experimento."
    )


def _data_files(base_dir: Path) -> List[str]:
    return sorted(
        p.name for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p.name != MANIFEST_NAME
    )


def _read_file(path: Path, read_kwargs: Dict) -> pd.DataFrame:
    """`read_kwargs` vale só para CSV/TSV (sep, encoding, decimal...), para não
    quebrar a leitura de um parquet na mesma pasta."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, **read_kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, **{"sep": "\t", **read_kwargs})
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Formato não suportado: {path.name} (suportados: {SUPPORTED_EXTENSIONS})")

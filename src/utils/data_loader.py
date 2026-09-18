from typing import List, Dict, Optional, Tuple
import pandas as pd
from loguru import logger

from src.api.schemas.annotation_experiment.dataset import DatasetConfig

import sys
import os

from src.config.datasets_collected import DATASETS, LABEL_MEANINGS
from src.utils.local_datasets import refresh_local_datasets, load_local_dataframe
from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import hf_hub_download

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def load_hf_dataset(
    dataset_name: str,
    cache_dir: str,
    dataset_global_config: DatasetConfig,
    dataset_specific_config: Optional[Dict] = None
    
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Carrega um dataset (HF ou local) usando as configurações globais + específicas.
    """
    # ------------------------------
    # 1. Buscar config do dataset
    # ------------------------------
    if dataset_specific_config is None:
        refresh_local_datasets()
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' não encontrado.\n"
                f"Datasets disponíveis: {list(DATASETS.keys())}"
            )
        dataset_specific_config = DATASETS[dataset_name].copy()

    logger.info(f"Carregando dataset: {dataset_name}")
    logger.debug(f"Configuração específica: {dataset_specific_config}")
    logger.debug(f"Configuração global: {dataset_global_config}")

    # ------------------------------
    # 2. Configurações globais
    # ------------------------------
    split = dataset_specific_config.get("split", dataset_global_config.split)
    hf_file = dataset_specific_config.get("hf_file", dataset_global_config.hf_file)
    combine_splits = dataset_specific_config.get("combine_splits", dataset_global_config.combine_splits)
    sample_size = dataset_specific_config.get("sample_size", dataset_global_config.sample_size)
    random_state = dataset_specific_config.get("random_state", dataset_global_config.random_state)

    source = dataset_specific_config.get("source", "hf")
    if source not in ("hf", "local"):
        raise ValueError(f"source '{source}' inválido para o dataset '{dataset_name}' (use 'hf' ou 'local')")

    try:
        # ================================================================
        # DATASET LOCAL (PROPRIETÁRIO / FORA DO HF)
        # ================================================================
        if source == "local":
            logger.info(f"Carregando dataset local de: {dataset_specific_config['path']}")

            df = load_local_dataframe(
                dataset_name=dataset_name,
                spec=dataset_specific_config,
                hf_file=hf_file,
                split=split,
                combine_splits=combine_splits,
            )
            dataset = Dataset.from_pandas(df, preserve_index=False)

            logger.info(f"Dataset carregado: {len(dataset)} exemplos")

        # ================================================================
        # DOWNLOAD DIRETO DE ARQUIVO DO HF (PARQUET / CSV / ETC)
        # ================================================================
        elif hf_file:
            logger.info("Baixando parquet direto do HuggingFace Hub")

            file_path = hf_hub_download(
                repo_id=dataset_specific_config['path'],
                repo_type="dataset",
                filename=hf_file,
                cache_dir=os.path.join(cache_dir, "hf")
            )

            logger.info(f"Arquivo baixado em: {file_path}")

            df = pd.read_parquet(file_path)
            dataset = Dataset.from_pandas(df)

            logger.info(f"Dataset carregado: {len(dataset)} exemplos") 

        # ================================================================
        # COMBINAÇÃO DE SPLITS (SE APLICÁVEL)
        # ================================================================
        elif combine_splits:
            logger.info(f"Combinando splits: {combine_splits}")
            datasets_list = []

            for sp in combine_splits:
                try:
                    ds = load_dataset(dataset_specific_config['path'], split=sp, cache_dir=os.path.join(cache_dir, "hf"))
                    logger.info(f"  ✓ {sp}: {len(ds)} exemplos")
                    datasets_list.append(ds)
                except Exception as e:
                    logger.warning(f"  ⚠️  Split {sp} indisponível ({e})")

            if not datasets_list:
                raise ValueError("Nenhum split disponível para combinar")

            dataset = concatenate_datasets(datasets_list)
            logger.info(f"Total combinado: {len(dataset)} exemplos")

        # ================================================================
        # SPLIT ÚNICO
        # ================================================================
        else:
            dataset = load_dataset(
                dataset_specific_config["path"],
                split=split,    
                cache_dir=os.path.join(cache_dir, "hf")
            )
            logger.info(f"Split '{split}': {len(dataset)} exemplos")
            
        # ================================================================
        # EXTRAIR CATEGORIAS
        # ================================================================
        label_column = dataset_specific_config.get("label_column")
        categories = dataset_specific_config.get("categories")

        if categories is None:
            if label_column and label_column in dataset.column_names:
                categories = sorted(list(set(dataset[label_column])))
                logger.info(f"Categorias extraídas automaticamente: {categories}")
            elif LABEL_MEANINGS.get(dataset_name):
                # Sem ground truth: classes vêm do LABEL_MEANINGS
                categories = sorted(int(k) for k in LABEL_MEANINGS[dataset_name])
                logger.info(f"Categorias obtidas de LABEL_MEANINGS: {categories}")
            else:
                categories = []
                logger.info("Nenhuma categoria disponível")

        # ================================================================
        # AMOSTRAGEM
        # ================================================================
        if random_state:
            dataset = dataset.shuffle(seed=random_state)
            logger.info(f"Dataset embaralhado com seed={random_state}")
        
        if sample_size:
            sample_size = min(sample_size, len(dataset))
            
            dataset = dataset.select(range(sample_size))
            logger.info(f"Amostra reduzida para {sample_size} exemplos")

        # ================================================================
        # EXTRAIR TEXTO
        # ================================================================
        text_column = dataset_specific_config["text_column"]
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Coluna de texto '{text_column}' não encontrada.\n"
                f"Colunas disponíveis: {dataset.column_names}"
            )

        texts = dataset[text_column]
        logger.info(f"Coluna de texto: {text_column}")

        # ================================================================
        # GROUND TRUTH (SE EXISTIR)
        # ================================================================
        ground_truth = None
        if label_column and label_column in dataset.column_names:
            ground_truth = dataset[label_column]
            logger.info(f"Ground truth carregado da coluna '{label_column}'")

        return texts, categories, ground_truth

    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        raise


# =============================================================================
# DATAFRAME
# =============================================================================
def add_label_description(df, dataset_name):
    if dataset_name not in LABEL_MEANINGS:
        refresh_local_datasets()
    mapping = LABEL_MEANINGS.get(dataset_name)

    if mapping is None:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado no LABEL_MEANINGS")

    # Converter label para string e mapear
    df["label_description"] = df["label"].astype(str).map(mapping)

    return df

def load_hf_dataset_as_dataframe(
    dataset_name: str,
    cache_dir: str,
    dataset_global_config: Dict,
    dataset_specific_config: Optional[Dict] = None
) -> pd.DataFrame:

    texts, categories, ground_truth = load_hf_dataset(
        dataset_name, 
        cache_dir, 
        dataset_global_config, 
        dataset_specific_config
    )

    df = pd.DataFrame({
        "text": texts
    })

    if ground_truth is not None:
        df["label"] = ground_truth
        
    df = add_label_description(df, dataset_name)

    logger.info(f"DataFrame criado com {len(df)} linhas")
    return df, categories


# =============================================================================
# LISTAGEM / INFO
# =============================================================================

def list_available_datasets() -> List[str]:
    refresh_local_datasets()
    return list(DATASETS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    refresh_local_datasets()
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado.")
    return DATASETS[dataset_name].copy()


# =============================================================================
# DESCOBRIR ESTRUTURA HF
# =============================================================================

def discover_dataset_structure(hf_path: str, num_examples: int = 3):
    from src.config.datasets_collected import get_dataset_config_names, get_dataset_split_names

    logger.info(f"Descobrindo estrutura do dataset: {hf_path}")

    try:
        try:
            configs = get_dataset_config_names(hf_path)
            logger.info(f"Configurações disponíveis: {configs}")
        except:
            logger.info("Configurações: [default]")

        try:
            splits = get_dataset_split_names(hf_path)
            logger.info(f"Splits disponíveis: {splits}")
        except:
            splits = ["train"]

        dataset = load_dataset(hf_path, split=f"{splits[0]}[:{num_examples}]")
        logger.info(f"Colunas: {dataset.column_names}")
        logger.info(f"Features: {dataset.features}")

        logger.info(f"Primeiros {num_examples} exemplos:")
        for i, example in enumerate(dataset):
            logger.info(f"Exemplo {i + 1}: {example}")

    except Exception as e:
        logger.error(f"Erro ao descobrir estrutura: {e}")


# =============================================================================
# SALVAR RESULTADOS
# =============================================================================

def save_annotated_dataset(
    df: pd.DataFrame,
    output_path: str = "./results/annotated_dataset.csv",
    include_ground_truth: bool = True
):
    df_save = df.copy()

    if not include_ground_truth and "ground_truth" in df_save.columns:
        df_save = df_save.drop(columns=["ground_truth"])

    df_save.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Dataset anotado salvo em: {output_path}")


if __name__ == "__main__":
    logger.info("DATASET CONFIGURATION MODE")

    logger.info("Datasets configurados:")
    for ds in list_available_datasets():
        info = get_dataset_info(ds)
        desc = info.get("description", "Sem descrição")
        logger.info(f" • {ds}: {desc}")

    logger.info("Use discover_dataset_structure('waashk/seu-dataset') para explorar datasets.")

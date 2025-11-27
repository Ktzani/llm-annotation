from typing import List, Dict, Optional, Tuple
import pandas as pd
from loguru import logger

import sys

from src.config.datasets_config import DATASETS, CACHE_DIR
from src.experiments.base_experiment import DATASET_CONFIG
from datasets import load_dataset, concatenate_datasets

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def load_hf_dataset(
    dataset_name: str,
    config: Optional[Dict] = None
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Carrega um dataset do HuggingFace usando as configurações globais + específicas.
    """
    # ------------------------------
    # 1. Buscar config do dataset
    # ------------------------------
    if config is None:
        if dataset_name not in DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' não encontrado.\n"
                f"Datasets disponíveis: {list(DATASETS.keys())}"
            )
        config = DATASETS[dataset_name].copy()

    logger.info(f"Carregando dataset: {dataset_name}")
    logger.debug(f"Configuração específica: {config}")
    logger.debug(f"Configuração global: {DATASET_CONFIG}")

    # ------------------------------
    # 2. Configurações globais
    # ------------------------------
    split = config.get("split", DATASET_CONFIG["split"])
    combine_splits = config.get("combine_splits", DATASET_CONFIG["combine_splits"])
    sample_size = config.get("sample_size", DATASET_CONFIG["sample_size"])

    try:
        # ================================================================
        # COMBINAÇÃO DE SPLITS (SE APLICÁVEL)
        # ================================================================
        if combine_splits:
            logger.info(f"Combinando splits: {combine_splits}")
            datasets_list = []

            for sp in combine_splits:
                try:
                    ds = load_dataset(config['path'], split=sp, cache_dir=CACHE_DIR)
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
                config["path"],
                split=split,
                cache_dir=CACHE_DIR
            )
            logger.info(f"Split '{split}': {len(dataset)} exemplos")
            
        # ================================================================
        # EXTRAIR CATEGORIAS
        # ================================================================
        label_column = config.get("label_column")
        categories = config.get("categories")

        if categories is None:
            if label_column and label_column in dataset.column_names:
                categories = sorted(list(set(dataset[label_column])))
                logger.info(f"Categorias extraídas automaticamente: {categories}")
            else:
                categories = []
                logger.info("Nenhuma categoria disponível")

        # ================================================================
        # AMOSTRAGEM
        # ================================================================
        if sample_size is not None:
            sample_size = min(sample_size, len(dataset))
            dataset = dataset.select(range(sample_size))
            logger.info(f"Amostra reduzida para {sample_size} exemplos")

        # ================================================================
        # EXTRAIR TEXTO
        # ================================================================
        text_column = config["text_column"]
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

def load_hf_dataset_as_dataframe(
    dataset_name: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:

    texts, categories, ground_truth = load_hf_dataset(dataset_name, config)

    df = pd.DataFrame({
        "text": texts
    })

    if ground_truth is not None:
        df["label"] = ground_truth

    logger.info(f"DataFrame criado com {len(df)} linhas")
    return df, categories


# =============================================================================
# LISTAGEM / INFO
# =============================================================================

def list_available_datasets() -> List[str]:
    return list(DATASETS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado.")
    return DATASETS[dataset_name].copy()


# =============================================================================
# DESCOBRIR ESTRUTURA HF
# =============================================================================

def discover_dataset_structure(hf_path: str, num_examples: int = 3):
    from datasets_config import get_dataset_config_names, get_dataset_split_names

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

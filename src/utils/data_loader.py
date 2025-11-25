"""
Data Loader - Carrega datasets do HuggingFace
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
from loguru import logger

import sys
from pathlib import Path as PathLib

config_path = PathLib(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_path))

from datasets import HUGGINGFACE_DATASETS, DATASET_CONFIG


def load_hf_dataset(
    dataset_name: str,
    config: Optional[Dict] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Carrega um dataset do HuggingFace para anotação
    
    Args:
        dataset_name: Nome do dataset em HUGGINGFACE_DATASETS
        config: Configuração customizada (opcional)
        use_cache: Se True, usa cache do HuggingFace
        force_reload: Se True, recarrega mesmo com cache
    
    Returns:
        Tuple com (texts, categories, ground_truth_labels)
    """
    # Configuração
    if config is None:
        if dataset_name not in HUGGINGFACE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' não encontrado.\n"
                f"Datasets disponíveis: {list(HUGGINGFACE_DATASETS.keys())}"
            )
        config = HUGGINGFACE_DATASETS[dataset_name]
    
    logger.info(f"Carregando dataset: {dataset_name}")
    logger.info(f"Path: {config['path']}")
    
    # Importar datasets
    from datasets import load_dataset
    
    # Carregar
    split = config.get('split', DATASET_CONFIG['split'])
    
    try:
        dataset = load_dataset(
            config['path'],
            split=split,
            cache_dir=".cache/huggingface" if use_cache else None
        )
        
        if force_reload:
            dataset = dataset.shuffle()
        
        logger.success(f"Dataset carregado: {len(dataset)} exemplos")
        
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {e}")
        raise
    
    # Extrair textos
    text_column = config['text_column']
    texts = dataset[text_column]
    
    # Extrair labels (se existir)
    label_column = config.get('label_column')
    ground_truth = None
    
    if label_column and label_column in dataset.column_names:
        ground_truth = dataset[label_column]
        logger.info(f"Ground truth disponível: {len(ground_truth)} labels")
    
    # Categorias
    categories = config.get('categories')
    
    if categories is None and ground_truth:
        # Extrair automaticamente
        categories = sorted(list(set(ground_truth)))
        logger.info(f"Categorias extraídas: {categories}")
    elif categories is None:
        raise ValueError("Defina 'categories' para dataset sem labels")
    
    # Sample
    sample_size = config.get('sample_size', DATASET_CONFIG['sample_size'])
    
    if sample_size and sample_size < len(texts):
        texts = texts[:sample_size]
        if ground_truth:
            ground_truth = ground_truth[:sample_size]
        logger.info(f"Usando sample: {sample_size} textos")
    
    logger.success("Dataset pronto para anotação")
    
    return texts, categories, ground_truth


def list_available_datasets() -> List[str]:
    """Lista datasets disponíveis"""
    return list(HUGGINGFACE_DATASETS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    """Retorna informações de um dataset"""
    if dataset_name not in HUGGINGFACE_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' não encontrado")
    
    return HUGGINGFACE_DATASETS[dataset_name]


def discover_dataset_structure(dataset_path: str) -> Dict:
    """
    Descobre estrutura de um dataset
    
    Args:
        dataset_path: Path do dataset (ex: "waashk/agnews")
    
    Returns:
        Dicionário com informações da estrutura
    """
    logger.info(f"Descobrindo estrutura: {dataset_path}")
    
    from datasets import load_dataset
    
    try:
        # Carregar pequena amostra
        dataset = load_dataset(dataset_path, split="train[:10]")
        
        info = {
            "columns": dataset.column_names,
            "num_rows": len(dataset),
            "features": str(dataset.features),
            "sample": dataset[0]
        }
        
        logger.success("Estrutura descoberta")
        
        return info
    
    except Exception as e:
        logger.error(f"Erro: {e}")
        raise

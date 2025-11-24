from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
import pandas as pd

import sys
from pathlib import Path as PathLib

config_path = PathLib(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_path))

from datasets import HUGGINGFACE_DATASETS

def load_hf_dataset(
    dataset_name: str,
    config: Optional[Dict] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Carrega um dataset do HuggingFace para anotação
    
    Args:
        dataset_name: Nome do dataset em HUGGINGFACE_DATASETS ou path direto
        config: Configuração customizada (opcional)
        use_cache: Se True, usa cache do HuggingFace
        force_reload: Se True, recarrega mesmo com cache
    
    Returns:
        Tuple com (texts, categories, ground_truth_labels)
        - texts: Lista de textos para anotar
        - categories: Lista de categorias possíveis
        - ground_truth_labels: Labels verdadeiros (se disponível) ou None
    
    Example:
        >>> texts, categories, labels = load_hf_dataset("exemplo_com_labels")
        >>> print(f"Carregados {len(texts)} textos para anotação")
    """
    # Usar config predefinida ou customizada
    if config is None:
        if dataset_name not in HUGGINGFACE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' não encontrado.\n"
                f"Datasets disponíveis: {list(HUGGINGFACE_DATASETS.keys())}\n"
                f"Ou use load_custom_dataset() para carregar diretamente."
            )
        config = HUGGINGFACE_DATASETS[dataset_name]
    
    print(f"\n{'='*80}")
    print(f"📦 Carregando dataset: {dataset_name}")
    print(f"{'='*80}")
    print(f"Path: {config['path']}")
    if 'description' in config:
        print(f"Descrição: {config['description']}")
    
    try:
        cache_dir = "./data/.cache/huggingface" if use_cache else None
        
        # Tratar combinação de splits
        if config.get('combine_splits'):
            print(f"\n🔄 Combinando splits: {config['combine_splits']}")
            datasets_list = []
            
            for split in config['combine_splits']:
                try:
                    ds = load_dataset(
                        config['path'],
                        split=split,
                        cache_dir=cache_dir
                    )
                    datasets_list.append(ds)
                    print(f"   ✓ {split}: {len(ds)} exemplos")
                except Exception as e:
                    print(f"   ⚠️  {split}: não disponível ({str(e)})")
            
            if not datasets_list:
                raise ValueError("Nenhum split disponível para combinar")
            
            # Concatenar todos os datasets
            from datasets import concatenate_datasets
            dataset = concatenate_datasets(datasets_list)
            print(f"\n   ✓ Total combinado: {len(dataset)} exemplos")
        
        else:
            # Carregar split único
            split = config['split']
            dataset = load_dataset(
                config['path'],
                split=split,
                cache_dir=cache_dir
            )
            print(f"\n✓ Split '{split}': {len(dataset)} exemplos")
        
        # Aplicar amostragem se configurado
        if config.get('sample_size') is not None:
            sample_size = min(config['sample_size'], len(dataset))
            dataset = dataset.select(range(sample_size))
            print(f"✓ Amostra selecionada: {sample_size} exemplos")
        
        # Extrair textos
        text_column = config['text_column']
        if text_column not in dataset.column_names:
            raise ValueError(
                f"Coluna de texto '{text_column}' não encontrada.\n"
                f"Colunas disponíveis: {dataset.column_names}"
            )
        
        texts = dataset[text_column]
        print(f"✓ Textos extraídos da coluna: '{text_column}'")
        
        # Extrair categorias
        categories = config['categories']
        if categories is None:
            # Extrair categorias automaticamente dos labels
            label_column = config.get('label_column')
            if label_column and label_column in dataset.column_names:
                unique_labels = set(dataset[label_column])
                categories = sorted(list(unique_labels))
                print(f"✓ Categorias extraídas automaticamente: {categories}")
            else:
                raise ValueError(
                    "Categorias não fornecidas e não foi possível extrair automaticamente.\n"
                    "Forneça 'categories' na configuração ou 'label_column' com labels válidos."
                )
        else:
            print(f"✓ Categorias configuradas: {categories}")
        
        # Extrair ground truth labels (se disponível)
        ground_truth = None
        label_column = config.get('label_column')
        if label_column and label_column in dataset.column_names:
            ground_truth = dataset[label_column]
            print(f"✓ Ground truth disponível (coluna: '{label_column}')")
            print(f"  → Pode ser usado para validação da qualidade das anotações")
        else:
            print(f"ℹ️  Ground truth não disponível (anotação do zero)")
        
        print(f"\n{'='*80}")
        print(f"✅ Dataset pronto para anotação!")
        print(f"{'='*80}\n")
        
        return texts, categories, ground_truth
        
    except Exception as e:
        print(f"\n❌ Erro ao carregar dataset:")
        print(f"   {str(e)}\n")
        raise


def load_hf_dataset_as_dataframe(
    dataset_name: str,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Carrega dataset e retorna como DataFrame pandas
    
    Args:
        dataset_name: Nome do dataset
        config: Configuração customizada (opcional)
    
    Returns:
        DataFrame com colunas: text_id, text, ground_truth (se disponível)
    
    Example:
        >>> df = load_hf_dataset_as_dataframe("exemplo_com_labels")
        >>> print(df.head())
    """
    texts, categories, ground_truth = load_hf_dataset(dataset_name, config)
    
    df = pd.DataFrame({
        'text_id': range(len(texts)),
        'text': texts
    })
    
    if ground_truth is not None:
        df['ground_truth'] = ground_truth
    
    return df


def load_custom_dataset(
    hf_path: str,
    text_column: str,
    label_column: Optional[str] = None,
    categories: Optional[List[str]] = None,
    split: str = "train",
    combine_splits: Optional[List[str]] = None,
    sample_size: Optional[int] = None
) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """
    Carrega um dataset personalizado diretamente sem pré-configurar
    
    Args:
        hf_path: Path do dataset no HuggingFace (ex: "waashk/meu-dataset")
        text_column: Nome da coluna com textos
        label_column: Nome da coluna com labels (opcional)
        categories: Lista de categorias (opcional, será extraída se None)
        split: Split a carregar ("train", "test", etc)
        combine_splits: Lista de splits para combinar (sobrescreve split)
        sample_size: Número de exemplos a carregar (None = todos)
    
    Returns:
        Tuple com (texts, categories, ground_truth_labels)
    
    Example:
        >>> texts, cats, labels = load_custom_dataset(
        ...     "waashk/meu-dataset",
        ...     text_column="content",
        ...     label_column="category",
        ...     combine_splits=["train", "test"],  # Usar tudo
        ...     sample_size=100
        ... )
    """
    config = {
        'path': hf_path,
        'text_column': text_column,
        'label_column': label_column,
        'categories': categories,
        'split': split,
        'sample_size': sample_size,
    }
    
    if combine_splits:
        config['combine_splits'] = combine_splits
        config['split'] = None
    
    return load_hf_dataset("custom", config)


def list_available_datasets() -> List[str]:
    """
    Lista todos os datasets configurados
    
    Returns:
        Lista de nomes de datasets disponíveis
    """
    return list(HUGGINGFACE_DATASETS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Retorna informações sobre um dataset configurado
    
    Args:
        dataset_name: Nome do dataset
    
    Returns:
        Dicionário com configurações do dataset
    """
    if dataset_name not in HUGGINGFACE_DATASETS:
        raise ValueError(
            f"Dataset '{dataset_name}' não encontrado. "
            f"Disponíveis: {list(HUGGINGFACE_DATASETS.keys())}"
        )
    
    return HUGGINGFACE_DATASETS[dataset_name].copy()


def discover_dataset_structure(hf_path: str, num_examples: int = 3):
    """
    Descobre e exibe a estrutura de um dataset do HuggingFace
    
    Args:
        hf_path: Path do dataset (ex: "waashk/meu-dataset")
        num_examples: Número de exemplos a mostrar
    
    Example:
        >>> discover_dataset_structure("waashk/meu-dataset")
    """
    print(f"\n{'='*80}")
    print(f"🔍 Descobrindo estrutura: {hf_path}")
    print(f"{'='*80}\n")
    
    try:
        # Tentar carregar splits disponíveis
        from datasets import get_dataset_config_names, get_dataset_split_names
        
        try:
            configs = get_dataset_config_names(hf_path)
            print(f"Configurações disponíveis: {configs}")
        except:
            print("Configurações: [default]")
        
        try:
            splits = get_dataset_split_names(hf_path)
            print(f"Splits disponíveis: {splits}\n")
        except:
            splits = ["train"]
            print(f"Splits disponíveis: {splits} (padrão)\n")
        
        # Carregar amostra
        dataset = load_dataset(hf_path, split=f"{splits[0]}[:{num_examples}]")
        
        print(f"📋 Estrutura do dataset:")
        print(f"   Colunas: {dataset.column_names}")
        print(f"   Features: {dataset.features}\n")
        
        print(f"📝 Primeiros {num_examples} exemplos:")
        for i, example in enumerate(dataset):
            print(f"\n   Exemplo {i+1}:")
            for key, value in example.items():
                value_str = str(value)[:100]
                print(f"      {key}: {value_str}...")
        
        print(f"\n{'='*80}")
        print("✅ Estrutura descoberta!")
        print(f"{'='*80}\n")
        
        # Sugerir configuração
        print("💡 Sugestão de configuração:")
        print(f'''
"seu_dataset": {{
    "path": "{hf_path}",
    "text_column": "{dataset.column_names[0]}",  # AJUSTE SE NECESSÁRIO
    "label_column": None,  # ou nome da coluna de label
    "categories": ["Cat1", "Cat2", "Cat3"],  # DEFINA SUAS CATEGORIAS
    "split": "{splits[0]}",
    "sample_size": 100,  # Começar pequeno
    "description": "Seu dataset para anotação"
}},
''')
        
    except Exception as e:
        print(f"❌ Erro: {str(e)}\n")


# =============================================================================
# UTILITÁRIOS DE SALVAMENTO
# =============================================================================

def save_annotated_dataset(
    df: pd.DataFrame,
    output_path: str = "./results/annotated_dataset.csv",
    include_ground_truth: bool = True
):
    """
    Salva dataset anotado em formato padronizado
    
    Args:
        df: DataFrame com anotações
        output_path: Caminho para salvar
        include_ground_truth: Se False, remove coluna ground_truth
    """
    df_save = df.copy()
    
    if not include_ground_truth and 'ground_truth' in df_save.columns:
        df_save = df_save.drop(columns=['ground_truth'])
    
    df_save.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Dataset anotado salvo: {output_path}")


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" " * 25 + "DATASET CONFIGURATION")
    print("="*80 + "\n")
    
    # 1. Listar datasets configurados
    print("📋 Datasets configurados:")
    for ds in list_available_datasets():
        info = get_dataset_info(ds)
        desc = info.get('description', 'Sem descrição')
        print(f"   • {ds}: {desc}")
    
    print("\n" + "="*80 + "\n")
    
    # 2. Descobrir estrutura de um dataset
    print("🔍 Para descobrir a estrutura de um dataset:")
    print('   discover_dataset_structure("waashk/seu-dataset")\n')
    
    # 3. Carregar um dataset (comentado - descomente para testar)
    # print("📦 Carregando dataset de exemplo...")
    # texts, categories, labels = load_hf_dataset("exemplo_com_labels")
    # print(f"   Textos: {len(texts)}")
    # print(f"   Categorias: {categories}")
    # print(f"   Ground truth: {'Sim' if labels else 'Não'}")

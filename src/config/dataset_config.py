# =============================================================================
# CONFIGURAÇÕES DE DATASETS DO HUGGINGFACE (waashk)
# =============================================================================

HUGGINGFACE_DATASETS = {
    # ==========================================================================
    # INSTRUÇÕES DE USO:
    # 
    # 1. Veja seus datasets em: https://huggingface.co/waashk
    # 2. Para cada dataset, adicione uma entrada aqui
    # 3. Use 'split': 'train' como padrão (podemos combinar splits se necessário)
    # 4. Ajuste 'sample_size' para começar com amostra pequena
    # 
    # EXEMPLO DE ESTRUTURA:
    # "nome_dataset": {
    #     "path": "waashk/nome-exato-no-hf",
    #     "text_column": "nome_da_coluna_texto",
    #     "label_column": "nome_da_coluna_label",  # ou None se não tiver
    #     "categories": ["Cat1", "Cat2"],  # ou None para extrair auto
    #     "split": "train",  # ou combine múltiplos splits
    #     "sample_size": None,  # None = tudo, ou número específico
    # }
    # ==========================================================================
    
    # Exemplo 1: Dataset com labels (para validação)
    "exemplo_com_labels": {
        "path": "waashk/seu-dataset-aqui",  # AJUSTE
        "text_column": "text",              # AJUSTE
        "label_column": "label",            # AJUSTE
        "categories": None,                  # Extrair automaticamente
        "split": "train",                    # Usar todo o dataset
        "sample_size": 100,                  # Começar pequeno!
        "description": "Dataset exemplo com ground truth para validação"
    },
    
    # Exemplo 2: Dataset sem labels (anotação pura)
    "exemplo_sem_labels": {
        "path": "waashk/outro-dataset",     # AJUSTE
        "text_column": "content",           # AJUSTE
        "label_column": None,               # Sem labels existentes
        "categories": ["Categoria1", "Categoria2", "Categoria3"],  # DEFINA
        "split": "train",
        "sample_size": None,                # Carregar tudo
        "description": "Dataset para anotação do zero"
    },
    
    # Exemplo 3: Combinar múltiplos splits
    "exemplo_dataset_completo": {
        "path": "waashk/dataset-completo",
        "text_column": "text",
        "label_column": "category",
        "categories": None,
        "split": None,  # Será tratado especialmente para combinar splits
        "combine_splits": ["train", "test", "validation"],  # Combinar todos
        "sample_size": None,
        "description": "Dataset completo com todos os splits combinados"
    },
}



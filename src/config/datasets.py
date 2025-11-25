"""
Dataset Configurations - HuggingFace Datasets (waashk)
"""

# =============================================================================
# CONFIGURAÇÕES DE DATASETS DO HUGGINGFACE (waashk)
# =============================================================================

EXEMPLO_USO_DATASETS = {
    # ==========================================================================
    # INSTRUÇÕES DE USO:
    # 
    # 1. Veja seus datasets em: https://huggingface.co/waashk
    # 2. Para cada dataset, adicione uma entrada aqui
    # 
    # EXEMPLO DE ESTRUTURA:
    # "nome_dataset": {
    #     "path": "waashk/nome-exato-no-hf",
    #     "text_column": "nome_da_coluna_texto",
    #     "label_column": "nome_da_coluna_label",  # ou None se não tiver
    #     "categories": ["Cat1", "Cat2"],  # ou None para extrair auto
    #     "description": "Breve descrição do dataset"
    # }
    # ==========================================================================
    
    # Exemplo 1: Dataset com labels (para validação)
    "exemplo_com_labels": {
        "path": "PATH/seu-dataset-aqui",  # AJUSTE
        "text_column": "text",              # AJUSTE
        "label_column": "label",            # AJUSTE
        "categories": None,                  # Extrair automaticamente
        "description": "Dataset exemplo com ground truth para validação"
    },
    
    # Exemplo 2: Dataset sem labels (anotação pura)
    "exemplo_sem_labels": {
        "path": "PATH/outro-dataset",     # AJUSTE
        "text_column": "content",           # AJUSTE
        "label_column": None,               # Sem labels existentes
        "categories": ["Categoria1", "Categoria2", "Categoria3"],  # DEFINA             
        "description": "Dataset para anotação do zero"
    },
    
    # Exemplo 3: Combinar múltiplos splits
    "exemplo_dataset_completo": {
        "path": "PATH/dataset-completo",
        "text_column": "text",
        "label_column": "category",
        "categories": None,
        "description": "Dataset completo com todos os splits combinados"
    },
}

# =============================================================================
# DATASETS REAIS (waashk)
# =============================================================================

HUGGINGFACE_DATASETS = {
    "agnews": {
        "path": "waashk/agnews",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "AG News para classificação de notícias"
    },
    
    "mpqa": {
        "path": "waashk/mpqa",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "MPQA para análise de opinião / sentimento"  # conforme HF :contentReference[oaicite:2]{index=2}
    },
    "webkb": {
        "path": "waashk/webkb",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "WebKB: páginas web classificadas por tipo"  # conforme HF :contentReference[oaicite:3]{index=3}
    },
    "ohsumed": {
        "path": "waashk/ohsumed",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "OHSUMED: resumos médicos para classificação"  # conforme HF :contentReference[oaicite:4]{index=4}
    },
    "acm": {
        "path": "waashk/acm",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "ACM (parte do atcBench, do waashk)"  # conforme README do ACM no atcBench :contentReference[oaicite:5]{index=5}
    },
    "yelp_2013": {
        "path": "waashk/yelp_2013",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Yelp 2013 para classificação de reviews"  # confirmado no HF :contentReference[oaicite:6]{index=6}
    },
    "dblp": {
        "path": "waashk/dblp",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "DBLP (do atcBench)"
    },
    "books": {
        "path": "waashk/books",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Books dataset do atcBench"
    },
    "reut90": {
        "path": "waashk/reut90",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Reuters90 (do atcBench)"
    },
    "wos11967": {
        "path": "waashk/wos11967",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "WOS-11967 (do atcBench)"
    },
    "twitter": {
        "path": "waashk/twitter",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Twitter dataset (atcBench)"
    },
    "trec": {
        "path": "waashk/trec",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "TREC (atcBench)"
    },
    "wos5736": {
        "path": "waashk/wos5736",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "WOS-5736 (atcBench)"
    },
    "sst1": {
        "path": "waashk/sst1",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "SST-1 (atcBench)"
    },
    "pang_movie": {
        "path": "waashk/pang_movie_2L",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Pang movie reviews (atcBench)"
    },
    "movie_review": {
        "path": "waashk/movie_review",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Movie Review (atcBench)"
    },
    "vader_movie": {
        "path": "waashk/vader_movie_2L",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "VADER Movie sentiment (atcBench)"
    },
    "subj": {
        "path": "waashk/subj",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Subjectivity dataset (atcBench)"
    },
    "sst2": {
        "path": "waashk/sst2",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "SST-2 (atcBench)"
    },
    "yelp_reviews": {
        "path": "waashk/yelp_reviews_2L",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Yelp Reviews (atcBench)"
    },
    "20ng": {
        "path": "waashk/20ng",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "20ng dataset (atcBench)"
    },
    "medline": {
        "path": "waashk/medline",
        "text_column": "text",
        "label_column": "label",
        "categories": None,
        "description": "Medline dataset (atcBench)"
    },
}

# =============================================================================
# CONFIGURAÇÃO GLOBAL DE DATASETS
# =============================================================================

# Use 'split': 'train' como padrão (podemos combinar splits se necessário)
# Ajuste 'sample_size' para começar com amostra pequena
# Ajuste 'combine_splits' para combinar múltiplos splits quando necessário. Ex: ["train", "test"]
DATASET_CONFIG = {
    "split": "train",
    "combine_splits": None,
    "sample_size": 100
}

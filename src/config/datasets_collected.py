"""
Dataset Configurations - HuggingFace Datasets (waashk)
"""
CACHE_DIR = "..\..\data\.cache\hf"
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

DATASETS = {
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

LABEL_MEANINGS = {
    "agnews": {
        "0": "world",
        "1": "sports",
        "2": "business",
        "3": "Sci/Tech",
    },
    "mpqa": {
        "0": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "webkb": {
        "0": "student",
        "1": "faculty",
        "2": "staff",
        "3": "department",
        "4": "course",
        "5": "project",
        "6": "other",
    },
    "ohsumed": {
        "0": "Bacterial Infections and Mycoses",
        "1": "Virus Diseases",
        "2": "Parasitic Diseases",
        "3": "Neoplasms",
        "4": "Musculoskeletal Diseases",
        "5": "Digestive System Diseases",
        "6": "Stomatognathic Diseases",
        "7": "Respiratory Tract Diseases",
        "8": "Otorhinolaryngologic Diseases",
        "9": "Nervous System Diseases",
        "10": "Eye Diseases",
        "11": "Urologic and Male Genital Diseases",
        "12": "Female Genital Diseases and Pregnancy Complications",
        "13": "Cardiovascular Diseases",
        "14": "Hemic and Lymphatic Diseases",
        "15": "Neonatal Diseases and Abnormalities",
        "16": "Skin and Connective Tissue Diseases",
        "17": "Nutritional and Metabolic Diseases",
        "18": "Endocrine Diseases",
        "19": "Immunologic Diseases",
        "20": "Disorders of Environmental Origin",
        "21": "Animal Diseases",
        "22": "Pathological Conditions, Signs and Symptoms"
    },
    "acm": {
        "0": "General Literature",
        "1": "Hardware",
        "2": "Computer Systems Organization",
        "3": "Software",
        "4": "Data",
        "5": "Theory of Computation",
        "6": "Mathematics of Computing",
        "7": "Information Systems",
        "8": "Computing Methodologies",
        "9": "Computer Applications",
        "10": "Computing Milieux",
    },
    "yelp_2013": {
        "0": "very negative sentiment",
        "1": "negative sentiment",
        "2": "neutral sentiment",
        "3": "positive sentiment",
        "4": "very positive sentiment",
    },
    "dblp": {
        "0": "computer vision",
        "1": "computational linguistics",
        "2": "biomedical engineering",
        "3": "software engineering",
        "4": "graphics",
        "5": "data mining",
        "6": "security and cryptography",
        "7": "signal processing",
        "8": "robotics",
        "9": "theory",
    },
    "books": {
        "0": "children",
        "1": "graphic comics",
        "2": "paranormal fantasy",
        "3": "history & biography",
        "4": "crime & mystery thriller",
        "5": "poetry",
        "6": "romance",
        "7": "young adult",
    },
    "reut90": {
        "0": "rape-oil",
        "1": "groundnut-oil",
        "2": "dmk",
        "3": "cpi",
        "4": "meal-feed",
        "5": "wpi",
        "6": "livestock",
        "7": "palm-oil",
        "8": "jobs",
        "9": "palladium",
        "10": "nzdlr",
        "11": "sunseed",
        "12": "alum",
        "13": "sorghum",
        "14": "potato",
        "15": "carcass",
        "16": "money-fx",
        "17": "grain",
        "18": "coffee",
        "19": "hog",
        "20": "palmkernel",
        "21": "rand",
        "22": "zinc",
        "23": "heat",
        "24": "interest",
        "25": "orange",
        "26": "oilseed",
        "27": "rice",
        "28": "lumber",
        "29": "rubber",
        "30": "cotton",
        "31": "crude",
        "32": "fuel",
        "33": "cocoa",
        "34": "coconut",
        "35": "nkr",
        "36": "platinum",
        "37": "coconut-oil",
        "38": "instal-debt",
        "39": "acq",
        "40": "ship",
        "41": "earn",
        "42": "lei",
        "43": "sugar",
        "44": "dlr",
        "45": "sun-meal",
        "46": "naphtha",
        "47": "lead",
        "48": "pet-chem",
        "49": "oat",
        "50": "jet",
        "51": "l-cattle",
        "52": "trade",
        "53": "castor-oil",
        "54": "tea",
        "55": "money-supply",
        "56": "dfl",
        "57": "retail",
        "58": "income",
        "59": "gas",
        "60": "copper",
        "61": "nickel",
        "62": "barley",
        "63": "corn",
        "64": "rapeseed",
        "65": "copra-cake",
        "66": "propane",
        "67": "veg-oil",
        "68": "soy-meal",
        "69": "tin",
        "70": "groundnut",
        "71": "wheat",
        "72": "iron-steel",
        "73": "reserves",
        "74": "lin-oil",
        "75": "silver",
        "76": "gnp",
        "77": "yen",
        "78": "housing",
        "79": "soy-oil",
        "80": "sun-oil",
        "81": "strategic-metal",
        "82": "cpu",
        "83": "nat-gas",
        "84": "ipi",
        "85": "gold",
        "86": "rye",
        "87": "soybean",
        "88": "cotton-oil",
        "89": "bop"
    },
    "wos11967": {
        "0": "Computer vision",
        "1": "Machine learning",
        "2": "network security",
        "3": "Cryptography",
        "4": "Operating systems",
        "5": "Electricity",
        "6": "Electrical circuits",
        "7": "Digital control",
        "8": "Prejudice",
        "9": "Social cognition",
        "10": "Person perception",
        "11": "Nonverbal communication",
        "12": "Prosocial behavior",
        "13": "Computer-aided design",
        "14": "Hydraulics",
        "15": "Manufacturing engineering",
        "16": "Machine design",
        "17": "Fluid mechanics",
        "18": "Ambient Intelligence",
        "19": "Geotextile",
        "20": "Remote Sensing",
        "21": "Rainwater Harvesting",
        "22": "Water Pollution",
        "23": "Addiction",
        "24": "Allergies",
        "25": "Alzheimer's Disease",
        "26": "Ankylosing Spondylitis",
        "27": "Anxiety",
        "28": "Molecular biology",
        "29": "Cell biology",
        "30": "Human Metabolism",
        "31": "Immunology",
        "32": "Genetics"
    },
    "twitter": {
        "0": None,
        "1": None,
        "2": None,
        "3": None,
        "4": None,
        "5": None
    },
    "trec": {
        "0": "DESCRIPTION",
        "1": "ENTITY",
        "2": "ABBREVIATION",
        "3": "HUMAN",
        "4": "LOCATION",
        "5": "NUMERIC VALUE",
    },
    "wos5736": {
        "0": "Electricity",
        "1": "Digital control",
        "2": "Operational amplifier",
        "3": "Social cognition",
        "4": "Child abuse",
        "5": "Depression",
        "6": "Attention",
        "7": "Molecular biology",
        "8": "Immunology",
        "9": "Polymerase chain reaction",
        "10": "Northern blotting",
    },
    "sst1": {
        "0": "very negative sentiment",
        "1": "negative sentiment",
        "2": "neutral sentiment",
        "3": "positive sentiment",
        "4": "very positive sentiment",
    },
    "pang_movie": {
        "0": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "movie_review": {
        "0": "negative sentiment",
        "1": "positive sentiment", 
    },
    "vader_movie": {
        "0": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "subj": {
        "0": "subjective sentence",
        "1": "objective sentence",
    },
    "sst2": {
        "0": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "yelp_reviews": {
        "0": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "20ng": {
        "0": "atheist resources",
        "1": "computer graphics",
        "2": "computer os ms windows misc",
        "3": "computer system ibm pc hardware",
        "4": "computer system mac hardware",
        "5": "computer windows x",
        "6": "misc miscellaneous for sale",
        "7": "rec autos",
        "8": "rec motorcycles",
        "9": "rec sport baseball",
        "10": "rec sport hockey",
        "11": "science crypt",
        "12": "science electronics",
        "13": "science med",
        "14": "science space",
        "15": "society religion christian",
        "16": "talk politics guns",
        "17": "talk politics mideast",
        "18": "talk politics misc miscellaneous",
        "19": "talk religion misc miscellaneous"
    },
    "medline": {
        "0": None,
        "1": None,
        "2": None,
        "3": None,
        "4": None,
        "5": None,
        "6": None
    },
    
    ## Extras Datasetlabels
    "aisopos_ntua_2L": {
        "-1": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "vader_nyt_2L": {
        "-1": "Negative Sentiment",
        "1": "Positive Sentiment",
    },
    "imdb_reviews": {
        "1": "extremely negative sentiment",
        "2": "very negative sentiment",
        "3": "negative sentiment",
        "4": "slightly negative sentiment",
        "5": "negative neutral ",
        "6": "positive neutral",
        "7": "slightly positive sentiment",
        "8": "positive sentiment",
        "9": "very positive sentiment",
        "10": "extremely positive sentiment",
    },
    "sogou": {
        "1": "sports",
        "2": "finance",
        "3": "entertainment",
        "4": "automobile",
        "5": "technology",
    },
    "yelp_2015": {
        "1": "very negative sentiment",
        "2": "negative sentiment",
        "3": "neutral sentiment",
        "4": "positive sentiment",
        "5": "very positive sentiment",
    },
}
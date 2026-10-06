"""
Human Validation Configurations - Amostragem incremental e materiais da validação humana
"""

# =============================================================================
# AMOSTRAGEM
# =============================================================================

HUMAN_VALIDATION_SEED = 42

# Documentos por grupo de concordância na 1ª rodada e em cada rodada seguinte
INITIAL_ROUND_SIZE = 30
INCREMENT_SIZE = 20

# Mínimo por classe na 1ª rodada (o estimador de variância estratificado exige n_h >= 2)
MIN_PER_CLASS_FIRST_ROUND = 2

# =============================================================================
# ESTIMAÇÃO (critério de parada)
# =============================================================================

# Para quando a MoE do IC de Wald da métrica principal fica <= MOE_THRESHOLD
MOE_THRESHOLD = 0.05
CONFIDENCE_LEVEL = 0.95
PRIMARY_METRIC = "acerto_referencia"

# Pseudo-contagem na variância de cada estrato (0 = Wald original do artigo).
# 0,25 escolhido por simulação: Wald subcobre no grupo C (46-69%), k=1 supercobre (~100%)
VARIANCE_PSEUDO_COUNT = 0.25

ESTIMATION_METRICS = {
    "acerto_referencia": "rótulo humano (maioria) = rótulo de referência",
    "acerto_llm": "rótulo humano (maioria) = rótulo consolidado das LLMs",
    "outro_rotulo_possivel": "maioria dos avaliadores indicou outro rótulo possível",
    # Situações do RQ4 (denominador: todos os documentos completos)
    "benchmark_correct": "humanos convergem para o rótulo de referência",
    "benchmark_mislabeling": "humanos convergem para outra classe",
    "genuine_ambiguity": "avaliadores sem maioria ou lado oposto do conflito defensável para a maioria",
    "insufficient_information": "maioria escolheu 'informação insuficiente'",
    # Obs.: acerto_referencia e acerto_llm contam documentos sem maioria como 0
}

AGREEMENT_GROUPS = {
    "A": "as três LLMs concordam entre si e divergem do rótulo de referência",
    "B": "duas LLMs divergem do rótulo de referência e uma concorda com ele",
    "C": "as três LLMs concordam entre si e com o rótulo de referência",
}

# =============================================================================
# MATERIAIS
# =============================================================================

EVALUATORS = ["avaliador_1", "avaliador_2", "avaliador_3"]

# Opção extra de rotulo_escolhido para texto sem informação suficiente / fora da taxonomia
INSUFFICIENT_INFO_OPTION = "não é possível decidir com este texto"

# =============================================================================
# INTERFACE WEB
# =============================================================================

# Banco SQLite da interface (respostas, rodadas e sessões), em <results>/validacao_humana/
INTERFACE_DB_NAME = "validacao_humana.db"
INTERFACE_HOST = "0.0.0.0"
INTERFACE_PORT = 8001
# Códigos de acesso vêm do .env (nunca do repositório):
#   HV_ADMIN_CODE=<código do administrador>
#   HV_CODE_AVALIADOR_1=<código>  (uma variável por avaliador: HV_CODE_<NOME EM MAIÚSCULAS>)
ADMIN_USER = "admin"

# Prefixo do id_anonimo (fallback: duas primeiras letras do dataset)
ID_PREFIXES = {"books": "BK", "dblp": "DB"}

# Exemplos do guia: N por classe, sorteados entre os textos mais curtos do grupo C
EXAMPLES_PER_CLASS = 3
EXAMPLE_LENGTH_QUANTILE = 0.25
EXAMPLE_MAX_CHARS = 500

# Descrições de apoio do guia, redigidas a partir do nome da classe em
# LABEL_MEANINGS (não extraídas da documentação); o rótulo usado é sempre o nome canônico
CLASS_DEFINITIONS = {
    "books": {
        "0": "Livros infantis: histórias, contos ou livros ilustrados voltados a crianças, com linguagem simples.",
        "1": "Quadrinhos, graphic novels e mangás: obras narradas por arte sequencial (desenhos e balões).",
        "2": "Fantasia e sobrenatural: mundos mágicos, criaturas míticas, vampiros, lobisomens, bruxas ou poderes sobrenaturais.",
        "3": "História e biografia: não ficção sobre fatos históricos ou sobre a vida de pessoas reais (biografias, memórias).",
        "4": "Crime, mistério e suspense: investigações, crimes, detetives, conspirações e tramas de tensão.",
        "5": "Poesia: coletâneas de poemas ou obras escritas em verso.",
        "6": "Romance: histórias centradas no relacionamento amoroso entre personagens.",
        "7": "Jovem adulto: ficção voltada a adolescentes, com protagonistas jovens e temas de amadurecimento.",
    },
    "dblp": {
        "0": "Visão computacional: análise e interpretação de imagens e vídeos (reconhecimento, detecção, segmentação, rastreamento).",
        "1": "Linguística computacional: processamento automático de linguagem natural, de texto ou de fala.",
        "2": "Engenharia biomédica: computação e engenharia aplicadas à medicina e à biologia (sinais fisiológicos, imagens médicas, dispositivos).",
        "3": "Engenharia de software: processos, métodos e ferramentas de desenvolvimento, teste, manutenção e verificação de software.",
        "4": "Computação gráfica: modelagem, renderização, animação e visualização de objetos e cenas.",
        "5": "Mineração de dados: descoberta de padrões em grandes bases de dados (agrupamento, regras de associação, recomendação).",
        "6": "Segurança e criptografia: proteção de sistemas e dados, protocolos criptográficos, ataques e privacidade.",
        "7": "Processamento de sinais: análise, filtragem, codificação e transmissão de sinais (áudio, comunicações, sensores).",
        "8": "Robótica: percepção, planejamento, controle e navegação de robôs e sistemas autônomos.",
        "9": "Teoria da computação: algoritmos, complexidade, lógica e fundamentos matemáticos da computação.",
    },
}

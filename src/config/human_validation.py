"""
Human Validation Configurations - Amostragem incremental e materiais da validação humana
"""

# =============================================================================
# AMOSTRAGEM
# =============================================================================

HUMAN_VALIDATION_SEED = 42

# Documentos por grupo de concordância na 1ª rodada e em cada rodada seguinte
INITIAL_ROUND_SIZE = 30
INCREMENT_SIZE = 10

# Quando um grupo para, suas vagas vão para os ativos: a carga da rodada fica em
# INCREMENT_SIZE x 3 grupos (3 ativos: 10 cada; 2 ativos: 15 cada; 1 ativo: 30)
REDISTRIBUTE_STOPPED_QUOTA = True

# Mínimo por classe na 1ª rodada (o estimador de variância estratificado exige n_h >= 2)
MIN_PER_CLASS_FIRST_ROUND = 2

# =============================================================================
# ESTIMAÇÃO (critério de parada)
# =============================================================================

# n mínimo por grupo antes de checar a parada (regra do TLC, Merlo et al., CIKM'25)
MIN_SAMPLE_FOR_STOPPING = 30

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

# Avaliadores de cada dataset (3 por dataset; a maioria é calculada entre eles); logins anônimos
EVALUATORS_BY_DATASET = {
    "books": ["books_1", "books_2", "books_3"],
    "dblp": ["dblp_1", "dblp_2", "dblp_3"],
}
# Quem é quem: aparece só na tela do administrador
EVALUATOR_NAMES = {
    "books_1": "Serufo", "books_2": "Botelho", "books_3": "Gestefane",
    "dblp_1": "Kenia", "dblp_2": "Liziane", "dblp_3": "Luciano",
}
# Todos os avaliadores (login e códigos de acesso)
EVALUATORS = [name for names in EVALUATORS_BY_DATASET.values() for name in names]

# Opção extra de rotulo_escolhido para texto sem informação suficiente / fora da taxonomia
INSUFFICIENT_INFO_OPTION = "não é possível decidir com este texto"

# =============================================================================
# INTERFACE WEB
# =============================================================================

# Avisos do guia mostrados ao avaliador (vieram do feedback do piloto)
GUIDE_NOTES = {
    "books": [
        "Avalie só pelo que está escrito no texto. Se você só reconheceria o rótulo por já conhecer o livro, "
        "use \"não é possível decidir com este texto\".",
        "Marcações como \"#2\", \"vol. 3\" ou \"(série, #1)\" indicam livros de uma série e aparecem em qualquer "
        "gênero; sozinhas, não indicam quadrinhos.",
        "\"romance\" aqui é o gênero de histórias de amor, e não \"romance\" no sentido de livro de ficção.",
        "Muitos livros combinam gêneros (por exemplo, mistério para jovens). Escolha o que melhor descreve o texto "
        "na etapa 1 e indique o outro na etapa 2.",
    ],
    "dblp": [
        "Avalie só pelo que está escrito no texto (título e resumo), e não pelo que você sabe sobre o artigo ou o evento.",
        "Se o artigo cruza áreas (por exemplo, visão computacional e computação gráfica), escolha a área principal na "
        "etapa 1 e indique a outra na etapa 2.",
    ],
}

# Datasets conduzidos pela interface; a data de cada experimento vem do .env
# (HV_EXPERIMENT_<DATASET>, ex.: HV_EXPERIMENT_BOOKS=2026-09-13_07-19-54)
VALIDATION_DATASETS = ["books", "dblp"]

# Banco SQLite da interface (respostas, rodadas e sessões), em data/validacao_humana/
INTERFACE_DB_NAME = "validacao_humana.db"
# Ao fechar uma rodada, abre a próxima sozinha se algum grupo ainda não parou
AUTO_START_NEXT_ROUND = True
# Com os três completos, a rodada fecha sozinha após esta janela sem novas respostas
# (tempo para revisão; 0 desliga o fechamento automático)
REVIEW_WINDOW_MINUTES = 5
AUTO_CLOSE_CHECK_SECONDS = 30
# Aviso por email ao fechar uma rodada (credenciais no .env: HV_NOTIFY_EMAIL + HV_RESEND_* ou HV_SMTP_*)
SMTP_DEFAULT_HOST = "smtp.gmail.com"
SMTP_DEFAULT_PORT = 587
# Resend (HTTPS): usado quando HV_RESEND_API_KEY existe, pois o Railway Hobby bloqueia SMTP.
# O remetente de teste só entrega para o email dono da conta Resend
RESEND_API_URL = "https://api.resend.com/emails"
RESEND_DEFAULT_SENDER = "Validação humana <onboarding@resend.dev>"
INTERFACE_HOST = "0.0.0.0"
INTERFACE_PORT = 8001
# Códigos de acesso vêm do .env (nunca do repositório):
#   HV_ADMIN_CODE=<código do administrador>
#   HV_EVALUATOR_<LOGIN>_CODE=<código>  (uma variável por avaliador, ex.: HV_EVALUATOR_BOOKS_1_CODE)
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
        "1": "Quadrinhos, graphic novels e mangás: obras narradas por arte sequencial (desenhos e balões). "
             "Só escolha quando o texto indicar esse formato; número de volume ou de série sozinho não indica quadrinhos.",
        "2": "Fantasia e sobrenatural: mundos mágicos, criaturas míticas, vampiros, lobisomens, bruxas ou poderes sobrenaturais.",
        "3": "História e biografia: não ficção sobre fatos históricos ou sobre a vida de pessoas reais (biografias, memórias).",
        "4": "Crime, mistério e suspense: investigações, crimes, detetives, conspirações e tramas de tensão.",
        "5": "Poesia: coletâneas de poemas ou obras escritas em verso.",
        "6": "Romance como gênero romântico (love story): a trama gira em torno de um relacionamento amoroso. "
             "Não confundir com \"romance\" no sentido de livro de ficção em geral.",
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

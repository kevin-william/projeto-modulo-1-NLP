import os

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRETORIO_INPUT = os.path.join(DIRETORIO_BASE, "input")
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
DIRETORIO_PLOTS = os.path.join(DIRETORIO_SAIDA, "plots")
DIRETORIO_DISPLACY = os.path.join(DIRETORIO_SAIDA, "displacy")
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "fase4_pipeline.log")

# Parquet gerado pela Fase 1 (colunas: token, processado, lema, entidade, rotulo_entidade, pos, id_artigo, is_stop).
_PARQUET_INPUT_LOCAL = os.path.join(DIRETORIO_INPUT, "100-artigos_anotacao_lg_lemmatizacao_palavra.parquet")
_PARQUET_FASE1 = os.path.join(
    DIRETORIO_BASE, "..", "fase1", "output",
    "100-artigos_anotacao_lg_lemmatizacao_palavra.parquet",
)
CAMINHO_PARQUET_ENTRADA = _PARQUET_INPUT_LOCAL if os.path.exists(_PARQUET_INPUT_LOCAL) else _PARQUET_FASE1

# Artefato .lpf2 gerado pela Fase 2.
_ARTEFATO_INPUT_LOCAL = os.path.join(DIRETORIO_INPUT, "fase2_artifact.lpf2")
_ARTEFATO_FASE2 = os.path.join(
    DIRETORIO_BASE, "..", "fase2", "output", "artifacts", "fase2_artifact.lpf2"
)
CAMINHO_ARTEFATO_FASE2 = _ARTEFATO_INPUT_LOCAL if os.path.exists(_ARTEFATO_INPUT_LOCAL) else _ARTEFATO_FASE2

# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
REGEX_EMAILS = r"[^\s]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+"
REGEX_URLS = r"https?://(?:www\.)?[a-zA-Z0-9\-_.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
REGEX_DATAS = r"\b\d{2}/\d{2}/\d{4}\b"
REGEX_CPFS = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
REGEX_VALORES = r"R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"
REGEX_CODIGOS = r"\b[A-Z]{2,6}-\d{3,6}\b"

# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------
MODELO_SPACY = "pt_core_news_lg"
MODELO_SPACY_FALLBACK = "pt_core_news_sm"
AMOSTRA_DISPLACY = 5
ENTIDADES_VALIDAS_GRAFO = {"PER", "ORG", "LOC", "GPE", "MISC", "PERSON", "ORGANIZATION", "GPE_LOC"}

# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------
MAX_LEVENSHTEIN_DISTANCE = 2
NORMALIZAR_CASE = True
REMOVER_ACENTOS = True

# ---------------------------------------------------------------------------
# Relacoes SVO
# ---------------------------------------------------------------------------
ENTIDADES_RELACAO = {"ORG", "PERSON", "GPE", "LOC", "PER"}
WINDOW_MAX_SENTENCAS = 5

# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------
MINIMO_FREQUENCIA_ENTIDADE = 2
NUMERO_MINIMO_NOS_GRAFO = 20
TOP_CENTRALIDADE = 10
METRICAS_CENTRALIDADE = ["betweenness", "degree", "eigenvector", "closeness"]

# ---------------------------------------------------------------------------
# Visualizacao
# ---------------------------------------------------------------------------
FIGSIZE_GRAFO = (18, 14)
FIGSIZE_BARRAS = (12, 8)
NODE_SIZE_FACTOR = 800
DPI_OUTPUT = 150

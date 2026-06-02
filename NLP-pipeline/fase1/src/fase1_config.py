import os
import random
import numpy as np

SEED_ALEATORIO = 42
random.seed(SEED_ALEATORIO)
np.random.seed(SEED_ALEATORIO)

DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "1100-artigos_wikipedia-formatados.txt")
CAMINHO_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "1100-artigos_wikipedia-formatados.txt")
DIRETORIO_INPUT = os.path.join(DIRETORIO_BASE, "input")
DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "nlp_1100-artigos_wikipedia-formatados-v001.log")
CAMINHO_PARQUET_SAIDA = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001.parquet")
CAMINHO_NUVEM_PALAVRAS = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001.png")
CAMINHO_ANALISE_VOCABULARIO = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados-v001.json")
CAMINHO_TABELA_COMPARACAO = os.path.join(DIRETORIO_SAIDA, "1100-artigos_wikipedia-formatados_comparacao_stemming_lematizacao.csv")

# CAMINHO_ENTRADA = os.path.join(DIRETORIO_BASE, "input", "artigos_wikipedia.txt")
# DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")
# CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "nlp_pipeline.log")
# CAMINHO_PARQUET_SAIDA = os.path.join(DIRETORIO_SAIDA, "anotacao_lg.parquet")
# CAMINHO_NUVEM_PALAVRAS = os.path.join(DIRETORIO_SAIDA, "wordcloud.png")
# CAMINHO_ANALISE_VOCABULARIO = os.path.join(DIRETORIO_SAIDA, "vocabulario_analise.json")

MODELO_SPACY = "pt_core_news_lg"
TAMANHO_LOTE = 5

MARCADOR_INICIO_ARTIGO = "===== ARTICLE START ====="
MARCADOR_FIM_ARTIGO = "===== ARTICLE END ====="

# Métodos de processamento de tokens: 'none', 'lemmatizacao', 'stemming'
# Pode ser uma lista com um ou múltiplos valores para execução sequencial
METODOS_PROCESSAMENTO_TOKENS = ['lemmatizacao']

# Método de stemming padrão: 'snowball' (produção, português) ou 'porter' (disponível para fins didáticos)
METODO_STEMMING = 'snowball'

# Fonte das stopwords: 'spacy', 'nltk' ou 'ambas'
FONTE_STOPWORDS = 'spacy'

# Stopwords extras a serem adicionadas às stopwords padrão do spaCy
STOPWORDS_EXTRAS = []
# STOPWORDS_EXTRAS = ['o', 'a', 'os', 'as', 'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas']

# Tokenização exclusivamente por palavra nesta etapa

# Filtro de tamanho mínimo: artigos com menos de N palavras são removidos antes do processamento
MINIMO_PALAVRAS_ARTIGO = 100

# Habilita/desabilita a remocao de stopwords durante a geracao do parquet de saida.
HABILITAR_REMOCAO_STOPWORDS = True

# POS tags permitidos no parquet de saida (whitelist). Lista vazia desabilita o filtro de POS.
POS_TAGS_PERMITIDOS = ["NOUN", "VERB", "ADJ", "ADV"]

# ---------------------------------------------------------------------------
# Configuracao da WordCloud (MVP)
# ---------------------------------------------------------------------------
# Largura da imagem final em pixels; aumente para mais detalhe visual.
LARGURA_NUVEM_PALAVRAS = 1200

# Altura da imagem final em pixels.
ALTURA_NUVEM_PALAVRAS = 600

# Limite maximo de palavras exibidas na nuvem.
MAXIMO_PALAVRAS_NUVEM = 70

# Paleta de cores do matplotlib usada pela WordCloud (ex.: viridis, plasma, magma).
PALETA_CORES_NUVEM = "magma"

# Cor de fundo da imagem da nuvem.
COR_FUNDO_NUVEM = "white"

# Tamanho minimo da fonte; valores muito baixos geram ruido visual.
TAMANHO_MINIMO_FONTE_NUVEM = 20

# Tamanho maximo da fonte; deve ser >= TAMANHO_MINIMO_FONTE_NUVEM.
TAMANHO_MAXIMO_FONTE_NUVEM = 160

# Semente da WordCloud para manter layout reproduzivel entre execucoes.
SEMENTE_NUVEM_PALAVRAS = SEED_ALEATORIO

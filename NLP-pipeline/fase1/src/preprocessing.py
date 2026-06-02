import re
import nltk
import spacy
import pandas as pd
from nltk.stem import PorterStemmer, SnowballStemmer
from fase1_config import MODELO_SPACY, STOPWORDS_EXTRAS, FONTE_STOPWORDS
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)

_stopwords_personalizadas = set()
_instancia_nlp = None

# _stemmer_porter: instância do Porter Stemmer disponível para fins didáticos.
# A função aplicar_stemming() aceita metodo='porter' para usar este stemmer
# em comparações com o Snowball (padrão em português).
_stemmer_porter = PorterStemmer()
_stemmer_snowball = SnowballStemmer(language="portuguese")


def inicializar_nltk():
    """
    Realiza o download automático dos recursos NLTK necessários para este módulo:
    - punkt e punkt_tab: tokenizadores de sentenças e palavras
    - stopwords: corpus de stopwords multilíngue
    """
    for recurso in ('punkt', 'punkt_tab', 'stopwords'):
        try:
            nltk.download(recurso, quiet=True)
        except Exception as exc:
            logger.warning("Nao foi possivel baixar recurso NLTK '%s': %s", recurso, exc)


def obter_instancia_nlp():
    global _instancia_nlp
    if _instancia_nlp is None:
        logger.info("Carregando modelo spaCy: %s", MODELO_SPACY)
        _instancia_nlp = spacy.load(MODELO_SPACY)
        logger.info("Modelo carregado com sucesso")
    return _instancia_nlp


def normalizar_texto(texto):
    """
    Normaliza o texto antes da tokenização:
    - Converte para letras minúsculas
    - Remove URLs (http/https e www)
    - Remove caracteres que não sejam letras (A-Z, À-ÿ), dígitos ou espaços
    - Colapsa múltiplos espaços em um único espaço

    Args:
        texto: Texto bruto a ser normalizado.

    Returns:
        Texto normalizado como string.
    """
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-zA-Z0-9À-ÿ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def aplicar_stemming(texto, metodo='snowball'):
    """
    Aplica stemming ao texto usando o método especificado.

    Args:
        texto: Texto para aplicar stemming.
        metodo: Método de stemming ('porter' ou 'snowball').

    Returns:
        Texto com stemming aplicado.
    """
    if metodo == 'porter':
        return _stemmer_porter.stem(texto)
    return _stemmer_snowball.stem(texto)


def obter_stopwords_nltk(idioma='portuguese'):
    """
    Retorna o conjunto de stopwords da biblioteca NLTK para o idioma especificado.

    Args:
        idioma: Nome do idioma aceito pelo NLTK (ex.: 'portuguese', 'english').

    Returns:
        Conjunto de stopwords como set de strings.
    """
    from nltk.corpus import stopwords as nltk_stopwords_corpus
    return set(nltk_stopwords_corpus.words(idioma))


def obter_stopwords(fonte=None):
    """
    Retorna o conjunto de stopwords para o idioma português.

    Args:
        fonte: Fonte das stopwords: 'spacy', 'nltk' ou 'ambas'.
               Se None, usa FONTE_STOPWORDS da configuração.

    Returns:
        Conjunto de stopwords como set de strings.
    """
    if fonte is None:
        fonte = FONTE_STOPWORDS

    stopwords_extras = set(palavra.lower() for palavra in STOPWORDS_EXTRAS) | _stopwords_personalizadas

    if fonte == 'nltk':
        return obter_stopwords_nltk() | stopwords_extras

    nlp = obter_instancia_nlp()
    spacy_sw = nlp.Defaults.stop_words

    if fonte == 'ambas':
        return spacy_sw | obter_stopwords_nltk() | stopwords_extras

    # padrão: 'spacy'
    return spacy_sw | stopwords_extras


def adicionar_stopwords_personalizadas(palavras):
    for palavra in palavras:
        _stopwords_personalizadas.add(palavra.lower())
    logger.info("Stopwords customizadas adicionadas: %s", palavras)


def remover_stopwords_personalizadas(palavras):
    for palavra in palavras:
        _stopwords_personalizadas.discard(palavra.lower())
    logger.info("Stopwords customizadas removidas: %s", palavras)


def tokenizar_por_tipo(texto_artigo, metodo_processamento='none'):
    """
    Tokeniza um artigo por palavra.

    Args:
        texto_artigo: Texto do artigo.
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.

    Returns:
        Dicionário com 'tokens'.
    """
    texto_normalizado = normalizar_texto(texto_artigo)
    nlp = obter_instancia_nlp()
    doc = nlp(texto_normalizado)

    tokens = []
    for token in doc:
        lema = token.lemma_
        if metodo_processamento == 'lemmatizacao':
            # O modelo pt_core_news_lg expande contrações (ex: "do" → "de o").
            # Lemas com espaço não são filtrados pelas stopwords simples, por isso
            # fazemos fallback para o token original (forma contraída é stopword válida).
            processado = token.text if ' ' in lema else lema
        elif metodo_processamento == 'stemming':
            processado = aplicar_stemming(lema)
        else:
            processado = token.text
        tokens.append({
            "texto": token.text,
            "lema": lema,
            "processado": processado,
            "pos": token.pos_,
            "tag": token.tag_,
            "dependencia": token.dep_,
            "eh_stopword": token.is_stop,
            "eh_pontuacao": token.is_punct,
            "eh_alfabetico": token.is_alpha,
            "formato": token.shape_,
        })
    return {"tokens": tokens}


def tokenizar_artigo(texto_artigo, metodo_processamento='none'):
    """
    Tokeniza um artigo por palavra com o método de processamento especificado.

    Args:
        texto_artigo: Texto do artigo.
        metodo_processamento: Método de processamento ('none', 'lemmatizacao' ou 'stemming').

    Returns:
        Dicionário com 'tokens'. Cada token contém o campo 'processado'
        com o valor resultante do método escolhido.
    """
    return tokenizar_por_tipo(texto_artigo, metodo_processamento=metodo_processamento)


def remover_stopwords_dos_tokens(tokens):
    stopwords_atuais = obter_stopwords()
    tokens_filtrados = [
        token for token in tokens
        if token.get("processado", token["texto"]).lower() not in stopwords_atuais
        and not token["eh_pontuacao"]
    ]
    return tokens_filtrados


# ---------------------------------------------------------------------------
# Tokenizadores NLTK — funções de demonstração didática
# Produzem listas de strings (sem enriquecimento linguístico do spaCy).
# ---------------------------------------------------------------------------

def tokenizar_sentencas(texto):
    """
    Tokeniza o texto em sentenças usando NLTK (sent_tokenize).
    Função de demonstração didática — produz lista de strings.

    Args:
        texto: Texto de entrada.

    Returns:
        Lista de sentenças como strings.
    """
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(texto, language='portuguese')


def tokenizar_palavras_nltk(texto):
    """
    Tokeniza o texto em palavras usando NLTK (word_tokenize).
    Função de demonstração didática — produz lista de strings.

    Args:
        texto: Texto de entrada.

    Returns:
        Lista de tokens como strings.
    """
    from nltk.tokenize import word_tokenize
    return word_tokenize(texto, language='portuguese')


def tokenizar_casual(texto):
    """
    Tokeniza o texto de forma casual usando NLTK (casual_tokenize).
    Adequado para textos informais, redes sociais e afins.
    Função de demonstração didática — produz lista de strings.

    Args:
        texto: Texto de entrada.

    Returns:
        Lista de tokens como strings.
    """
    from nltk.tokenize import casual_tokenize
    return casual_tokenize(texto)


# ---------------------------------------------------------------------------
# Tabela comparativa stemming vs lematização (padrão didático do professor)
# ---------------------------------------------------------------------------

def gerar_tabela_comparacao_stemming_lematizacao(texto):
    """
    Gera um DataFrame comparando stemming e lematização para cada palavra
    alfabética do texto, seguindo o padrão didático do professor.

    Colunas: palavra | stemming | lematização | classe_gramatical

    Args:
        texto: Texto de entrada a ser analisado.

    Returns:
        DataFrame pandas com as colunas acima.
    """
    nlp = obter_instancia_nlp()
    doc = nlp(texto)
    comparacao = []
    for token in doc:
        if token.is_alpha:
            palavra = token.text.lower()
            comparacao.append({
                "palavra": palavra,
                "stemming": _stemmer_snowball.stem(palavra),
                "lematização": token.lemma_,
                "classe_gramatical": token.pos_,
            })
    return pd.DataFrame(comparacao)

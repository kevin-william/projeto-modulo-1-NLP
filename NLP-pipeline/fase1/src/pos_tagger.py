import pandas as pd
from preprocessing import obter_instancia_nlp, aplicar_stemming, normalizar_texto
from fase1_config import TAMANHO_LOTE, HABILITAR_REMOCAO_STOPWORDS, POS_TAGS_PERMITIDOS
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _aplicar_metodo(lema, texto, metodo_processamento):
    """Retorna o valor processado de um token conforme o método escolhido."""
    if metodo_processamento == 'lemmatizacao':
        return lema
    if metodo_processamento == 'stemming':
        return aplicar_stemming(lema)
    return texto


def _extrair_entidade(token):
    """Retorna (entidade, rotulo_entidade) para um token spaCy."""
    if token.ent_iob_ in ("B", "I"):
        return token.ent_type_ or "", token.ent_type_ or ""
    return "", ""


def _construir_linha(id_artigo, id_token, texto, lema, processado, pos, tag,
                     dep, cabeca, entidade, rotulo_entidade, tipo_tokenizacao, artigo,
                     formato="", is_stop=False):
    """Monta o dicionário de uma linha do DataFrame final."""
    return {
        "id_artigo": id_artigo,
        "id_token": id_token,
        "token": texto,
        "pos": pos,
        "tag": tag,
        "lema": lema,
        "processado": processado,
        "is_stop": is_stop,
        "relacao_dependencia": dep,
        "token_cabeca": cabeca,
        "entidade": entidade,
        "rotulo_entidade": rotulo_entidade,
        "tipo_tokenizacao": tipo_tokenizacao,
        "titulo": artigo["titulo"],
        "url": artigo["url"],
        "formato": formato,
    }


def _processar_palavras(id_artigo, artigo, documento, metodo_processamento,
                        habilitar_stopwords=False, pos_permitidos=None):
    """Gera linhas com tokenização por palavra (unigrama)."""
    linhas = []
    id_token = 1
    for token in documento:
        if pos_permitidos:
            if token.pos_ not in pos_permitidos:
                continue
        elif token.is_punct:
            continue

        if habilitar_stopwords and token.is_stop:
            continue

        lema = token.lemma_
        processado = _aplicar_metodo(lema, token.text, metodo_processamento)
        entidade, rotulo_entidade = _extrair_entidade(token)
        linhas.append(_construir_linha(
            id_artigo, id_token, token.text, lema, processado,
            token.pos_, token.tag_, token.dep_,
            token.head.text if token.head else "",
            entidade, rotulo_entidade, 'palavra', artigo, token.shape_,
            is_stop=token.is_stop,
        ))
        id_token += 1
    return linhas


def processar_lote_artigos(artigos, metodo_processamento='none',
                           habilitar_stopwords=None, pos_permitidos=None):
    """
    Processa uma lista de artigos em lote com spaCy, tokenizando por palavra.
    Aplica normalização textual antes do processamento linguístico.

    Args:
        artigos: Lista de dicionários de artigos (titulo, url, conteudo).
        metodo_processamento: 'none', 'lemmatizacao' ou 'stemming'.
        habilitar_stopwords: Remove tokens marcados como stopword pelo modelo.
            None usa o valor de HABILITAR_REMOCAO_STOPWORDS do config.
        pos_permitidos: Whitelist de POS tags a manter. None usa POS_TAGS_PERMITIDOS do config.
            Lista vazia desabilita o filtro de POS.

    Returns:
        DataFrame pandas com todos os tokens anotados.
    """
    nlp = obter_instancia_nlp()
    textos = [normalizar_texto(artigo["conteudo"]) for artigo in artigos]

    _habilitar_stopwords = HABILITAR_REMOCAO_STOPWORDS if habilitar_stopwords is None else habilitar_stopwords
    _pos_permitidos = POS_TAGS_PERMITIDOS if pos_permitidos is None else pos_permitidos

    logger.info(
        "Processando %d artigos em lote (batch_size=%d, metodo=%s, stopwords=%s, pos_filter=%s)...",
        len(textos), TAMANHO_LOTE, metodo_processamento,
        _habilitar_stopwords, _pos_permitidos or "desabilitado",
    )

    linhas = []
    for id_artigo, (artigo, documento) in enumerate(
        zip(artigos, nlp.pipe(textos, batch_size=TAMANHO_LOTE)), start=1
    ):
        novas_linhas = _processar_palavras(
            id_artigo, artigo, documento, metodo_processamento,
            habilitar_stopwords=_habilitar_stopwords,
            pos_permitidos=_pos_permitidos,
        )
        linhas.extend(novas_linhas)
        logger.info(
            "Artigo %d processado: '%s' -> %d tokens",
            id_artigo, artigo["titulo"], len(novas_linhas),
        )

    dataframe = pd.DataFrame(linhas)
    logger.info("DataFrame criado: %d linhas, %d colunas", len(dataframe), len(dataframe.columns))
    return dataframe

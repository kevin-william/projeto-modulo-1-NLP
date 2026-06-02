import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import carregar_artigos, filtrar_artigos_por_tamanho
from preprocessing import (
    tokenizar_artigo,
    remover_stopwords_dos_tokens,
    obter_stopwords,
)
from vocab_analysis import analisar_vocabulario
from collections import Counter


def _tokenizar_e_filtrar(conteudo, metodo='lemmatizacao'):
    resultado = tokenizar_artigo(conteudo, metodo_processamento=metodo)
    tokens = resultado["tokens"]
    tokens_filtrados = remover_stopwords_dos_tokens(tokens)
    return tokens, tokens_filtrados


def test_pipeline_minima_completa():
    artigos = carregar_artigos()
    tokens, tokens_filtrados = _tokenizar_e_filtrar(artigos[0]["conteudo"])

    contador_lema_bruto = Counter(token["processado"].lower() for token in tokens if token["processado"].strip())
    contador_lema_filtrado = Counter(token["processado"].lower() for token in tokens_filtrados if token["processado"].strip())

    assert len(contador_lema_bruto) > 0
    assert len(contador_lema_filtrado) > 0
    assert len(contador_lema_filtrado) <= len(contador_lema_bruto)


def test_pipeline_minima_stemming():
    artigos = carregar_artigos()
    tokens, tokens_filtrados = _tokenizar_e_filtrar(artigos[0]["conteudo"], metodo='stemming')

    contador_bruto = Counter(token["processado"].lower() for token in tokens if token["processado"].strip())
    contador_filtrado = Counter(token["processado"].lower() for token in tokens_filtrados if token["processado"].strip())

    assert len(contador_bruto) > 0
    assert len(contador_filtrado) <= len(contador_bruto)


def test_efeito_remocao_stopwords():
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    tokens, tokens_filtrados = _tokenizar_e_filtrar(conteudo)
    stopwords_atuais = obter_stopwords()

    stopwords_apos_filtro = sum(
        1 for token in tokens_filtrados if token["processado"].lower() in stopwords_atuais
    )
    assert stopwords_apos_filtro == 0, "Nao deveria haver stopwords apos remocao"
    assert len(tokens_filtrados) <= len(tokens), "Filtro deveria reduzir ou manter tokens"


def test_saida_analise_vocabulario():
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    resultado = tokenizar_artigo(conteudo, metodo_processamento='lemmatizacao')
    tokens = resultado["tokens"]

    tokens_info = [
        {"texto": token["texto"], "lema": token["processado"], "pos": token["pos"]}
        for token in tokens
    ]
    tokens_filtrados = remover_stopwords_dos_tokens(tokens)
    tokens_filtrados_info = [
        {"texto": token["texto"], "lema": token["processado"], "pos": token["pos"]}
        for token in tokens_filtrados
    ]

    analise = analisar_vocabulario(tokens_info, tokens_filtrados_info)
    assert "quantidade_vocabulario_bruto" in analise
    assert "quantidade_vocabulario_filtrado" in analise
    assert "percentual_reducao_vocabulario" in analise
    assert analise["quantidade_vocabulario_filtrado"] <= analise["quantidade_vocabulario_bruto"]


# ---------------------------------------------------------------------------
# Testes de filtro mínimo de palavras na pipeline
# ---------------------------------------------------------------------------

def test_filtro_minimo_nao_remove_artigos_com_limiar_zero():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=0)
    assert len(validos) == len(artigos)
    assert len(removidos) == 0


def test_filtro_minimo_remove_artigos_com_limiar_alto():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=100000)
    assert len(removidos) > 0
    assert len(validos) + len(removidos) == len(artigos)


# ---------------------------------------------------------------------------
# Testes de tokens brutos vs filtrados (Bug #2 regression)
# ---------------------------------------------------------------------------

def test_tokens_brutos_contem_mais_tokens_que_filtrados():
    """tokens_brutos_info deve incluir stopwords, portanto ter mais tokens que tokens_filtrados."""
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    resultado = tokenizar_artigo(conteudo, metodo_processamento='none')
    todos_tokens = resultado["tokens"]
    stopwords_atuais = obter_stopwords()

    tokens_brutos = todos_tokens
    tokens_filtrados = [
        t for t in todos_tokens
        if t.get("processado", t["texto"]).lower() not in stopwords_atuais
        and not t["eh_pontuacao"]
    ]

    assert len(tokens_brutos) > len(tokens_filtrados), (
        "tokens_brutos deve conter stopwords e ter mais tokens que tokens_filtrados"
    )


def test_tokens_brutos_contem_stopwords():
    """tokens_brutos_info deve conter ao menos uma stopword do corpus."""
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]
    resultado = tokenizar_artigo(conteudo, metodo_processamento='none')
    todos_tokens = resultado["tokens"]
    stopwords_atuais = obter_stopwords()

    stopwords_encontradas = [
        t for t in todos_tokens
        if t.get("processado", t["texto"]).lower() in stopwords_atuais
    ]
    assert len(stopwords_encontradas) > 0, "Corpus deve conter stopwords nos tokens brutos"


def test_lemmatizacao_processado_difere_de_none():
    """Campo 'processado' deve diferir entre os métodos lemmatizacao e none para o mesmo corpus."""
    artigos = carregar_artigos()
    conteudo = artigos[0]["conteudo"]

    resultado_lem = tokenizar_artigo(conteudo, metodo_processamento='lemmatizacao')
    resultado_none = tokenizar_artigo(conteudo, metodo_processamento='none')

    processados_lem = {t["processado"] for t in resultado_lem["tokens"]}
    processados_none = {t["processado"] for t in resultado_none["tokens"]}

    assert processados_lem != processados_none, (
        "O conjunto de tokens processados deve ser diferente entre lemmatizacao e none"
    )

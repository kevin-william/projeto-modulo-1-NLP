import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from pos_tagger import processar_lote_artigos

_ARTIGOS = [
    {
        "titulo": "Teste",
        "url": "https://example.com",
        "conteudo": "O processamento de linguagem natural é uma área fascinante da inteligência artificial.",
    }
]


def test_processar_lote_artigos():
    dataframe = processar_lote_artigos(_ARTIGOS)
    assert len(dataframe) > 0
    colunas_esperadas = [
        "id_artigo", "id_token", "token", "pos", "tag",
        "lema", "processado", "relacao_dependencia", "token_cabeca", "entidade", "rotulo_entidade",
        "tipo_tokenizacao", "titulo", "url", "formato",
    ]
    for coluna in colunas_esperadas:
        assert coluna in dataframe.columns, f"Coluna '{coluna}' ausente"
    assert dataframe["id_artigo"].iloc[0] == 1
    assert dataframe["titulo"].iloc[0] == "Teste"
    assert dataframe["url"].iloc[0] == "https://example.com"


def test_coluna_tipo_tokenizacao_presente():
    dataframe = processar_lote_artigos(_ARTIGOS)
    assert "tipo_tokenizacao" in dataframe.columns


def test_coluna_tipo_tokenizacao_valor_padrao():
    dataframe = processar_lote_artigos(_ARTIGOS)
    assert (dataframe["tipo_tokenizacao"] == "palavra").all()


def test_coluna_processado_none():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='none')
    assert (dataframe["processado"] == dataframe["token"]).all(), \
        "Com method='none', 'processed' deve ser igual ao token original"


def test_coluna_processado_lemmatizacao():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='lemmatizacao')
    assert (dataframe["processado"] == dataframe["lema"]).all(), \
        "Com method='lemmatizacao', 'processed' deve ser igual ao lemma"


def test_coluna_processado_stemming():
    dataframe = processar_lote_artigos(_ARTIGOS, metodo_processamento='stemming')
    dataframe_alfabetico = dataframe[dataframe["token"].str.isalpha()]
    houve_mudanca = (dataframe_alfabetico["processado"] != dataframe_alfabetico["lema"]).any()
    assert houve_mudanca, "Stemming deveria alterar ao menos um token"


def test_pos_tags_nao_vazio():
    artigos = [
        {
            "titulo": "Teste 2",
            "url": "https://example.com/2",
            "conteudo": "O gato preto correu rapidamente.",
        }
    ]
    dataframe = processar_lote_artigos(artigos)
    valores_pos = dataframe["pos"].unique()
    assert len(valores_pos) > 0, "Deveria ter POS tags"
    assert "VERB" in valores_pos or "NOUN" in valores_pos


def test_lema_presente():
    artigos = [
        {
            "titulo": "Teste 3",
            "url": "https://example.com/3",
            "conteudo": "Os computadores processam dados.",
        }
    ]
    dataframe = processar_lote_artigos(artigos)
    lemas = dataframe["lema"].dropna().tolist()
    assert len(lemas) > 0
    lemas_minusculos = [lema.lower() for lema in lemas]
    assert any("computador" in lema for lema in lemas_minusculos)


# ---------------------------------------------------------------------------
# Testes de tipos de tokenização
# ---------------------------------------------------------------------------

def test_tipo_tokenizacao_sempre_palavra():
    """Tokenização é sempre por palavra; coluna tipo_tokenizacao deve ser 'palavra'."""
    dataframe = processar_lote_artigos(_ARTIGOS)
    assert (dataframe["tipo_tokenizacao"] == "palavra").all()


# ---------------------------------------------------------------------------
# Testes de filtros: stopwords e POS
# ---------------------------------------------------------------------------

def test_stopword_removal_habilitado():
    """Tokens marcados como stopword pelo modelo nao devem aparecer com habilitar_stopwords=True."""
    artigos = [
        {
            "titulo": "Stop",
            "url": "https://example.com",
            "conteudo": "O processamento de linguagem natural é uma área fascinante.",
        }
    ]
    dataframe = processar_lote_artigos(artigos, habilitar_stopwords=True, pos_permitidos=[])
    from preprocessing import obter_stopwords
    stopwords_atuais = obter_stopwords()
    tokens_no_df = set(dataframe["token"].str.lower().tolist())
    assert not tokens_no_df.intersection(stopwords_atuais), (
        f"Stopwords encontradas no DataFrame: {tokens_no_df.intersection(stopwords_atuais)}"
    )


def test_pos_filter_habilitado():
    """Somente os POS especificados devem aparecer no DataFrame com pos_permitidos configurado."""
    artigos = [
        {
            "titulo": "POS",
            "url": "https://example.com",
            "conteudo": "O gato preto correu rapidamente pelo jardim.",
        }
    ]
    pos_alvo = ["NOUN", "VERB", "ADJ", "ADV"]
    dataframe = processar_lote_artigos(artigos, habilitar_stopwords=False, pos_permitidos=pos_alvo)
    pos_no_df = set(dataframe["pos"].unique().tolist())
    pos_inesperados = pos_no_df - set(pos_alvo)
    assert not pos_inesperados, f"POS inesperados no DataFrame: {pos_inesperados}"
    assert len(dataframe) > 0


def test_sem_filtros_contem_stopwords():
    """Sem filtros ativos, stopwords devem aparecer no DataFrame (comportamento original).
    Nota: PUNCT nunca aparece pois normalizar_texto remove pontuacao antes do spaCy."""
    artigos = [
        {
            "titulo": "Raw",
            "url": "https://example.com",
            "conteudo": "O gato preto correu rapidamente",
        }
    ]
    dataframe = processar_lote_artigos(artigos, habilitar_stopwords=False, pos_permitidos=[])
    from preprocessing import obter_stopwords
    stopwords_atuais = obter_stopwords()
    tokens_lower = set(dataframe["token"].str.lower().tolist())
    assert tokens_lower.intersection(stopwords_atuais), (
        "Deveria haver stopwords no DataFrame quando habilitar_stopwords=False"
    )


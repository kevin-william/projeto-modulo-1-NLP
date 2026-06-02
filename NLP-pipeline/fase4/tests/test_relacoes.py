"""
Testes para relacoes.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd


def _spacy_disponivel():
    try:
        from ner_analysis import carregar_modelo_spacy
        carregar_modelo_spacy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_extrair_triplas_svo_retorna_lista():
    from ner_analysis import carregar_modelo_spacy
    from relacoes import extrair_triplas_svo
    nlp = carregar_modelo_spacy()
    doc = nlp("A Petrobras comprou a empresa Google.")
    triplas = extrair_triplas_svo(doc)
    assert isinstance(triplas, list)


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_extrair_coocorrencias_sentenca():
    from ner_analysis import carregar_modelo_spacy
    from relacoes import extrair_coocorrencias_sentenca
    nlp = carregar_modelo_spacy()
    doc = nlp("A Petrobras e a Microsoft atuam no mercado.")
    cooc = extrair_coocorrencias_sentenca(doc)
    assert isinstance(cooc, list)


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel")
def test_extrair_relacoes_corpus_retorna_dataframe():
    from ner_analysis import carregar_modelo_spacy
    from relacoes import extrair_relacoes_corpus
    nlp = carregar_modelo_spacy()
    docs = [
        "A Petrobras anunciou lucros no Rio de Janeiro.",
        "O Google comprou empresa em São Paulo.",
    ]
    df = extrair_relacoes_corpus(docs, nlp)
    assert isinstance(df, pd.DataFrame)


def test_construir_arestas_grafo_colunas():
    from relacoes import construir_arestas_grafo
    df_relacoes = pd.DataFrame({
        "sujeito": ["Petrobras", "Google", "Petrobras"],
        "tipo_sujeito": ["ORG", "ORG", "ORG"],
        "verbo": ["anunciar", "comprar", "anunciar"],
        "objeto": ["Rio", "Kaggle", "Sao Paulo"],
        "tipo_objeto": ["LOC", "ORG", "LOC"],
        "tipo": ["svo", "svo", "svo"],
        "documento_id": [0, 0, 1],
    })
    df_arestas = construir_arestas_grafo(df_relacoes)
    assert "source" in df_arestas.columns
    assert "target" in df_arestas.columns
    assert "frequencia" in df_arestas.columns


def test_construir_arestas_grafo_df_vazio():
    from relacoes import construir_arestas_grafo
    df_vazio = pd.DataFrame(columns=["sujeito", "tipo_sujeito", "verbo", "objeto", "tipo_objeto", "tipo"])
    resultado = construir_arestas_grafo(df_vazio)
    assert "source" in resultado.columns
    assert "target" in resultado.columns


def test_filtrar_relacoes_relevantes():
    from relacoes import filtrar_relacoes_relevantes
    df = pd.DataFrame({
        "source": ["A", "B", "C"],
        "target": ["B", "C", "D"],
        "relation": ["v", "v", "v"],
        "frequencia": [5, 1, 3],
    })
    filtrado = filtrar_relacoes_relevantes(df, min_freq=2)
    assert all(filtrado["frequencia"] >= 2)
    assert len(filtrado) == 2

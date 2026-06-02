"""
Testes para grafo_conhecimento.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd


def _criar_grafo_teste():
    from grafo_conhecimento import GrafoConhecimento
    gc = GrafoConhecimento()
    # Cria arestas suficientes para ter >= 20 nos
    pares = [
        ("Petrobras", "Rio de Janeiro"), ("Petrobras", "Lula"),
        ("Microsoft", "Google"), ("Google", "Sao Paulo"),
        ("Lula", "Brasilia"), ("Petrobras", "Microsoft"),
        ("Amazon", "Seattle"), ("Apple", "California"),
        ("Tesla", "Elon Musk"), ("Meta", "Zuckerberg"),
        ("BMW", "Alemanha"), ("Toyota", "Japao"),
        ("Samsung", "Coreia"), ("Huawei", "China"),
        ("Oracle", "Larry Ellison"), ("IBM", "Nova York"),
        ("Intel", "Santa Clara"), ("AMD", "Sunnyvale"),
        ("Nvidia", "Jensen Huang"), ("SpaceX", "Elon Musk"),
    ]
    df_arestas = pd.DataFrame(
        [(s, t, "relaciona", 1) for s, t in pares],
        columns=["source", "target", "relation", "frequencia"],
    )
    gc.construir_grafo_entidade_entidade(df_arestas)
    tipos = ["ORG"] * 30
    entidades_lista = list(set([s for s, _ in pares] + [t for _, t in pares]))
    df_ents = pd.DataFrame({
        "texto": entidades_lista,
        "tipo": ["ORG"] * len(entidades_lista),
        "fonte": ["spacy"] * len(entidades_lista),
    })
    gc.adicionar_nos_entidades(df_ents)
    return gc


def test_grafo_numero_minimo_nos():
    gc = _criar_grafo_teste()
    from fase4_config import NUMERO_MINIMO_NOS_GRAFO
    assert gc.grafo.number_of_nodes() >= NUMERO_MINIMO_NOS_GRAFO


def test_calcular_centralidade_retorna_dfs():
    gc = _criar_grafo_teste()
    centralidade = gc.calcular_centralidade()
    assert isinstance(centralidade, dict)
    assert len(centralidade) > 0
    for metrica, df in centralidade.items():
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "entidade" in df.columns
            assert "centralidade" in df.columns


def test_obter_nos_mais_centrais():
    gc = _criar_grafo_teste()
    gc.calcular_centralidade()
    top = gc.obter_nos_mais_centrais(top_n=5)
    assert isinstance(top, pd.DataFrame)
    assert len(top) <= 5


def test_obter_ranking_consolidado():
    gc = _criar_grafo_teste()
    gc.calcular_centralidade()
    ranking = gc.obter_ranking_consolidado(top_n=5)
    assert isinstance(ranking, pd.DataFrame)
    assert "entidade" in ranking.columns
    assert "rank_medio" in ranking.columns


def test_responder_pergunta_analitica_retorna_string():
    gc = _criar_grafo_teste()
    gc.calcular_centralidade()
    resposta = gc.responder_pergunta_analitica()
    assert isinstance(resposta, str)
    assert len(resposta) > 0


def test_detectar_comunidades_retorna_dict():
    gc = _criar_grafo_teste()
    comunidades = gc.detectar_comunidades()
    assert isinstance(comunidades, dict)
    assert len(comunidades) >= 1


def test_exportar_grafo(tmp_path):
    gc = _criar_grafo_teste()
    gc.calcular_centralidade()
    caminho_arestas = str(tmp_path / "edges.csv")
    caminho_nos = str(tmp_path / "nos.csv")
    gc.exportar_grafo(caminho_arestas, caminho_nos)
    assert os.path.exists(caminho_arestas)
    assert os.path.exists(caminho_nos)
    df_edges = pd.read_csv(caminho_arestas)
    df_nos = pd.read_csv(caminho_nos)
    assert "source" in df_edges.columns
    assert "entidade" in df_nos.columns


def test_construir_grafo_vazio():
    from grafo_conhecimento import GrafoConhecimento
    gc = GrafoConhecimento()
    df_vazio = pd.DataFrame(columns=["source", "target", "relation", "frequencia"])
    g = gc.construir_grafo_entidade_entidade(df_vazio)
    assert g.number_of_nodes() == 0


def test_gerar_resumo_interpretativo():
    gc = _criar_grafo_teste()
    gc.calcular_centralidade()
    gc.detectar_comunidades()
    arestas = gc.obter_arestas_mais_frequentes()
    resumo = gc.gerar_resumo_interpretativo(gc.centralidade, arestas)
    assert isinstance(resumo, str)
    assert len(resumo) > 0

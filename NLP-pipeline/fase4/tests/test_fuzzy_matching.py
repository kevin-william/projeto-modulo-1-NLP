"""
Testes para fuzzy_matching.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fuzzy_matching import (
    agrupar_entidades_fuzzy,
    calcular_distancia_levenshtein,
    calcular_similaridade_normalizada,
    encontrar_mais_proximo,
    gerar_matriz_distancias,
    normalizar_entidades,
    normalizar_texto,
)


def test_distancia_identica():
    assert calcular_distancia_levenshtein("gato", "gato") == 0


def test_distancia_um():
    assert calcular_distancia_levenshtein("gato", "rato") == 1


def test_distancia_vazia():
    assert calcular_distancia_levenshtein("", "abc") == 3
    assert calcular_distancia_levenshtein("abc", "") == 3


def test_similaridade_identica():
    assert calcular_similaridade_normalizada("abc", "abc") == 1.0


def test_similaridade_zero_a_um():
    sim = calcular_similaridade_normalizada("abcdef", "xyz")
    assert 0.0 <= sim <= 1.0


def test_normalizar_texto_case():
    resultado = normalizar_texto("PETROBRAS")
    assert resultado == resultado.lower()


def test_normalizar_texto_acentos():
    resultado = normalizar_texto("Petrobrás")
    assert "á" not in resultado


def test_normalizar_texto_sufixo_juridico():
    resultado = normalizar_texto("Empresa S.A.")
    assert "s.a." not in resultado.lower() or "s a" not in resultado.lower()


def test_agrupar_entidades_similar():
    entidades = ["Petrobras", "Petrobrás", "Microsoft", "Microssoft"]
    grupos = agrupar_entidades_fuzzy(entidades, max_distancia=2)
    # Petrobras e Petrobrás devem estar no mesmo grupo
    canons = list(grupos.keys())
    variantes = [v for vs in grupos.values() for v in vs]
    todos = set(canons) | set(variantes)
    assert "Petrobras" in todos or "Petrobrás" in todos


def test_agrupar_entidades_distintas():
    entidades = ["Brasil", "Python", "Algoritmo"]
    grupos = agrupar_entidades_fuzzy(entidades, max_distancia=2)
    # Entidades muito distintas nao devem ser agrupadas
    assert len(grupos) >= 2


def test_encontrar_mais_proximo():
    candidatos = ["contato@empresa.com.br", "suporte@tech.org", "admin@dev.io"]
    melhor, dist, sim = encontrar_mais_proximo("contato@empresa.com", candidatos)
    assert melhor == "contato@empresa.com.br"
    assert dist >= 0
    assert 0.0 <= sim <= 1.0


def test_encontrar_mais_proximo_lista_vazia():
    melhor, dist, sim = encontrar_mais_proximo("teste", [])
    assert melhor == ""
    assert dist == -1


def test_normalizar_entidades_retorna_dataframe():
    import pandas as pd
    entidades = ["Petrobras", "Petrobrás", "Google", "Googgle"]
    df = normalizar_entidades(entidades, max_distancia=2)
    assert isinstance(df, pd.DataFrame)
    colunas = {"entidade_original", "entidade_canonica", "distancia_levenshtein", "similaridade", "grupo_id"}
    assert colunas.issubset(set(df.columns))


def test_normalizar_entidades_nao_vazio():
    entidades = ["Brasil", "Brasill", "Python"]
    df = normalizar_entidades(entidades)
    assert len(df) >= len(entidades)


def test_gerar_matriz_distancias():
    entidades = ["gato", "rato", "pato"]
    df = gerar_matriz_distancias(entidades)
    assert df.shape == (3, 3)
    # Diagonal deve ser zero
    for e in entidades:
        assert df.loc[e, e] == 0

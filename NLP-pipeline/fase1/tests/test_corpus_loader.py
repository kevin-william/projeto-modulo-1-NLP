import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from corpus_loader import carregar_artigos, obter_estatisticas_corpus, filtrar_artigos_por_tamanho


def test_carregar_artigos_contagem():
    artigos = carregar_artigos()
    assert len(artigos) > 0, "Deveria carregar ao menos 1 artigo"


def test_estrutura_artigo():
    artigos = carregar_artigos()
    for artigo in artigos:
        assert "titulo" in artigo
        assert "url" in artigo
        assert "conteudo" in artigo
        assert isinstance(artigo["titulo"], str) and len(artigo["titulo"]) > 0
        assert isinstance(artigo["url"], str) and len(artigo["url"]) > 0
        assert isinstance(artigo["conteudo"], str)


def test_titulos_artigos():
    artigos = carregar_artigos()
    titulos = [artigo["titulo"] for artigo in artigos]
    assert len(titulos) > 0
    assert any("Python" in titulo for titulo in titulos)


def test_obter_estatisticas_corpus():
    artigos = carregar_artigos()
    estatisticas = obter_estatisticas_corpus(artigos)
    assert estatisticas["total_artigos"] == len(artigos)
    assert estatisticas["total_caracteres"] > 0
    assert estatisticas["media_caracteres_por_artigo"] > 0


# ---------------------------------------------------------------------------
# Testes do filtro por tamanho mínimo de palavras
# ---------------------------------------------------------------------------

def test_filtrar_artigos_limiar_zero_nao_remove_nada():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=0)
    assert len(validos) == len(artigos)
    assert len(removidos) == 0


def test_filtrar_artigos_limiar_alto_remove_artigos():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=100000)
    assert len(removidos) > 0, "Com limiar muito alto, ao menos um artigo deve ser removido"


def test_filtrar_artigos_soma_total_preservada():
    artigos = carregar_artigos()
    validos, removidos = filtrar_artigos_por_tamanho(artigos, minimo_palavras=500)
    assert len(validos) + len(removidos) == len(artigos)


def test_filtrar_artigos_validos_tem_palavras_suficientes():
    artigos = carregar_artigos()
    limiar = 100
    validos, _ = filtrar_artigos_por_tamanho(artigos, minimo_palavras=limiar)
    for artigo in validos:
        quantidade = len(artigo["conteudo"].split())
        assert quantidade >= limiar, (
            f"Artigo '{artigo['titulo']}' tem {quantidade} palavras mas deveria ter >= {limiar}"
        )


def test_filtrar_artigos_retorna_tupla():
    artigos = carregar_artigos()
    resultado = filtrar_artigos_por_tamanho(artigos, minimo_palavras=10)
    assert isinstance(resultado, tuple)
    assert len(resultado) == 2

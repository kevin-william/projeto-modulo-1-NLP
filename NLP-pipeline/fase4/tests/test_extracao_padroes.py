"""
Testes para extracao_padroes.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from extracao_padroes import (
    extrair_codigos,
    extrair_cpfs,
    extrair_datas,
    extrair_emails,
    extrair_padroes_corpus,
    extrair_todos_padroes,
    extrair_urls,
    extrair_valores_monetarios,
    gerar_estatisticas_padroes,
)


TEXTO_TESTE = (
    "Contato: joao@empresa.com.br ou suporte@tech.org. "
    "Acesse https://www.empresa.com.br/pagina para mais info. "
    "Data: 15/03/2024. CPF: 123.456.789-00. "
    "Valor: R$ 1.250,00. Codigo: AB-12345."
)


def test_extrair_emails_encontra_emails():
    resultado = extrair_emails(TEXTO_TESTE)
    assert len(resultado) >= 1
    assert any("@" in e for e in resultado)


def test_extrair_emails_vazio():
    assert extrair_emails("Sem email aqui.") == []


def test_extrair_urls_encontra_url():
    resultado = extrair_urls(TEXTO_TESTE)
    assert len(resultado) >= 1
    assert any("http" in u for u in resultado)


def test_extrair_urls_vazio():
    assert extrair_urls("Sem URL aqui.") == []


def test_extrair_datas_encontra_data():
    resultado = extrair_datas(TEXTO_TESTE)
    assert "15/03/2024" in resultado


def test_extrair_datas_vazio():
    assert extrair_datas("Sem data aqui.") == []


def test_extrair_cpfs_encontra_cpf():
    resultado = extrair_cpfs(TEXTO_TESTE)
    assert "123.456.789-00" in resultado


def test_extrair_cpfs_vazio():
    assert extrair_cpfs("Sem CPF aqui.") == []


def test_extrair_valores_monetarios():
    resultado = extrair_valores_monetarios(TEXTO_TESTE)
    assert len(resultado) >= 1


def test_extrair_codigos():
    resultado = extrair_codigos(TEXTO_TESTE)
    assert len(resultado) >= 1


def test_extrair_todos_padroes_retorna_chaves():
    resultado = extrair_todos_padroes(TEXTO_TESTE)
    for chave in ("email", "url", "data", "cpf", "valor_monetario", "codigo"):
        assert chave in resultado


def test_extrair_todos_padroes_estrutura_ocorrencia():
    resultado = extrair_todos_padroes(TEXTO_TESTE)
    for tipo, ocorrencias in resultado.items():
        for oc in ocorrencias:
            assert "valor" in oc
            assert "posicao_inicio" in oc
            assert "posicao_fim" in oc


def test_extrair_padroes_corpus_colunas():
    df = extrair_padroes_corpus([TEXTO_TESTE], ["Doc 1"])
    colunas = {"documento_id", "titulo", "tipo_padrao", "valor", "posicao_inicio", "posicao_fim"}
    assert colunas.issubset(set(df.columns))


def test_extrair_padroes_corpus_nao_vazio():
    df = extrair_padroes_corpus([TEXTO_TESTE], ["Doc 1"])
    assert len(df) > 0


def test_gerar_estatisticas_padroes():
    df = extrair_padroes_corpus([TEXTO_TESTE], ["Doc 1"])
    stats = gerar_estatisticas_padroes(df)
    assert "tipo_padrao" in stats.columns
    assert "total" in stats.columns
    assert len(stats) > 0


def test_gerar_estatisticas_padroes_df_vazio():
    import pandas as pd
    stats = gerar_estatisticas_padroes(pd.DataFrame())
    assert "tipo_padrao" in stats.columns

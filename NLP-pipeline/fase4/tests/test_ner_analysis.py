"""
Testes para ner_analysis.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def _tentar_carregar_spacy():
    try:
        from ner_analysis import carregar_modelo_spacy
        nlp = carregar_modelo_spacy()
        return nlp
    except Exception:
        return None


TEXTO_SINTETICO = (
    "A Petrobras anunciou investimentos no Rio de Janeiro. "
    "O presidente Lula se reuniu com representantes da Microsoft."
)


@pytest.mark.skipif(
    _tentar_carregar_spacy() is None,
    reason="Nenhum modelo spaCy disponivel"
)
def test_extrair_entidades_documento():
    from ner_analysis import carregar_modelo_spacy, extrair_entidades_documento
    nlp = carregar_modelo_spacy()
    doc = nlp(TEXTO_SINTETICO)
    entidades = extrair_entidades_documento(doc)
    assert isinstance(entidades, list)
    for ent in entidades:
        assert "texto" in ent
        assert "tipo" in ent
        assert "inicio" in ent
        assert "fim" in ent


@pytest.mark.skipif(
    _tentar_carregar_spacy() is None,
    reason="Nenhum modelo spaCy disponivel"
)
def test_extrair_entidades_corpus():
    import pandas as pd
    from ner_analysis import carregar_modelo_spacy, extrair_entidades_corpus
    nlp = carregar_modelo_spacy()
    df = extrair_entidades_corpus([TEXTO_SINTETICO], nlp)
    assert isinstance(df, pd.DataFrame)
    assert "texto" in df.columns
    assert "tipo" in df.columns
    assert "documento_id" in df.columns


@pytest.mark.skipif(
    _tentar_carregar_spacy() is None,
    reason="Nenhum modelo spaCy disponivel"
)
def test_contar_entidades_por_tipo():
    import pandas as pd
    from ner_analysis import carregar_modelo_spacy, contar_entidades_por_tipo, extrair_entidades_corpus
    nlp = carregar_modelo_spacy()
    df = extrair_entidades_corpus([TEXTO_SINTETICO], nlp)
    counts = contar_entidades_por_tipo(df)
    assert "tipo" in counts.columns
    assert "total" in counts.columns


@pytest.mark.skipif(
    _tentar_carregar_spacy() is None,
    reason="Nenhum modelo spaCy disponivel"
)
def test_gerar_displacy_html_cria_arquivo():
    from ner_analysis import carregar_modelo_spacy, gerar_displacy_html
    nlp = carregar_modelo_spacy()
    doc = nlp(TEXTO_SINTETICO)
    with tempfile.TemporaryDirectory() as tmpdir:
        caminho = os.path.join(tmpdir, "teste.html")
        gerar_displacy_html(doc, caminho)
        assert os.path.exists(caminho)
        tamanho = os.path.getsize(caminho)
        assert tamanho > 0


def test_contar_entidades_por_tipo_df_vazio():
    import pandas as pd
    from ner_analysis import contar_entidades_por_tipo
    df_vazio = pd.DataFrame(columns=["texto", "tipo"])
    resultado = contar_entidades_por_tipo(df_vazio)
    assert "tipo" in resultado.columns
    assert "total" in resultado.columns


def test_consolidar_entidades_parquet_spacy():
    import pandas as pd
    from ner_analysis import consolidar_entidades_parquet_spacy
    df_spacy = pd.DataFrame({
        "texto": ["Petrobras", "Rio de Janeiro"],
        "tipo": ["ORG", "LOC"],
        "documento_id": [0, 0],
    })
    df_parquet = pd.DataFrame({
        "entidade": ["Lula"],
        "rotulo_entidade": ["PER"],
    })
    resultado = consolidar_entidades_parquet_spacy(df_spacy, df_parquet)
    assert isinstance(resultado, pd.DataFrame)
    assert "texto" in resultado.columns
    assert "tipo" in resultado.columns

"""
Teste de integracao end-to-end da Fase 4 com dados sinteticos.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import pandas as pd


# Verificar se spaCy esta disponivel para executar integracao completa
def _spacy_disponivel():
    try:
        from ner_analysis import carregar_modelo_spacy
        carregar_modelo_spacy()
        return True
    except Exception:
        return False


def _criar_parquet_sintetico(caminho: str):
    """Cria parquet com estrutura compativel com fase 1."""
    dados = []
    entidades = [
        ("Petrobras", "ORG"), ("Google", "ORG"), ("Microsoft", "ORG"),
        ("Lula", "PER"), ("Biden", "PER"),
        ("Rio de Janeiro", "LOC"), ("Brasilia", "LOC"),
    ]
    palavras_neutras = ["anunciou", "comprou", "investiu", "declarou"]
    for art_id in range(10):
        ent = entidades[art_id % len(entidades)]
        for i, palavra in enumerate(["O", ent[0], "anunciou", "resultados", "em", "2024"]):
            e = ent[0] if palavra == ent[0] else None
            r = ent[1] if palavra == ent[0] else None
            dados.append({
                "id_artigo": art_id,
                "id_token": i,
                "token": palavra,
                "lema": palavra.lower(),
                "processado": palavra.lower(),
                "pos": "PROPN" if e else "NOUN",
                "entidade": e,
                "rotulo_entidade": r,
                "titulo": f"Artigo {art_id}",
                "url": f"http://exemplo.com/{art_id}",
            })
    df = pd.DataFrame(dados)
    df.to_parquet(caminho, index=False)
    return df


@pytest.mark.skipif(not _spacy_disponivel(), reason="spaCy nao disponivel para integracao")
def test_pipeline_com_dados_sinteticos(tmp_path, monkeypatch):
    """
    Teste de integracao: executa o pipeline com dados sinteticos
    e verifica que todos os arquivos esperados sao gerados.
    """
    import fase4_config as cfg

    # Cria estrutura temporaria
    dir_input = tmp_path / "input"
    dir_output = tmp_path / "output"
    dir_plots = dir_output / "plots"
    dir_displacy = dir_output / "displacy"
    for d in [dir_input, dir_output, dir_plots, dir_displacy]:
        d.mkdir(parents=True, exist_ok=True)

    caminho_parquet = str(dir_input / "test.parquet")
    _criar_parquet_sintetico(caminho_parquet)

    # Monkeypatches de configuracao
    monkeypatch.setattr(cfg, "CAMINHO_PARQUET_ENTRADA", caminho_parquet)
    monkeypatch.setattr(cfg, "CAMINHO_ARTEFATO_FASE2", str(dir_input / "nao_existe.lpf2"))
    monkeypatch.setattr(cfg, "DIRETORIO_SAIDA", str(dir_output))
    monkeypatch.setattr(cfg, "DIRETORIO_PLOTS", str(dir_plots))
    monkeypatch.setattr(cfg, "DIRETORIO_DISPLACY", str(dir_displacy))
    monkeypatch.setattr(cfg, "CAMINHO_LOG", str(dir_output / "test.log"))

    # Recarrega logger com novo caminho
    import logging
    for handler in logging.getLogger("fase4").handlers[:]:
        handler.close()
        logging.getLogger("fase4").removeHandler(handler)

    from main import executar_fase4_principal
    saidas = executar_fase4_principal()

    # Verifica que o relatorio foi gerado
    assert "relatorio_interpretativo" in saidas
    assert os.path.exists(saidas["relatorio_interpretativo"])

    # Verifica conteudo do relatorio
    with open(saidas["relatorio_interpretativo"], encoding="utf-8") as fh:
        conteudo = fh.read()
    assert len(conteudo) > 50

    # Verifica CSVs essenciais
    for chave in ("entidades_extraidas", "fuzzy_matches", "grafo_edges_csv", "nos_grafo_csv"):
        if chave in saidas:
            assert os.path.exists(saidas[chave]), f"Arquivo ausente: {chave}"

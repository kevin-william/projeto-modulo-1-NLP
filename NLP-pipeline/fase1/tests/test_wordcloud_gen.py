import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import wordcloud_gen
from fase1_config import (
    LARGURA_NUVEM_PALAVRAS,
    ALTURA_NUVEM_PALAVRAS,
    MAXIMO_PALAVRAS_NUVEM,
    PALETA_CORES_NUVEM,
    COR_FUNDO_NUVEM,
    TAMANHO_MINIMO_FONTE_NUVEM,
    TAMANHO_MAXIMO_FONTE_NUVEM,
    SEMENTE_NUVEM_PALAVRAS,
)


class _WordCloudFalso:
    ultimo_kwargs = None

    def __init__(self, **kwargs):
        _WordCloudFalso.ultimo_kwargs = kwargs

    def generate(self, _texto):
        return self


def _desativar_plot(monkeypatch):
    monkeypatch.setattr(wordcloud_gen.plt, "figure", lambda *args, **kwargs: None)
    monkeypatch.setattr(wordcloud_gen.plt, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(wordcloud_gen.plt, "axis", lambda *args, **kwargs: None)
    monkeypatch.setattr(wordcloud_gen.plt, "tight_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(wordcloud_gen.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(wordcloud_gen.plt, "close", lambda *args, **kwargs: None)


def test_gerar_nuvem_palavras_usa_defaults_do_config(monkeypatch):
    monkeypatch.setattr(wordcloud_gen, "WordCloud", _WordCloudFalso)
    monkeypatch.setattr(wordcloud_gen, "obter_stopwords", lambda: set())
    _desativar_plot(monkeypatch)

    tokens = [{"processado": "dados"}, {"processado": "linguagem"}]
    wordcloud_gen.gerar_nuvem_palavras(tokens)

    assert _WordCloudFalso.ultimo_kwargs["width"] == LARGURA_NUVEM_PALAVRAS
    assert _WordCloudFalso.ultimo_kwargs["height"] == ALTURA_NUVEM_PALAVRAS
    assert _WordCloudFalso.ultimo_kwargs["max_words"] == MAXIMO_PALAVRAS_NUVEM
    assert _WordCloudFalso.ultimo_kwargs["colormap"] == PALETA_CORES_NUVEM
    assert _WordCloudFalso.ultimo_kwargs["background_color"] == COR_FUNDO_NUVEM
    assert _WordCloudFalso.ultimo_kwargs["min_font_size"] == TAMANHO_MINIMO_FONTE_NUVEM
    assert _WordCloudFalso.ultimo_kwargs["max_font_size"] == TAMANHO_MAXIMO_FONTE_NUVEM
    assert _WordCloudFalso.ultimo_kwargs["random_state"] == SEMENTE_NUVEM_PALAVRAS


def test_gerar_nuvem_palavras_permite_override_de_parametros(monkeypatch):
    monkeypatch.setattr(wordcloud_gen, "WordCloud", _WordCloudFalso)
    monkeypatch.setattr(wordcloud_gen, "obter_stopwords", lambda: set())
    _desativar_plot(monkeypatch)

    tokens = [{"processado": "python"}, {"processado": "nlp"}]
    wordcloud_gen.gerar_nuvem_palavras(
        tokens,
        largura=900,
        altura=450,
        maximo_palavras=120,
        paleta_cores="plasma",
        cor_fundo="black",
        tamanho_minimo_fonte=12,
        tamanho_maximo_fonte=110,
        random_state=7,
    )

    assert _WordCloudFalso.ultimo_kwargs["width"] == 900
    assert _WordCloudFalso.ultimo_kwargs["height"] == 450
    assert _WordCloudFalso.ultimo_kwargs["max_words"] == 120
    assert _WordCloudFalso.ultimo_kwargs["colormap"] == "plasma"
    assert _WordCloudFalso.ultimo_kwargs["background_color"] == "black"
    assert _WordCloudFalso.ultimo_kwargs["min_font_size"] == 12
    assert _WordCloudFalso.ultimo_kwargs["max_font_size"] == 110
    assert _WordCloudFalso.ultimo_kwargs["random_state"] == 7


def test_gerar_nuvem_palavras_valida_parametros_invalidos():
    tokens = [{"processado": "texto"}]

    with pytest.raises(ValueError, match="tamanho_maximo_fonte"):
        wordcloud_gen.gerar_nuvem_palavras(
            tokens,
            tamanho_minimo_fonte=20,
            tamanho_maximo_fonte=10,
        )

    with pytest.raises(ValueError, match="paleta_cores"):
        wordcloud_gen.gerar_nuvem_palavras(tokens, paleta_cores="")

    with pytest.raises(ValueError, match="maximo_palavras"):
        wordcloud_gen.gerar_nuvem_palavras(tokens, maximo_palavras=0)


class _CapturarTextoWordCloud:
    """WordCloud falso que captura o texto recebido por generate()."""
    texto_recebido = None

    def __init__(self, **kwargs):
        pass

    def generate(self, texto):
        _CapturarTextoWordCloud.texto_recebido = texto
        return self


def test_incluir_stopwords_usa_campo_processado(monkeypatch, tmp_path):
    """Com incluir_stopwords=True a nuvem deve usar o campo 'processado', não 'texto'."""
    monkeypatch.setattr(wordcloud_gen, "WordCloud", _CapturarTextoWordCloud)
    monkeypatch.setattr(wordcloud_gen, "obter_stopwords", lambda: set())
    _desativar_plot(monkeypatch)

    tokens = [
        {"texto": "running", "processado": "correr"},
        {"texto": "computed", "processado": "computar"},
    ]
    wordcloud_gen.gerar_nuvem_palavras(
        tokens, incluir_stopwords=True, caminho_saida=str(tmp_path / "nuvem.png")
    )

    assert _CapturarTextoWordCloud.texto_recebido is not None
    assert "correr" in _CapturarTextoWordCloud.texto_recebido
    assert "computar" in _CapturarTextoWordCloud.texto_recebido
    assert "running" not in _CapturarTextoWordCloud.texto_recebido
    assert "computed" not in _CapturarTextoWordCloud.texto_recebido


def test_incluir_stopwords_false_usa_campo_processado(monkeypatch, tmp_path):
    """Com incluir_stopwords=False (padrão) a nuvem também deve usar 'processado'."""
    monkeypatch.setattr(wordcloud_gen, "WordCloud", _CapturarTextoWordCloud)
    monkeypatch.setattr(wordcloud_gen, "obter_stopwords", lambda: {"de"})
    _desativar_plot(monkeypatch)

    tokens = [
        {"texto": "Banco", "processado": "banco"},
        {"texto": "de", "processado": "de"},
        {"texto": "Dados", "processado": "dado"},
    ]
    wordcloud_gen.gerar_nuvem_palavras(
        tokens, incluir_stopwords=False, caminho_saida=str(tmp_path / "nuvem.png")
    )

    assert "banco" in _CapturarTextoWordCloud.texto_recebido
    assert "dado" in _CapturarTextoWordCloud.texto_recebido
    assert "de" not in _CapturarTextoWordCloud.texto_recebido


def test_lemmatizacao_e_none_produzem_texto_diferente(monkeypatch, tmp_path):
    """Tokens lemmatizados e originais devem gerar textos de nuvem distintos."""
    monkeypatch.setattr(wordcloud_gen, "WordCloud", _CapturarTextoWordCloud)
    monkeypatch.setattr(wordcloud_gen, "obter_stopwords", lambda: set())
    _desativar_plot(monkeypatch)

    tokens_lemmatizacao = [
        {"texto": "correndo", "processado": "correr"},
        {"texto": "pensamentos", "processado": "pensamento"},
    ]
    tokens_none = [
        {"texto": "correndo", "processado": "correndo"},
        {"texto": "pensamentos", "processado": "pensamentos"},
    ]

    wordcloud_gen.gerar_nuvem_palavras(
        tokens_lemmatizacao, incluir_stopwords=True, caminho_saida=str(tmp_path / "lem.png")
    )
    texto_lemmatizacao = _CapturarTextoWordCloud.texto_recebido

    wordcloud_gen.gerar_nuvem_palavras(
        tokens_none, incluir_stopwords=True, caminho_saida=str(tmp_path / "none.png")
    )
    texto_none = _CapturarTextoWordCloud.texto_recebido

    assert texto_lemmatizacao != texto_none

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
_MODELOS_DIR = os.path.join(_SRC_DIR, "modelos_topicos")
_SHARED_DIR = os.path.join(_TESTS_DIR, "..", "..", "shared")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _MODELOS_DIR)
sys.path.insert(0, _SHARED_DIR)
sys.path.insert(0, os.path.join(_TESTS_DIR, "..", ".."))

import pytest


# Parâmetros permissivos para corpus sintético pequeno
_PARAMS_TREINO = dict(no_below=1, no_above=1.0, keep_n=200)
_PARAMS_MODELO = {"passes": 2, "iterations": 10, "random_state": 42}


@pytest.fixture
def tokens_lda():
    return [
        ["machine", "learning", "algorithm", "data"],
        ["deep", "neural", "network", "training"],
        ["natural", "language", "processing", "text"],
        ["computer", "vision", "image", "recognition"],
        ["reinforcement", "learning", "agent", "policy"],
        ["statistical", "analysis", "regression", "data"],
        ["python", "programming", "software", "development"],
        ["database", "sql", "query", "optimization"],
        ["cloud", "computing", "distributed", "systems"],
        ["cybersecurity", "encryption", "network", "security"],
        ["machine", "learning", "deep", "neural"],
        ["data", "science", "analysis", "statistics"],
        ["algorithm", "design", "complexity", "theory"],
        ["web", "development", "javascript", "framework"],
        ["operating", "system", "kernel", "process"],
    ]


class TestLDA:

    def test_shape_distribuicao_documentos(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=5, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        dist = lda.obter_distribuicao_documentos()
        assert dist.shape == (len(tokens_lda), 5)

    def test_obter_topicos_retorna_lista_correta(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        topicos = lda.obter_topicos(top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra in topico:
                assert isinstance(palavra, str)

    def test_obter_topicos_sem_argumento_vocabulario(self, tokens_lda):
        """obter_topicos() nao recebe vocabulario — o Dictionary e interno ao modelo."""
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)
        # deve funcionar sem nenhum argumento adicional
        topicos = lda.obter_topicos()
        assert len(topicos) == 3

    def test_perplexidade_eh_float(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        perplexidade = lda.obter_perplexidade()
        assert isinstance(perplexidade, float)

    def test_coerencia_eh_float(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        coerencia = lda.obter_coerencia()
        assert isinstance(coerencia, float)

    def test_obter_corpus_nao_vazio(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        assert len(lda.obter_corpus()) == len(tokens_lda)

    def test_obter_dictionary_nao_nulo(self, tokens_lda):
        from gensim.corpora import Dictionary
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        assert isinstance(lda.obter_dictionary(), Dictionary)
        assert len(lda.obter_dictionary()) > 0

    def test_erro_se_nao_treinado(self):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3)

        with pytest.raises(RuntimeError):
            lda.obter_topicos()

        with pytest.raises(RuntimeError):
            lda.obter_distribuicao_documentos()

        with pytest.raises(RuntimeError):
            lda.obter_perplexidade()

        with pytest.raises(RuntimeError):
            lda.obter_coerencia()

    def test_obter_topicos_com_pesos_retorna_tuplas(self, tokens_lda):
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=3, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        topicos = lda.obter_topicos_com_pesos(top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra, peso in topico:
                assert isinstance(palavra, str)
                assert isinstance(peso, float)
                assert peso >= 0

    def test_distribuicao_documentos_soma_aproximadamente_1(self, tokens_lda):
        import numpy as np
        from lda_modelo import ModeloLDA

        lda = ModeloLDA(num_topicos=5, params=_PARAMS_MODELO)
        lda.treinar(tokens_lda, **_PARAMS_TREINO)

        dist = lda.obter_distribuicao_documentos()
        somas = dist.sum(axis=1)
        assert all(abs(s - 1.0) < 0.01 for s in somas), "Cada linha deve somar ~1.0"

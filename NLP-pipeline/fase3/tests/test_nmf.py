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

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture
def dados_nmf():
    docs = [
        "machine learning algorithm data",
        "deep neural network training",
        "natural language processing text",
        "computer vision image recognition",
        "reinforcement learning agent policy",
        "statistical analysis regression data",
        "python programming software development",
        "database sql query optimization",
        "cloud computing distributed systems",
        "cybersecurity encryption network security",
        "machine learning deep neural",
        "data science analysis statistics",
        "algorithm design complexity theory",
        "web development javascript framework",
        "operating system kernel process",
    ]
    vectorizer = TfidfVectorizer(max_features=50)
    matriz = vectorizer.fit_transform(docs)
    vocabulario = vectorizer.get_feature_names_out().tolist()
    return matriz, vocabulario, len(docs)


class TestNMF:

    def test_shape_distribuicao_documentos(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        matriz, _, n_docs = dados_nmf
        nmf = ModeloNMF(num_topicos=5)
        nmf.treinar(matriz)

        distribuicao = nmf.obter_distribuicao_documentos()
        assert distribuicao.shape == (n_docs, 5)

    def test_obter_topicos_retorna_lista_correta(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        matriz, vocabulario, _ = dados_nmf
        nmf = ModeloNMF(num_topicos=3)
        nmf.treinar(matriz)

        topicos = nmf.obter_topicos(vocabulario, top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra in topico:
                assert isinstance(palavra, str)

    def test_valores_distribuicao_sao_nao_negativos(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        matriz, _, _ = dados_nmf
        nmf = ModeloNMF(num_topicos=5)
        nmf.treinar(matriz)

        distribuicao = nmf.obter_distribuicao_documentos()
        assert np.all(distribuicao >= 0)

    def test_erro_se_nao_treinado(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        _, vocabulario, _ = dados_nmf
        nmf = ModeloNMF(num_topicos=3)

        with pytest.raises(RuntimeError):
            nmf.obter_topicos(vocabulario)

        with pytest.raises(RuntimeError):
            nmf.obter_distribuicao_documentos()

    def test_obter_topicos_com_pesos_retorna_tuplas_nao_negativas(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        matriz, vocabulario, _ = dados_nmf
        nmf = ModeloNMF(num_topicos=3)
        nmf.treinar(matriz)

        topicos = nmf.obter_topicos_com_pesos(vocabulario, top_n=5)
        assert len(topicos) == 3
        for topico in topicos:
            assert len(topico) == 5
            for palavra, peso in topico:
                assert isinstance(palavra, str)
                assert isinstance(peso, float)
                assert peso >= 0, f"NMF deve ter pesos nao-negativos, mas obteve {peso}"

    def test_obter_topicos_com_pesos_erro_se_nao_treinado(self, dados_nmf):
        from nmf_modelo import ModeloNMF

        _, vocabulario, _ = dados_nmf
        nmf = ModeloNMF(num_topicos=3)

        with pytest.raises(RuntimeError):
            nmf.obter_topicos_com_pesos(vocabulario)

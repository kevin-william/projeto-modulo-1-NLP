import os
import sys
import tempfile

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
_MODELOS_DIR = os.path.join(_SRC_DIR, "modelos_topicos")
_SHARED_DIR = os.path.join(_TESTS_DIR, "..", "..", "shared")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _MODELOS_DIR)
sys.path.insert(0, _SHARED_DIR)

import pytest


class TestEDA:

    @pytest.fixture
    def documentos_teste(self):
        return [
            "machine learning algorithm data",
            "deep neural network training",
            "natural language processing text",
            "computer vision image recognition deep",
            "reinforcement learning agent policy",
        ]

    @pytest.fixture
    def titulos_teste(self):
        return ["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"]

    def test_retorno_contem_chaves_esperadas(self, documentos_teste, titulos_teste, tmp_path):
        from eda import executar_eda

        metricas = executar_eda(documentos_teste, titulos_teste, str(tmp_path))

        chaves_esperadas = [
            "num_documentos", "comprimento_medio", "comprimento_mediano",
            "comprimento_std", "comprimento_min", "comprimento_max",
            "tamanho_vocabulario", "total_tokens", "top_20_termos",
        ]
        for chave in chaves_esperadas:
            assert chave in metricas, f"Chave '{chave}' ausente no retorno"

    def test_comprimento_min_maior_que_zero(self, documentos_teste, titulos_teste, tmp_path):
        from eda import executar_eda

        metricas = executar_eda(documentos_teste, titulos_teste, str(tmp_path))
        assert metricas["comprimento_min"] > 0

    def test_tamanho_vocabulario_maior_que_zero(self, documentos_teste, titulos_teste, tmp_path):
        from eda import executar_eda

        metricas = executar_eda(documentos_teste, titulos_teste, str(tmp_path))
        assert metricas["tamanho_vocabulario"] > 0

    def test_graficos_criados_no_diretorio(self, documentos_teste, titulos_teste, tmp_path):
        from eda import executar_eda

        executar_eda(documentos_teste, titulos_teste, str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), "eda_top_termos.png"))
        assert os.path.exists(os.path.join(str(tmp_path), "eda_zipf.png"))

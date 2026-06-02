import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from similarity.cosine_search import MotorBuscaCosseno
from search_interface import iniciar_interface_busca, _obter_modelo_word2vec


class FakePipeline:
    class FakeConfig:
        def get(self, key, default=None):
            if key == "TOP_K_RESULTADOS":
                return 5
            return default

    def __init__(self):
        self.motores_busca = {}
        self.vetorizadores = {}
        self.configuracoes = FakePipeline.FakeConfig()

    def buscar_texto(self, method, query, top_k=10):
        return self.motores_busca[method].search(np.array([1, 0, 0]), top_k)


def test_search_interface_imports():
    assert callable(iniciar_interface_busca)


def test_obter_modelo_word2vec_sem_word2vec():
    pipeline = FakePipeline()
    resultado = _obter_modelo_word2vec(pipeline)
    assert resultado is None


def test_obter_modelo_word2vec_modelo_none():
    pipeline = FakePipeline()
    fake_vetorizador = MagicMock()
    fake_vetorizador.model = None
    pipeline.vetorizadores["word2vec"] = fake_vetorizador
    resultado = _obter_modelo_word2vec(pipeline)
    assert resultado is None


def test_obter_modelo_word2vec_retorna_wv():
    pipeline = FakePipeline()
    fake_wv = MagicMock()
    fake_vetorizador = MagicMock()
    fake_vetorizador.model.wv = fake_wv
    pipeline.vetorizadores["word2vec"] = fake_vetorizador
    resultado = _obter_modelo_word2vec(pipeline)
    assert resultado is fake_wv


def test_similar_palavra_fora_do_vocabulario(capsys):
    pipeline = FakePipeline()
    pipeline.motores_busca["tfidf"] = MagicMock()  # necessario para entrar no loop
    fake_wv = MagicMock()
    fake_wv.__contains__.return_value = False  # nenhuma palavra no vocabulario
    fake_vetorizador = MagicMock()
    fake_vetorizador.model.wv = fake_wv
    pipeline.vetorizadores["word2vec"] = fake_vetorizador

    with patch("builtins.input", side_effect=["similar xyzzy", "sair"]):
        iniciar_interface_busca(pipeline)

    captured = capsys.readouterr()
    assert "vocabulario" in captured.out


def test_interface_sair_imediato(capsys):
    pipeline = FakePipeline()
    pipeline.motores_busca["tfidf"] = MagicMock()
    with patch("builtins.input", side_effect=["sair"]):
        iniciar_interface_busca(pipeline)
    captured = capsys.readouterr()
    assert "Encerrando" in captured.out

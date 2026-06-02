import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import pandas as pd
from unittest.mock import MagicMock
from vectorizers.ctfidf_vectorizer import CalculadoraCTfIdf


def _criar_tfidf_vetorizador_fake(vocab, idf_values):
    """Cria um VetorizadorTfidf mockado com vocabulario e idf controlados."""
    fake = MagicMock()
    fake.vectorizer.idf_ = idf_values
    fake.vectorizer.vocabulary_ = vocab
    fake.get_feature_names.return_value = list(vocab.keys())
    return fake


class TestCalculadoraCTfIdf:
    def setup_method(self):
        self.vocab = {"python": 0, "linguagem": 1, "historia": 2, "data": 3}
        self.idf = [2.0, 1.5, 1.8, 1.2]
        self.tfidf_fake = _criar_tfidf_vetorizador_fake(self.vocab, self.idf)
        self.calculadora = CalculadoraCTfIdf(self.tfidf_fake)

    def test_retorna_dict_por_categoria(self):
        docs = [
            "python linguagem python",
            "python linguagem",
            "historia data historia",
        ]
        cats = ["programacao", "programacao", "geral"]
        resultado = self.calculadora.calcular(docs, cats)
        assert isinstance(resultado, dict)
        assert "programacao" in resultado
        assert "geral" in resultado

    def test_scores_sao_series_pandas(self):
        docs = ["python linguagem", "historia data"]
        cats = ["programacao", "geral"]
        resultado = self.calculadora.calcular(docs, cats)
        assert isinstance(resultado["programacao"], pd.Series)
        assert isinstance(resultado["geral"], pd.Series)

    def test_scores_nao_negativos(self):
        docs = ["python linguagem", "historia data"]
        cats = ["programacao", "geral"]
        resultado = self.calculadora.calcular(docs, cats)
        for cat, scores in resultado.items():
            assert (scores >= 0).all(), f"Scores negativos na categoria '{cat}'"

    def test_top_palavra_coerente(self):
        # "python" aparece 3x em programacao; deve ter score alto
        docs = ["python python python linguagem"]
        cats = ["programacao"]
        resultado = self.calculadora.calcular(docs, cats)
        top_palavra = resultado["programacao"].index[0]
        assert top_palavra == "python"

    def test_scores_ordenados_decrescente(self):
        docs = ["python python linguagem historia data"]
        cats = ["geral"]
        resultado = self.calculadora.calcular(docs, cats)
        scores = resultado["geral"].values
        assert list(scores) == sorted(scores, reverse=True)

    def test_categoria_unica(self):
        docs = ["python", "linguagem", "historia"]
        cats = ["programacao", "programacao", "programacao"]
        resultado = self.calculadora.calcular(docs, cats)
        assert len(resultado) == 1
        assert "programacao" in resultado

    def test_multiplas_categorias(self):
        docs = ["python linguagem", "historia data", "python data"]
        cats = ["programacao", "geral", "misto"]
        resultado = self.calculadora.calcular(docs, cats)
        assert len(resultado) == 3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pytest
from vectorizers.word2vec_vectorizer import VetorizadorWord2Vec


class TestVetorizadorWord2Vec:
    def test_fit_with_sentences(self):
        sentences = [
            ["ola", "mundo", "programacao"],
            ["python", "linguagem", "natural"],
            ["processamento", "texto", "python"],
        ]
        w2v = VetorizadorWord2Vec(vector_size=10, window=3, min_count=1, epochs=5, seed=42)
        w2v.fit(sentences)
        assert w2v.model is not None

    def test_get_sentence_vector(self):
        sentences = [
            ["ola", "mundo", "abc"],
            ["python", "legal", "abc"],
            ["teste", "exemplo", "abc"],
        ]
        w2v = VetorizadorWord2Vec(vector_size=5, window=3, min_count=1, epochs=5, seed=42)
        w2v.fit(sentences)
        vec = w2v.obter_vetor_sentenca(["ola", "mundo"])
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (5,)

    def test_get_sentence_vector_unknown_words(self):
        sentences = [["a", "b", "c"]]
        w2v = VetorizadorWord2Vec(vector_size=5)
        w2v.fit(sentences)
        vec = w2v.obter_vetor_sentenca(["x", "y", "z"])
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (5,)
        assert np.all(vec == 0)

    def test_get_mean_document_embeddings(self):
        sentences = [
            ["ola", "mundo"],
            ["python", "legal"],
            ["teste", "abc"],
        ]
        w2v = VetorizadorWord2Vec(vector_size=5, epochs=3, seed=42)
        w2v.fit(sentences)
        embeddings = w2v.obter_embeddings_medios_documentos(sentences)
        assert embeddings.shape == (3, 5)

    def test_empty_sentences(self):
        w2v = VetorizadorWord2Vec(vector_size=5)
        w2v.fit([])
        vec = w2v.obter_vetor_sentenca(["teste"])
        assert np.all(vec == 0)

    def test_workers_param_accepted(self):
        w2v = VetorizadorWord2Vec(vector_size=5, workers=2, min_count=1, epochs=3, seed=0)
        assert w2v.params["workers"] == 2
        sentences = [["a", "b", "c"], ["d", "e", "f"]]
        w2v.fit(sentences)
        assert w2v.model is not None

    def test_workers_default_is_one(self):
        w2v = VetorizadorWord2Vec()
        assert w2v.params["workers"] == 1

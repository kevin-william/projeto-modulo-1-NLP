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
sys.path.insert(0, os.path.join(_TESTS_DIR, "..", ".."))

import pytest
from shared.artifacts import ArtifactFase2
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def criar_artefato_sintetico(com_tfidf=True, com_bow=True, n_docs=30):
    docs = [
        "machine learning algorithm data intelligence",
        "deep neural network training learning",
        "natural language processing text nlp",
        "computer vision image recognition deep",
        "reinforcement learning agent policy reward",
        "statistical analysis regression model data",
        "python programming software development code",
        "database sql query optimization data",
        "cloud computing distributed systems network",
        "cybersecurity encryption network security data",
    ] * (n_docs // 10)
    docs = docs[:n_docs]

    titulos = [f"Article {i}" for i in range(n_docs)]

    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(docs)

    from sklearn.feature_extraction.text import CountVectorizer
    bow_vectorizer = CountVectorizer(max_features=50)
    bow_matrix = bow_vectorizer.fit_transform(docs)

    return ArtifactFase2(
        documentos=docs,
        titulos=titulos,
        parametros={"bow": {"max_features": 50}, "tfidf": {"max_features": 50}},
        bow_matrix=bow_matrix if com_bow else None,
        tfidf_matrix=tfidf_matrix if com_tfidf else None,
        bow_vectorizer=bow_vectorizer if com_bow else None,
        tfidf_vectorizer=vectorizer if com_tfidf else None,
    )


def salvar_artefato_temporario(artefato):
    tmp = tempfile.NamedTemporaryFile(suffix=".lpf2", delete=False)
    artefato.save(tmp.name)
    tmp.close()
    return tmp.name


class TestCarregadorArtefato:

    def test_arquivo_valido_carrega(self):
        from carregador_artefato import carregar_artefato_fase2

        artefato = criar_artefato_sintetico()
        caminho = salvar_artefato_temporario(artefato)

        try:
            carregado = carregar_artefato_fase2(caminho)
            assert carregado is not None
            assert len(carregado.documentos) == 30
            assert len(carregado.titulos) == 30
        finally:
            os.unlink(caminho)

    def test_arquivo_ausente_lanca_file_not_found_error(self):
        from carregador_artefato import carregar_artefato_fase2

        with pytest.raises(FileNotFoundError):
            carregar_artefato_fase2("/caminho/inexistente/artefato.lpf2")

    def test_tipo_errado_lanca_type_error(self):
        from carregador_artefato import carregar_artefato_fase2

        tmp = tempfile.NamedTemporaryFile(suffix=".lpf2", delete=False)
        import joblib
        joblib.dump("string_qualquer", tmp.name)
        tmp.close()

        try:
            with pytest.raises(TypeError):
                carregar_artefato_fase2(tmp.name)
        finally:
            os.unlink(tmp.name)

    def test_artefato_sem_documentos_lanca_value_error(self):
        from carregador_artefato import carregar_artefato_fase2

        artefato = criar_artefato_sintetico()
        artefato.documentos = []
        caminho = salvar_artefato_temporario(artefato)

        try:
            with pytest.raises(ValueError):
                carregar_artefato_fase2(caminho)
        finally:
            os.unlink(caminho)

    def test_artefato_sem_tfidf_matrix_lanca_value_error(self):
        from carregador_artefato import carregar_artefato_fase2

        artefato = criar_artefato_sintetico(com_tfidf=True)
        artefato.tfidf_matrix = None
        caminho = salvar_artefato_temporario(artefato)

        try:
            with pytest.raises(ValueError):
                carregar_artefato_fase2(caminho)
        finally:
            os.unlink(caminho)

    def test_artefato_sem_tfidf_vectorizer_lanca_value_error(self):
        from carregador_artefato import carregar_artefato_fase2

        artefato = criar_artefato_sintetico(com_tfidf=True)
        artefato.tfidf_vectorizer = None
        caminho = salvar_artefato_temporario(artefato)

        try:
            with pytest.raises(ValueError):
                carregar_artefato_fase2(caminho)
        finally:
            os.unlink(caminho)

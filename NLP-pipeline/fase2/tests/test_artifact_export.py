import sys
import os

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_TESTS_DIR, "..", "src")
_SHARED_DIR = os.path.join(_TESTS_DIR, "..", "..", "shared")
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _SHARED_DIR)

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from artifacts import ArtifactFase2
from embedding_pipeline import PipelineEmbeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_parquet(tmp_path):
    df = pd.DataFrame({
        "id_artigo": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        "id_token": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        "token": [
            "aprendizado", "maquina", "algoritmo",
            "redes", "neurais", "profundo",
            "processamento", "linguagem", "natural",
            "mineracao", "dados", "texto",
            "inteligencia", "artificial", "computador",
        ],
        "pos": ["NOUN"] * 15,
        "tag": ["N"] * 15,
        "lema": [
            "aprendizado", "maquina", "algoritmo",
            "redes", "neurais", "profundo",
            "processamento", "linguagem", "natural",
            "mineracao", "dados", "texto",
            "inteligencia", "artificial", "computador",
        ],
        "processado": [
            "aprendizado", "maquina", "algoritmo",
            "redes", "neurais", "profundo",
            "processamento", "linguagem", "natural",
            "mineracao", "dados", "texto",
            "inteligencia", "artificial", "computador",
        ],
        "relacao_dependencia": ["nsubj"] * 15,
        "token_cabeca": [""] * 15,
        "entidade": [""] * 15,
        "rotulo_entidade": [""] * 15,
        "titulo": (
            ["Aprendizado de Maquina"] * 3
            + ["Redes Neurais"] * 3
            + ["PLN"] * 3
            + ["Mineracao de Dados"] * 3
            + ["IA"] * 3
        ),
        "url": (
            ["http://1"] * 3
            + ["http://2"] * 3
            + ["http://3"] * 3
            + ["http://4"] * 3
            + ["http://5"] * 3
        ),
    })
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)
    return str(parquet_path)


def _pipeline_config(tmp_path, metodos):
    return {
        "METODOS_EMBEDDING": metodos,
        "DIRETORIO_SAIDA": str(tmp_path),
        "TOP_K_RESULTADOS": 3,
        "PARAMS_BOW": {"max_features": 50, "min_df": 1},
        "PARAMS_TFIDF": {"max_features": 50, "min_df": 1},
        "PARAMS_WORD2VEC": {"vector_size": 10, "window": 2, "min_count": 1, "epochs": 5, "seed": 42},
        "HABILITAR_TSNE": False,
    }


def _artifact_path(tmp_path):
    return tmp_path / "artifacts" / "fase2_artifact.lpf2"


# ---------------------------------------------------------------------------
# 6.1 — artifact eh criado com bow
# ---------------------------------------------------------------------------

def test_artifact_eh_criado_com_bow(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["bow"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()
    assert _artifact_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# 6.2 — artifact eh criado com tfidf
# ---------------------------------------------------------------------------

def test_artifact_eh_criado_com_tfidf(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["tfidf"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()
    assert _artifact_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# 6.3 — artifact eh criado com bow e tfidf
# ---------------------------------------------------------------------------

def test_artifact_eh_criado_com_bow_e_tfidf(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["bow", "tfidf"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    caminho = _artifact_path(tmp_path)
    assert caminho.exists()

    artifact = ArtifactFase2.load(str(caminho))
    assert artifact.bow_matrix is not None
    assert artifact.tfidf_matrix is not None


# ---------------------------------------------------------------------------
# 6.4 — artifact NAO eh criado quando somente word2vec
# ---------------------------------------------------------------------------

def test_artifact_nao_eh_criado_somente_word2vec(sample_parquet, tmp_path, caplog):
    config = _pipeline_config(tmp_path, ["word2vec"])
    pipeline = PipelineEmbeddings(config, sample_parquet)

    import logging
    with caplog.at_level(logging.WARNING):
        pipeline.executar()

    assert not _artifact_path(tmp_path).exists()
    assert any("Artefato .lpf2 nao sera gerado" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# 6.5 — conteudo do artifact: bow_matrix
# ---------------------------------------------------------------------------

def test_artifact_conteudo_bow_matrix(sample_parquet, tmp_path):
    n_documentos = 5
    config = _pipeline_config(tmp_path, ["bow"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    artifact = ArtifactFase2.load(str(_artifact_path(tmp_path)))

    assert isinstance(artifact.bow_matrix, csr_matrix)
    assert artifact.bow_matrix.shape[0] == n_documentos
    assert artifact.bow_matrix.shape[1] > 0
    assert np.issubdtype(artifact.bow_matrix.dtype, np.integer)
    assert artifact.bow_matrix.min() >= 0
    assert hasattr(artifact.bow_vectorizer, "vocabulary_")


# ---------------------------------------------------------------------------
# 6.6 — conteudo do artifact: tfidf_matrix
# ---------------------------------------------------------------------------

def test_artifact_conteudo_tfidf_matrix(sample_parquet, tmp_path):
    n_documentos = 5
    config = _pipeline_config(tmp_path, ["tfidf"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    artifact = ArtifactFase2.load(str(_artifact_path(tmp_path)))

    assert isinstance(artifact.tfidf_matrix, csr_matrix)
    assert artifact.tfidf_matrix.shape[0] == n_documentos
    assert np.issubdtype(artifact.tfidf_matrix.dtype, np.floating)
    assert artifact.tfidf_matrix.min() >= 0.0
    assert hasattr(artifact.tfidf_vectorizer, "vocabulary_")
    assert hasattr(artifact.tfidf_vectorizer, "idf_")


# ---------------------------------------------------------------------------
# 6.7 — conteudo do artifact: documentos e titulos
# ---------------------------------------------------------------------------

def test_artifact_conteudo_documentos(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["bow"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    artifact = ArtifactFase2.load(str(_artifact_path(tmp_path)))

    assert isinstance(artifact.documentos, list)
    assert all(isinstance(d, str) for d in artifact.documentos)
    assert len(artifact.documentos) > 0
    assert isinstance(artifact.titulos, list)
    assert all(isinstance(t, str) for t in artifact.titulos)
    assert len(artifact.titulos) == len(artifact.documentos)


# ---------------------------------------------------------------------------
# 6.8 — conteudo do artifact: parametros
# ---------------------------------------------------------------------------

def test_artifact_conteudo_parametros(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["bow", "tfidf"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    artifact = ArtifactFase2.load(str(_artifact_path(tmp_path)))

    assert isinstance(artifact.parametros, dict)
    assert "bow" in artifact.parametros
    assert "tfidf" in artifact.parametros
    assert "max_features" in artifact.parametros["bow"]
    assert "min_df" in artifact.parametros["bow"]
    assert "max_features" in artifact.parametros["tfidf"]
    assert "min_df" in artifact.parametros["tfidf"]


# ---------------------------------------------------------------------------
# 6.9 — serializacao e desserializacao (round-trip)
# ---------------------------------------------------------------------------

def test_artifact_save_load(tmp_path):
    bow = csr_matrix(np.array([[1, 0, 2], [0, 3, 1]], dtype=np.int64))
    artifact = ArtifactFase2(
        bow_matrix=bow,
        tfidf_matrix=None,
        bow_vectorizer=None,
        tfidf_vectorizer=None,
        documentos=["doc um", "doc dois"],
        titulos=["Titulo 1", "Titulo 2"],
        parametros={"bow": {"max_features": 50, "min_df": 1}, "tfidf": {}},
    )

    caminho = str(tmp_path / "test.lpf2")
    artifact.save(caminho)
    loaded = ArtifactFase2.load(caminho)

    assert np.array_equal(loaded.bow_matrix.toarray(), artifact.bow_matrix.toarray())
    assert loaded.documentos == artifact.documentos
    assert loaded.titulos == artifact.titulos
    assert loaded.parametros == artifact.parametros


# ---------------------------------------------------------------------------
# 6.10 — word2vec nao interfere no artifact
# ---------------------------------------------------------------------------

def test_word2vec_nao_aparece_no_artifact(sample_parquet, tmp_path):
    config = _pipeline_config(tmp_path, ["bow", "tfidf", "word2vec"])
    pipeline = PipelineEmbeddings(config, sample_parquet)
    pipeline.executar()

    artifact = ArtifactFase2.load(str(_artifact_path(tmp_path)))

    assert not hasattr(artifact, "word2vec_matrix")
    assert artifact.bow_matrix is not None
    assert artifact.tfidf_matrix is not None
    assert artifact.bow_vectorizer is not None
    assert artifact.tfidf_vectorizer is not None

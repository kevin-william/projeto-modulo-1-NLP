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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from shared.artifacts import ArtifactFase2

from fase3_config import NUM_TOPICOS, PARAMS_LSA, PARAMS_LDA_GENSIM, PARAMS_NMF

_PARAMS_DICT_TESTE = dict(no_below=1, no_above=1.0, keep_n=200)
_PARAMS_LDA_TESTE = {"passes": 2, "iterations": 10, "random_state": 42}


def criar_artefato_sintetico(n_docs=30):
    base_docs = [
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
    ]
    docs = (base_docs * ((n_docs // len(base_docs)) + 1))[:n_docs]
    titulos = [f"Article {i}" for i in range(n_docs)]
    tokens = [doc.split() for doc in docs]

    vectorizer = TfidfVectorizer(max_features=50)
    tfidf_matrix = vectorizer.fit_transform(docs)

    bow_vectorizer = CountVectorizer(max_features=50)
    bow_matrix = bow_vectorizer.fit_transform(docs)

    return ArtifactFase2(
        documentos=docs,
        titulos=titulos,
        tokens=tokens,
        parametros={"bow": {"max_features": 50}, "tfidf": {"max_features": 50}},
        bow_matrix=bow_matrix,
        tfidf_matrix=tfidf_matrix,
        bow_vectorizer=bow_vectorizer,
        tfidf_vectorizer=vectorizer,
    )


class TestPipeline:

    def test_pipeline_completo_com_artefato_sintetico_sem_erros(self, tmp_path):
        artefato = criar_artefato_sintetico(n_docs=15)

        caminho_artefato = os.path.join(str(tmp_path), "test_artifact.lpf2")
        artefato.save(caminho_artefato)

        diretorio_plots = os.path.join(str(tmp_path), "plots")
        os.makedirs(diretorio_plots, exist_ok=True)

        # 1. Carregar
        from carregador_artefato import carregar_artefato_fase2
        carregado = carregar_artefato_fase2(caminho_artefato)
        assert len(carregado.documentos) == 15
        assert carregado.tokens is not None

        # 2. EDA
        from eda import executar_eda
        metricas = executar_eda(carregado.documentos, carregado.titulos, diretorio_plots)
        assert metricas["num_documentos"] == 15
        assert metricas["tamanho_vocabulario"] > 0

        # 3. Tokens
        tokens = carregado.tokens

        # 4. Treinar modelos
        vocabulario = carregado.tfidf_vectorizer.get_feature_names_out().tolist()

        from modelos_topicos import ModeloLSA, ModeloLDA, ModeloNMF
        from avaliacao import calcular_coerencia_gensim, comparar_modelos

        # LSA
        lsa = ModeloLSA(NUM_TOPICOS, PARAMS_LSA)
        lsa.treinar(carregado.tfidf_matrix)
        topicos_lsa = lsa.obter_topicos(vocabulario, 5)

        # LDA Gensim
        lda = ModeloLDA(NUM_TOPICOS, _PARAMS_LDA_TESTE)
        lda.treinar(tokens, **_PARAMS_DICT_TESTE)
        topicos_lda = lda.obter_topicos(top_n=5)
        perplexidade = lda.obter_perplexidade()

        # NMF
        nmf = ModeloNMF(NUM_TOPICOS, PARAMS_NMF)
        nmf.treinar(carregado.tfidf_matrix)
        topicos_nmf = nmf.obter_topicos(vocabulario, 5)

        # 5. Avaliar
        coef_lsa = calcular_coerencia_gensim(topicos_lsa, tokens)
        coef_lda = lda.obter_coerencia()
        coef_nmf = calcular_coerencia_gensim(topicos_nmf, tokens)

        assert isinstance(coef_lsa, float)
        assert isinstance(coef_lda, float)
        assert isinstance(coef_nmf, float)
        assert isinstance(perplexidade, float)

        # 6. DataFrame comparação
        resultados = {
            "LSA": {"num_topicos": NUM_TOPICOS, "coerencia": coef_lsa, "perplexidade": None},
            "LDA": {"num_topicos": NUM_TOPICOS, "coerencia": coef_lda, "perplexidade": perplexidade},
            "NMF": {"num_topicos": NUM_TOPICOS, "coerencia": coef_nmf, "perplexidade": None},
        }
        df = comparar_modelos(resultados)
        assert len(df) == 3
        assert list(df.columns) == ["modelo", "num_topicos", "coerencia_media", "perplexidade"]

        # 7. Visualizações
        from visualizacao import (
            plotar_top_palavras_por_topico,
            plotar_distribuicao_topicos_documentos,
            plotar_comparacao_metricas,
            plotar_wordcloud_topicos,
        )

        plotar_top_palavras_por_topico(lsa.obter_topicos_com_pesos(vocabulario, 5), "lsa", diretorio_plots)
        plotar_top_palavras_por_topico(lda.obter_topicos_com_pesos(top_n=5), "lda", diretorio_plots)
        plotar_top_palavras_por_topico(nmf.obter_topicos_com_pesos(vocabulario, 5), "nmf", diretorio_plots)

        plotar_distribuicao_topicos_documentos(lsa.obter_distribuicao_documentos(),
                                               carregado.titulos, "lsa", diretorio_plots)
        plotar_distribuicao_topicos_documentos(lda.obter_distribuicao_documentos(),
                                               carregado.titulos, "lda", diretorio_plots)
        plotar_distribuicao_topicos_documentos(nmf.obter_distribuicao_documentos(),
                                               carregado.titulos, "nmf", diretorio_plots)

        plotar_comparacao_metricas(df, diretorio_plots)

        plotar_wordcloud_topicos(lsa.obter_topicos_com_pesos(vocabulario, 5), "lsa", diretorio_plots)
        plotar_wordcloud_topicos(lda.obter_topicos_com_pesos(top_n=5), "lda", diretorio_plots)
        plotar_wordcloud_topicos(nmf.obter_topicos_com_pesos(vocabulario, 5), "nmf", diretorio_plots)

        # 8. Verificar arquivos gerados
        assert os.path.exists(os.path.join(diretorio_plots, "lsa_topicos.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "lda_topicos.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "nmf_topicos.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "lsa_heatmap.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "lda_heatmap.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "nmf_heatmap.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "comparacao_coerencia.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "lsa_wordcloud_topicos.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "lda_wordcloud_topicos.png"))
        assert os.path.exists(os.path.join(diretorio_plots, "nmf_wordcloud_topicos.png"))

    def test_integracao_carregador_eda_modelos(self, tmp_path):
        artefato = criar_artefato_sintetico(n_docs=10)

        caminho_artefato = os.path.join(str(tmp_path), "int_artifact.lpf2")
        artefato.save(caminho_artefato)

        # Carregar
        from carregador_artefato import carregar_artefato_fase2
        carregado = carregar_artefato_fase2(caminho_artefato)

        # EDA
        from eda import executar_eda
        executar_eda(carregado.documentos, carregado.titulos, str(tmp_path))

        tokens = carregado.tokens

        # Modelos
        from modelos_topicos import ModeloLSA, ModeloLDA, ModeloNMF

        vocabulario = carregado.tfidf_vectorizer.get_feature_names_out().tolist()

        lsa = ModeloLSA(5, {"random_state": 42})
        lsa.treinar(carregado.tfidf_matrix)
        assert lsa.obter_distribuicao_documentos().shape[0] == 10

        lda = ModeloLDA(5, _PARAMS_LDA_TESTE)
        lda.treinar(tokens, **_PARAMS_DICT_TESTE)
        assert lda.obter_distribuicao_documentos().shape[0] == 10
        assert isinstance(lda.obter_perplexidade(), float)

        nmf = ModeloNMF(5, {"random_state": 42})
        nmf.treinar(carregado.tfidf_matrix)
        assert nmf.obter_distribuicao_documentos().shape[0] == 10

        topicos_lsa = lsa.obter_topicos(vocabulario, 5)
        topicos_lda = lda.obter_topicos(top_n=5)
        topicos_nmf = nmf.obter_topicos(vocabulario, 5)

        from avaliacao import calcular_coerencia_gensim
        assert isinstance(calcular_coerencia_gensim(topicos_lsa, tokens), float)
        assert isinstance(lda.obter_coerencia(), float)
        assert isinstance(calcular_coerencia_gensim(topicos_nmf, tokens), float)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pandas as pd
import pytest
from embedding_pipeline import PipelineEmbeddings


@pytest.fixture
def sample_parquet(tmp_path):
    # "de" e "que" sao stopwords portuguesas; "mundo", "python", etc. nao sao.
    df = pd.DataFrame({
        "id_artigo": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        "id_token": [1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 4],
        "token": ["ola", "mundo", "de", "!", "python", "linguagem", "que", "processamento", "texto", "dados", "de"],
        "pos": ["NOUN", "NOUN", "ADP", "PUNCT", "NOUN", "NOUN", "SCONJ", "NOUN", "NOUN", "NOUN", "ADP"],
        "tag": ["N", "N", "PP", "PU", "N", "N", "C", "N", "N", "N", "PP"],
        "lema": ["ola", "mundo", "de", "!", "python", "linguagem", "que", "processamento", "texto", "dados", "de"],
        "processado": ["ola", "mundo", "de", "!", "python", "linguagem", "que", "processamento", "texto", "dados", "de"],
        "relacao_dependencia": ["nsubj", "obj", "case", "punct", "nsubj", "obj", "mark", "nsubj", "obj", "obj", "case"],
        "token_cabeca": ["", "", "", "", "", "", "", "", "", "", ""],
        "entidade": ["", "", "", "", "", "", "", "", "", "", ""],
        "rotulo_entidade": ["", "", "", "", "", "", "", "", "", "", ""],
        "titulo": ["Doc1"] * 4 + ["Doc2"] * 3 + ["Doc3"] * 4,
        "url": ["http://1"] * 4 + ["http://2"] * 3 + ["http://3"] * 4,
    })
    parquet_path = tmp_path / "sample.parquet"
    df.to_parquet(parquet_path, index=False)
    return str(parquet_path)


class TestEmbeddingPipeline:
    def test_load_and_prepare(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()
        assert len(pipeline.documentos) == 3
        assert pipeline.titulos_documentos == ["Doc1", "Doc2", "Doc3"]

    def test_run_bow(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        engines = pipeline.executar()
        assert "bow" in engines
        assert "bow" in pipeline.vetorizadores

    def test_run_multiple_methods(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow", "tfidf"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "PARAMS_TFIDF": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        engines = pipeline.executar()
        assert "bow" in engines
        assert "tfidf" in engines

    def test_search_text_bow(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline.executar()

        results = pipeline.buscar_texto("bow", "mundo", top_k=2)
        assert len(results) > 0

    def test_search_text_unknown_method(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline.executar()

        results = pipeline.buscar_texto("nonexistent", "query")
        assert results == []

    def test_stopword_removal_habilitado(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": True,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()

        # "de" e "que" sao stopwords portuguesas e nao devem aparecer nos documentos
        for doc in pipeline.documentos:
            tokens = doc.split()
            assert "de" not in tokens
            assert "que" not in tokens

        # tokens semanticos relevantes devem ser preservados
        todos_tokens = " ".join(pipeline.documentos).split()
        assert "mundo" in todos_tokens or "python" in todos_tokens or "processamento" in todos_tokens

    def test_stopword_removal_desabilitado(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()

        # Com stopwords desabilitado, "de" deve aparecer (POS=ADP, nao eh PUNCT)
        todos_tokens = " ".join(pipeline.documentos).split()
        assert "de" in todos_tokens

    def test_pos_filter_habilitado(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": False,
            "POS_TAGS_PERMITIDOS": ["NOUN", "VERB", "ADJ", "ADV"],
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()

        # "de" (ADP) e "que" (SCONJ) devem ser removidos pela whitelist de POS
        todos_tokens = " ".join(pipeline.documentos).split()
        assert "de" not in todos_tokens
        assert "que" not in todos_tokens

        # tokens NOUN devem permanecer
        assert "mundo" in todos_tokens or "python" in todos_tokens

    def test_preprocessar_consulta_remove_stopwords(self, sample_parquet):
        config = {
            "METODOS_EMBEDDING": ["bow"],
            "PARAMS_BOW": {"max_features": 100, "min_df": 1},
            "TOP_K_RESULTADOS": 5,
            "HABILITAR_TSNE": False,
            "HABILITAR_REMOCAO_STOPWORDS": True,
        }
        pipeline = PipelineEmbeddings(config, sample_parquet)
        pipeline._carregar_e_preparar()  # carrega self._stopwords

        tokens = pipeline._preprocessar_consulta_tokens("processamento de linguagem natural")
        assert "de" not in tokens
        assert "processamento" in tokens
        assert "linguagem" in tokens

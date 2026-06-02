import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
from vectorizers.bow_vectorizer import VetorizadorBow


class TestVetorizadorBow:
    def test_fit_transform(self):
        docs = ["ola mundo programacao", "python linguagem natural", "processamento texto python"]
        bow = VetorizadorBow(max_features=100, min_df=1)
        matrix = bow.fit_transform(docs)
        assert matrix.shape[0] == 3
        assert matrix.shape[1] >= 6
        assert len(bow.get_feature_names()) >= 6

    def test_transform(self):
        docs = ["ola mundo", "python e legal"]
        bow = VetorizadorBow()
        bow.fit_transform(docs)
        result = bow.transform(["mundo python"])
        assert result.shape[0] == 1

    def test_transform_before_fit(self):
        bow = VetorizadorBow()
        with pytest.raises(RuntimeError):
            bow.transform(["teste"])

    def test_empty_documents(self):
        bow = VetorizadorBow(min_df=1)
        with pytest.raises(ValueError):
            bow.fit_transform([""])

    def test_get_feature_names_before_fit(self):
        bow = VetorizadorBow()
        assert bow.get_feature_names() == []

    def test_stop_words_excludes_term(self):
        docs = ["python e uma linguagem", "python java ruby"]
        bow = VetorizadorBow(stop_words=["python"])
        matrix = bow.fit_transform(docs)
        features = list(bow.get_feature_names())
        assert "python" not in features

    def test_max_df_excludes_frequent_terms(self):
        docs = ["python e bom", "python e rapido", "python e util"]
        # "python" e "e" aparecem em 100% dos docs; max_df=0.5 deve remove-los
        bow = VetorizadorBow(max_df=0.5)
        matrix = bow.fit_transform(docs)
        features = list(bow.get_feature_names())
        assert "python" not in features

    def test_max_df_default_keeps_frequent_terms(self):
        docs = ["python e bom", "python e rapido"]
        bow = VetorizadorBow(max_df=1.0)
        bow.fit_transform(docs)
        features = list(bow.get_feature_names())
        assert "python" in features

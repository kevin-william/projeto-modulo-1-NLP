from sklearn.feature_extraction.text import TfidfVectorizer
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class VetorizadorTfidf:
    def __init__(self, max_features=None, min_df=1, norm="l2", max_df=1.0, stop_words=None):
        self.max_features = max_features
        self.min_df = min_df
        self.norm = norm
        self.max_df = max_df
        self.stop_words = stop_words
        self.vectorizer = None

    def fit_transform(self, documents):
        logger.info("Treinando TF-IDF: max_features=%s, min_df=%s, norm=%s, max_df=%s",
                     self.max_features, self.min_df, self.norm, self.max_df)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            norm=self.norm,
            max_df=self.max_df,
            stop_words=self.stop_words,
        )
        matrix = self.vectorizer.fit_transform(documents)
        logger.info("TF-IDF treinado: vocab_size=%d, matrix_shape=%s",
                     len(self.vectorizer.vocabulary_), matrix.shape)
        return matrix

    def transform(self, documents):
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF nao foi treinado. Execute fit_transform primeiro.")
        return self.vectorizer.transform(documents)

    def get_feature_names(self):
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()

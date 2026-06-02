from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class CalculadoraCTfIdf:
    """Calcula c-TF-IDF (class-based TF-IDF) reutilizando o IDF de um TfidfVectorizer ja treinado.

    Arquitetura conforme Aula 04:
    - Para cada categoria, concatena todos os documentos em um super-documento.
    - Calcula TF raw usando CountVectorizer com o vocabulario do TF-IDF ja fitado.
    - Multiplica TF normalizado pelo IDF do vectorizer treinado.
    - Retorna scores por categoria como pd.Series ordenada (maior score primeiro).
    """

    def __init__(self, tfidf_vetorizador):
        self.tfidf_vetorizador = tfidf_vetorizador

    def calcular(self, documentos, categorias):
        """Calcula c-TF-IDF para cada categoria.

        Args:
            documentos: list[str] — textos preparados (um por artigo).
            categorias: list[str] — categoria de cada documento (mesmo indice).

        Returns:
            dict[str, pd.Series] — {categoria: scores ordenados por relevancia}.
        """
        idf = self.tfidf_vetorizador.vectorizer.idf_
        vocab = self.tfidf_vetorizador.vectorizer.vocabulary_
        feature_names = self.tfidf_vetorizador.get_feature_names()

        count_vec = CountVectorizer(vocabulary=vocab)

        grupos = defaultdict(list)
        for doc, cat in zip(documentos, categorias):
            grupos[cat].append(doc)

        resultado = {}
        for categoria, docs in sorted(grupos.items()):
            super_doc = " ".join(docs)
            tf_matrix = count_vec.transform([super_doc])
            tf_array = tf_matrix.toarray()[0].astype(float)

            total = tf_array.sum()
            if total > 0:
                tf_array = tf_array / total

            c_tf_idf = tf_array * idf
            scores = pd.Series(c_tf_idf, index=feature_names).sort_values(ascending=False)
            resultado[categoria] = scores
            logger.info("c-TF-IDF calculado para categoria '%s': %d docs, top palavra='%s'",
                        categoria, len(docs), scores.index[0] if len(scores) > 0 else "-")

        return resultado

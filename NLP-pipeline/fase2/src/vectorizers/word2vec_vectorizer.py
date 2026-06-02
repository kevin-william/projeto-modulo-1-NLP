import numpy as np
from gensim.models import Word2Vec
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class VetorizadorWord2Vec:
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=30, seed=42, workers=1):
        self.params = {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "seed": seed,
            "workers": workers,
        }
        self.model = None
        self.vector_size = vector_size

    def fit(self, sentences):
        logger.info("Treinando Word2Vec: params=%s", self.params)
        processed = [s for s in sentences if len(s) > 0]
        if not processed:
            logger.warning("Nenhuma sentenca valida para Word2Vec. Criando modelo dummy.")
            self.model = None
            return self

        self.model = Word2Vec(
            sentences=processed,
            vector_size=self.params["vector_size"],
            window=self.params["window"],
            min_count=self.params["min_count"],
            epochs=self.params["epochs"],
            seed=self.params["seed"],
            workers=self.params["workers"],
        )
        vocab_size = len(self.model.wv)
        logger.info("Word2Vec treinado: vocab_size=%d, vector_size=%d", vocab_size, self.vector_size)
        return self

    def obter_vetor_sentenca(self, tokens):
        if self.model is None:
            return np.zeros(self.vector_size)

        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vectors:
            return np.zeros(self.vector_size)
        return np.mean(vectors, axis=0)

    def obter_embeddings_medios_documentos(self, sentences):
        embeddings = []
        for tokens in sentences:
            vec = self.obter_vetor_sentenca(tokens)
            embeddings.append(vec)
        result = np.array(embeddings)
        logger.info("Document embeddings gerados: shape=%s", result.shape)
        return result

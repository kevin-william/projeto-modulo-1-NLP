import numpy as np
from sklearn.decomposition import TruncatedSVD

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class ModeloLSA:
    def __init__(self, num_topicos, params=None):
        self.num_topicos = num_topicos
        self.params = params or {}
        self.modelo = None
        self.matriz_documento_topico = None

    def treinar(self, matriz_tfidf):
        logger.info("Treinando LSA com %d topicos sobre matriz TF-IDF shape=%s", self.num_topicos, matriz_tfidf.shape)
        self.modelo = TruncatedSVD(n_components=self.num_topicos, **self.params)
        self.matriz_documento_topico = self.modelo.fit_transform(matriz_tfidf)
        logger.info("LSA treinado. Matriz doc-topico shape=%s", self.matriz_documento_topico.shape)
        return self

    def obter_topicos(self, vocabulario, top_n=10):
        if self.modelo is None:
            raise RuntimeError("Modelo LSA nao foi treinado ainda")

        topicos = []
        for indice_topico in range(self.num_topicos):
            pesos = self.modelo.components_[indice_topico]
            indices_top = np.argsort(pesos)[::-1][:top_n]
            palavras = [vocabulario[i] for i in indices_top if i < len(vocabulario)]
            topicos.append(palavras)
        return topicos

    def obter_topicos_com_pesos(self, vocabulario, top_n=10):
        if self.modelo is None:
            raise RuntimeError("Modelo LSA nao foi treinado ainda")

        topicos = []
        for indice_topico in range(self.num_topicos):
            pesos = self.modelo.components_[indice_topico]
            indices_top = np.argsort(pesos)[::-1][:top_n]
            palavras_pesos = [
                (vocabulario[i], float(pesos[i]))
                for i in indices_top
                if i < len(vocabulario)
            ]
            topicos.append(palavras_pesos)
        return topicos

    def obter_distribuicao_documentos(self):
        if self.matriz_documento_topico is None:
            raise RuntimeError("Modelo LSA nao foi treinado ainda")
        return self.matriz_documento_topico

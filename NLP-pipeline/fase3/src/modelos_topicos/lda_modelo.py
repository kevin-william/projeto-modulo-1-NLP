import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class ModeloLDA:
    def __init__(self, num_topicos, params=None):
        self.num_topicos = num_topicos
        self.params = params or {}
        self.modelo = None
        self.dictionary = None
        self.corpus = None
        self.tokens = None

    def treinar(self, tokens, no_below=2, no_above=0.9, keep_n=10000):
        """Treina LDA Gensim a partir de uma lista de listas de tokens.

        Args:
            tokens: List[List[str]] — tokens por documento.
            no_below: ignora termos em menos de N documentos.
            no_above: ignora termos em mais de X% dos documentos.
            keep_n: mantém apenas os N termos mais frequentes.
        """
        logger.info("Construindo Dictionary a partir de %d documentos", len(tokens))
        self.tokens = tokens
        self.dictionary = Dictionary(tokens)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        logger.info("Dictionary: %d termos apos filter_extremes", len(self.dictionary))

        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokens]

        params = {k: v for k, v in self.params.items()}
        random_state = params.pop("random_state", 42)

        logger.info("Treinando LdaModel com %d topicos", self.num_topicos)
        self.modelo = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topicos,
            random_state=random_state,
            **params,
        )
        logger.info("LDA Gensim treinado.")
        return self

    def obter_topicos(self, top_n=10):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        topicos = []
        for i in range(self.num_topicos):
            palavras = [palavra for palavra, _ in self.modelo.show_topic(i, topn=top_n)]
            topicos.append(palavras)
        return topicos

    def obter_topicos_com_pesos(self, top_n=10):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        topicos = []
        for i in range(self.num_topicos):
            palavras_pesos = [(palavra, float(peso)) for palavra, peso in self.modelo.show_topic(i, topn=top_n)]
            topicos.append(palavras_pesos)
        return topicos

    def obter_distribuicao_documentos(self):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        n_docs = len(self.corpus)
        matriz = np.zeros((n_docs, self.num_topicos))
        for i, bow in enumerate(self.corpus):
            for topico_id, prob in self.modelo.get_document_topics(bow, minimum_probability=0.0):
                matriz[i, topico_id] = prob
        return matriz

    def obter_perplexidade(self):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        perplexidade = float(self.modelo.log_perplexity(self.corpus))
        logger.info("Perplexidade LDA (log): %.4f", perplexidade)
        return perplexidade

    def obter_coerencia(self, metodo="c_v"):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        cm = CoherenceModel(
            model=self.modelo,
            texts=self.tokens,
            dictionary=self.dictionary,
            coherence=metodo,
        )
        coerencia = float(cm.get_coherence())
        logger.info("Coerencia LDA (%s): %.4f", metodo, coerencia)
        return coerencia

    def obter_corpus(self):
        return self.corpus

    def obter_dictionary(self):
        return self.dictionary

    def obter_modelo(self):
        return self.modelo

    def salvar_modelo(self, caminho):
        if self.modelo is None:
            raise RuntimeError("Modelo LDA nao foi treinado ainda")
        self.modelo.save(caminho)
        self.dictionary.save(caminho + ".dict")
        logger.info("Modelo LDA salvo em: %s", caminho)

    @classmethod
    def carregar_modelo(cls, caminho, num_topicos):
        instancia = cls(num_topicos=num_topicos)
        instancia.modelo = LdaModel.load(caminho)
        instancia.dictionary = Dictionary.load(caminho + ".dict")
        return instancia

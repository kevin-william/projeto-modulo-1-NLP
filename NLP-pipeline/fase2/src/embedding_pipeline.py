import os
import numpy as np
import pandas as pd
from logger import inicializar_sistema_log
from vectorizers.bow_vectorizer import VetorizadorBow
from vectorizers.tfidf_vectorizer import VetorizadorTfidf
from vectorizers.word2vec_vectorizer import VetorizadorWord2Vec
from vectorizers.ctfidf_vectorizer import CalculadoraCTfIdf
from similarity.cosine_search import MotorBuscaCosseno
from visualization.tsne_plot import VisualizadorTSNE
from visualization.wordcloud_ctfidf import gerar_grid_wordclouds_ctfidf

logger = inicializar_sistema_log(__name__)


class PipelineEmbeddings:
    def __init__(self, configuracoes, caminho_parquet_entrada):
        self.configuracoes = configuracoes
        self.caminho_parquet_entrada = caminho_parquet_entrada
        self.motores_busca = {}
        self.vetorizadores = {}
        self.documentos = []
        self.titulos_documentos = []
        self.documentos_tokenizados = []
        self._stopwords = frozenset()
        self.ctfidf_por_categoria = {}

    def _carregar_e_preparar(self):
        logger.info("Carregando dados de: %s", self.caminho_parquet_entrada)
        dataframe = pd.read_parquet(self.caminho_parquet_entrada)
        logger.info("Parquet carregado: %d linhas, %d colunas", len(dataframe), len(dataframe.columns))

        habilitar_stopwords = self.configuracoes.get("HABILITAR_REMOCAO_STOPWORDS", False)
        pos_permitidos = self.configuracoes.get("POS_TAGS_PERMITIDOS") or []

        if habilitar_stopwords:
            from spacy.lang.pt import STOP_WORDS
            self._stopwords = frozenset(STOP_WORDS)
            logger.info("Remocao de stopwords habilitada: %d stopwords carregadas", len(self._stopwords))

        agrupado = dataframe.groupby("id_artigo")

        for _id_artigo, grupo in agrupado:
            titulo = str(grupo["titulo"].iloc[0])
            tokens_coluna = grupo["processado"].dropna()

            tokens_filtrados = []
            for indice, token_serie in tokens_coluna.items():
                token = str(token_serie).strip().lower()
                pos = str(grupo.loc[indice, "pos"]) if indice in grupo.index else ""

                if not token:
                    continue

                if pos_permitidos:
                    if pos not in pos_permitidos:
                        continue
                elif pos == "PUNCT":
                    continue

                if habilitar_stopwords and token in self._stopwords:
                    continue

                tokens_filtrados.append(token)

            texto_documento = " ".join(tokens_filtrados)
            self.documentos.append(texto_documento)
            self.titulos_documentos.append(titulo)
            self.documentos_tokenizados.append(tokens_filtrados)

        logger.info("Documentos preparados: %d artigos", len(self.documentos))
        return self

    def executar(self):
        self._carregar_e_preparar()

        metodos = self.configuracoes["METODOS_EMBEDDING"]
        logger.info("Metodos configurados: %s", metodos)

        for metodo in metodos:
            logger.info("--- Treinando metodo: %s ---", metodo)
            motor = None
            vetorizador = None

            if metodo == "bow":
                motor, vetorizador = self._treinar_bow()
            elif metodo == "tfidf":
                motor, vetorizador = self._treinar_tfidf()
            elif metodo == "word2vec":
                motor, vetorizador = self._treinar_word2vec()
            else:
                logger.warning("Metodo desconhecido: %s", metodo)
                continue

            if motor and vetorizador:
                self.motores_busca[metodo] = motor
                self.vetorizadores[metodo] = vetorizador

        if self.configuracoes.get("HABILITAR_CTFIDF", True) and "tfidf" in self.vetorizadores:
            self._calcular_ctfidf()

        if self.configuracoes.get("HABILITAR_TSNE", False):
            self._executar_tsne()

        metodos = self.configuracoes["METODOS_EMBEDDING"]
        tem_bow_ou_tfidf = "bow" in metodos or "tfidf" in metodos

        if tem_bow_ou_tfidf:
            caminho_artefato = os.path.join(
                self.configuracoes.get("DIRETORIO_SAIDA", "output"),
                "artifacts",
                "fase2_artifact.lpf2",
            )
            self._exportar_artefato(caminho_artefato)
        else:
            logger.warning("Artefato .lpf2 nao sera gerado: nenhum metodo bow ou tfidf configurado")

        logger.info("Treinamento concluido. %d search engines disponiveis.", len(self.motores_busca))
        return self.motores_busca

    def _categorizar_titulo(self, titulo):
        regras = self.configuracoes.get("REGRAS_CATEGORIAS", [])
        categoria_padrao = self.configuracoes.get("CATEGORIA_PADRAO", "geral")
        titulo_lower = titulo.lower()
        for palavras_chave, categoria in regras:
            if any(kw in titulo_lower for kw in palavras_chave):
                return categoria
        return categoria_padrao

    def _calcular_ctfidf(self):
        regras = self.configuracoes.get("REGRAS_CATEGORIAS", [])
        if not regras:
            logger.info("c-TF-IDF ignorado: REGRAS_CATEGORIAS vazio.")
            return

        categorias = [self._categorizar_titulo(t) for t in self.titulos_documentos]
        categorias_unicas = set(categorias)

        if len(categorias_unicas) < 2:
            cat = next(iter(categorias_unicas), "nenhuma")
            logger.info("c-TF-IDF ignorado: apenas 1 categoria unica ('%s').", cat)
            return

        logger.info("Calculando c-TF-IDF para %d categorias: %s",
                    len(categorias_unicas), sorted(categorias_unicas))

        calculadora = CalculadoraCTfIdf(self.vetorizadores["tfidf"])
        self.ctfidf_por_categoria = calculadora.calcular(self.documentos, categorias)

        caminho_saida = self.configuracoes.get("CAMINHO_SAIDA_WORDCLOUD_CTFIDF")
        if caminho_saida:
            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
            gerar_grid_wordclouds_ctfidf(self.ctfidf_por_categoria, caminho_saida)

    def _treinar_bow(self):
        params = self.configuracoes.get("PARAMS_BOW", {})
        vetorizador = VetorizadorBow(**params)
        vetores_documentos = vetorizador.fit_transform(self.documentos)
        motor = MotorBuscaCosseno("bow", self.documentos)
        motor.fit(vetores_documentos)
        return motor, vetorizador

    def _treinar_tfidf(self):
        params = self.configuracoes.get("PARAMS_TFIDF", {})
        vetorizador = VetorizadorTfidf(**params)
        vetores_documentos = vetorizador.fit_transform(self.documentos)
        motor = MotorBuscaCosseno("tfidf", self.documentos)
        motor.fit(vetores_documentos)
        return motor, vetorizador

    def _treinar_word2vec(self):
        params = self.configuracoes.get("PARAMS_WORD2VEC", {})
        modelo = VetorizadorWord2Vec(**params)
        modelo.fit(self.documentos_tokenizados)
        vetores_documentos = modelo.obter_embeddings_medios_documentos(self.documentos_tokenizados)
        motor = MotorBuscaCosseno("word2vec", self.documentos)
        motor.fit(vetores_documentos)
        return motor, modelo

    def _executar_tsne(self):
        if not self.motores_busca:
            logger.warning("Nenhum search engine para t-SNE")
            return

        primeiro_metodo = list(self.motores_busca.keys())[0]
        motor = self.motores_busca[primeiro_metodo]
        vetores = motor.doc_vectors

        if hasattr(vetores, "toarray"):
            vetores = vetores.toarray()
        if not isinstance(vetores, np.ndarray):
            vetores = np.array(vetores)

        params_tsne = self.configuracoes.get("PARAMS_TSNE", {})
        params_plot_tsne = self.configuracoes.get("PARAMS_PLOT_TSNE", {})
        caminho_saida_tsne = self.configuracoes.get("CAMINHO_SAIDA_TSNE", os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "tsne_plot.png"
        ))

        visualizador = VisualizadorTSNE(**params_tsne)
        embeddings_2d = visualizador.fit_transform(vetores)
        visualizador.plot(embeddings_2d, self.titulos_documentos, caminho_saida_tsne, primeiro_metodo, **params_plot_tsne)

    def _exportar_artefato(self, caminho_saida):
        import sys as _sys
        _shared_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared")
        )
        if _shared_dir not in _sys.path:
            _sys.path.insert(0, _shared_dir)
        from artifacts import ArtifactFase2

        bow_matrix = None
        bow_vectorizer = None
        tfidf_matrix = None
        tfidf_vectorizer = None

        if "bow" in self.vetorizadores:
            bow_matrix = self.motores_busca["bow"].doc_vectors
            bow_vectorizer = self.vetorizadores["bow"].vectorizer
        if "tfidf" in self.vetorizadores:
            tfidf_matrix = self.motores_busca["tfidf"].doc_vectors
            tfidf_vectorizer = self.vetorizadores["tfidf"].vectorizer

        artifact = ArtifactFase2(
            bow_matrix=bow_matrix,
            tfidf_matrix=tfidf_matrix,
            bow_vectorizer=bow_vectorizer,
            tfidf_vectorizer=tfidf_vectorizer,
            documentos=self.documentos,
            titulos=self.titulos_documentos,
            tokens=self.documentos_tokenizados,
            parametros={
                "bow": self.configuracoes.get("PARAMS_BOW", {}),
                "tfidf": self.configuracoes.get("PARAMS_TFIDF", {}),
            },
        )
        artifact.save(caminho_saida)
        logger.info("Artefato salvo: %s", caminho_saida)

    def _preprocessar_consulta_tokens(self, texto):
        """Normaliza uma consulta textual: lowercase, split e remocao de stopwords.
        Nao aplica filtro de POS pois queries nao possuem anotacao morfologica."""
        tokens = texto.strip().lower().split()
        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        return tokens

    def buscar_texto(self, metodo, texto_consulta, top_k=10):
        if metodo not in self.motores_busca:
            logger.warning("Metodo '%s' nao disponivel", metodo)
            return []

        if metodo not in self.vetorizadores:
            logger.warning("Vectorizer para '%s' nao disponivel", metodo)
            return []

        vetorizador = self.vetorizadores[metodo]
        tokens_consulta = self._preprocessar_consulta_tokens(texto_consulta)

        if metodo == "word2vec":
            vetor_consulta = vetorizador.obter_vetor_sentenca(tokens_consulta)
        else:
            consulta_processada = " ".join(tokens_consulta)
            vetor_consulta = vetorizador.transform([consulta_processada])
            if hasattr(vetor_consulta, "toarray"):
                vetor_consulta = vetor_consulta.toarray().flatten()

        return self.motores_busca[metodo].search(vetor_consulta, top_k)

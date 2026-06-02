import os

from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def _obter_classe_artefato():
    classes = []
    try:
        from artifacts import ArtifactFase2 as A1
        classes.append(A1)
    except ImportError:
        pass
    try:
        from shared.artifacts import ArtifactFase2 as A2
        classes.append(A2)
    except ImportError:
        pass
    return classes


def carregar_artefato_fase2(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo de artefato nao encontrado: {caminho}")

    import joblib

    obj = joblib.load(caminho)

    classes = _obter_classe_artefato()
    if not any(isinstance(obj, cls) for cls in classes):
        raise TypeError(f"O arquivo nao contem um ArtifactFase2 valido: {type(obj)}")

    if obj.documentos is None or len(obj.documentos) == 0:
        raise ValueError("Artefato invalido: lista de documentos vazia ou None")

    if obj.titulos is None or len(obj.titulos) == 0:
        raise ValueError("Artefato invalido: lista de titulos vazia ou None")

    if obj.tfidf_matrix is None:
        raise ValueError("Artefato invalido: tfidf_matrix eh None")

    if obj.tfidf_vectorizer is None:
        raise ValueError("Artefato invalido: tfidf_vectorizer eh None")

    logger.info("Artefato carregado com sucesso de: %s", caminho)
    logger.info("Documentos: %d", len(obj.documentos))
    logger.info("Shape TF-IDF matrix: %s", obj.tfidf_matrix.shape)

    if obj.bow_matrix is not None:
        logger.info("Shape BoW matrix: %s", obj.bow_matrix.shape)
    else:
        logger.info("BoW matrix nao disponivel no artefato")

    return obj

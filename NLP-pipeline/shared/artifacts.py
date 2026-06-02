from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class ArtifactFase2:
    """Artefato serializavel produzido pela fase 2 e consumido pela fase 3."""

    documentos: List[str]
    titulos: List[str]
    parametros: dict
    bow_matrix: Optional[csr_matrix] = field(default=None)
    tfidf_matrix: Optional[csr_matrix] = field(default=None)
    bow_vectorizer: Optional[CountVectorizer] = field(default=None)
    tfidf_vectorizer: Optional[TfidfVectorizer] = field(default=None)
    tokens: Optional[List[List[str]]] = field(default=None)

    def save(self, path: str) -> None:
        """Serializa o artefato para o caminho indicado usando joblib."""
        dirpath = os.path.dirname(os.path.abspath(path))
        os.makedirs(dirpath, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Artefato salvo em: %s", os.path.abspath(path))

    @classmethod
    def load(cls, path: str) -> ArtifactFase2:
        """Desserializa um artefato previamente salvo."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arquivo de artefato nao encontrado: {path}"
            )
        obj = joblib.load(path)
        if not isinstance(obj, ArtifactFase2):
            raise TypeError(
                f"O arquivo nao contem um ArtifactFase2 valido: {type(obj)}"
            )
        return obj

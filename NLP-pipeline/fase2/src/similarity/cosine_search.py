import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class MotorBuscaCosseno:
    def __init__(self, name, documents, document_ids=None):
        self.name = name
        self.documents = documents
        self.document_ids = document_ids or list(range(len(documents)))
        self.doc_vectors = None

    def fit(self, doc_vectors):
        self.doc_vectors = doc_vectors
        logger.info("[%s] Search engine pronto: %d documentos, shape=%s",
                     self.name, len(self.documents), self.doc_vectors.shape)

    def search(self, query_vector, top_k=10):
        if self.doc_vectors is None:
            return []

        query_2d = np.array(query_vector).reshape(1, -1)
        similarities = cosine_similarity(query_2d, self.doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            doc_text = self.documents[idx]
            preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
            results.append({
                "index": int(self.document_ids[idx]),
                "score": float(similarities[idx]),
                "document": doc_text,
                "preview": preview,
            })

        logger.info("[%s] Busca concluida: top_k=%d, melhor_score=%.4f",
                     self.name, top_k, results[0]["score"] if results else 0)
        return results

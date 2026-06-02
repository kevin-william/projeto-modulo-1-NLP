import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class VisualizadorTSNE:
    def __init__(self, n_components=2, perplexity=5, n_iter=1000, random_state=42, init="pca"):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.init = init
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            max_iter=n_iter,
            random_state=random_state,
            init=init,
        )

    def fit_transform(self, embeddings):
        logger.info("Executando t-SNE: perplexity=%d, max_iter=%d", self.perplexity, self.n_iter)
        n_samples = embeddings.shape[0]
        actual_perplexity = min(self.perplexity, max(1, n_samples - 1))
        if actual_perplexity != self.perplexity:
            self.tsne.perplexity = actual_perplexity
            logger.info("Perplexidade ajustada de %d para %d", self.perplexity, actual_perplexity)
        result = self.tsne.fit_transform(embeddings)
        logger.info("t-SNE concluido: shape=%s", result.shape)
        return result

    def plot(self, embeddings_2d, labels, output_path, method_name="Embeddings",
             figsize=(12, 10), dpi=150, marker_size=50, annotate_fontsize=7):
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=np.arange(len(embeddings_2d)),
            cmap="tab20",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.3,
            s=marker_size,
        )

        for i, label in enumerate(labels):
            plt.annotate(
                label[:30],
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=annotate_fontsize,
                alpha=0.7,
                textcoords="offset points",
                xytext=(5, 5),
            )

        plt.colorbar(scatter, label="Document index")
        plt.title(f"t-SNE Visualization - {method_name}", fontsize=14)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info("t-SNE plot salvo em: %s", output_path)

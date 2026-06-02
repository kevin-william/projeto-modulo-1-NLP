import os

DIRETORIO_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAMINHO_ARTEFATO_FASE2 = os.path.join(DIRETORIO_BASE, "input", "fase2_artifact.lpf2")

DIRETORIO_SAIDA = os.path.join(DIRETORIO_BASE, "output")

CAMINHO_LOG = os.path.join(DIRETORIO_SAIDA, "fase3_pipeline-v001.log")

DIRETORIO_PLOTS = os.path.join(DIRETORIO_SAIDA, "plots")

NUM_TOPICOS = 10

TOP_N_PALAVRAS = 10

PARAMS_LSA = {"random_state": 42}

PARAMS_LDA_GENSIM = {
    "passes": 20,
    "iterations": 1000,
    "alpha": "auto",
    "eta": "auto",
    "random_state": 42,
}

PARAMS_DICTIONARY = {
    "no_below": 3,
    "no_above": 0.80,
    "keep_n": 1000,
}

PARAMS_NMF = {
    "init": "nndsvda",
    "random_state": 42,
    "max_iter": 500,
}

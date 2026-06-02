"""
Extracao de relacoes entre entidades (triplas SVO e co-ocorrencia).

Alinhado ao codigo do professor (linhas 665-712 do compilado_codigos.txt):
  edges = [[entity.text for entity in doc.ents if entity.label_ in ['ORG']] ...]
  verb = token.text if token.pos_ == 'VERB'
  df = pd.DataFrame(edges, columns=['source', 'target'])
  df['weight'] = 1.
  Counter(chain(*edges)).most_common()
"""

from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Dict, List, Optional

import pandas as pd

from fase4_config import ENTIDADES_RELACAO, WINDOW_MAX_SENTENCAS
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def extrair_triplas_svo(doc) -> List[Dict]:
    """
    Extrai triplas sujeito-verbo-objeto de um doc spaCy.
    Padrao do professor (linhas 684-700): busca por entidades ORG/PERSON/GPE
    e verbo principal via token.pos_ == 'VERB'.
    """
    triplas: List[Dict] = []
    for sent in doc.sents:
        # Entidades relevantes na sentenca
        ents_sent = [
            ent for ent in sent.ents if ent.label_ in ENTIDADES_RELACAO
        ]
        if len(ents_sent) < 2:
            continue
        # Busca verbo principal (padrao do professor)
        verb: Optional[str] = None
        for token in sent:
            if token.pos_ == "VERB" and token.dep_ in (
                "ROOT",
                "ccomp",
                "xcomp",
                "conj",
                "acl",
            ):
                verb = token.lemma_ or token.text
                break
        if verb is None:
            for token in sent:
                if token.pos_ == "VERB":
                    verb = token.lemma_ or token.text
                    break
        if verb is None:
            continue
        # Gera pares de entidades com o verbo
        for i in range(len(ents_sent) - 1):
            ent1 = ents_sent[i]
            ent2 = ents_sent[i + 1]
            triplas.append(
                {
                    "sujeito": ent1.text,
                    "tipo_sujeito": ent1.label_,
                    "verbo": verb,
                    "objeto": ent2.text,
                    "tipo_objeto": ent2.label_,
                    "tipo": "svo",
                }
            )
    return triplas


def extrair_coocorrencias_sentenca(doc) -> List[Dict]:
    """
    Extrai co-ocorrencias de entidades dentro de sentenças e janela de
    sentencas consecutivas.
    """
    cooc: List[Dict] = []
    sentencas = list(doc.sents)
    for i, sent in enumerate(sentencas):
        ents = [ent for ent in sent.ents if ent.label_ in ENTIDADES_RELACAO]
        if len(ents) < 2:
            continue
        for a in range(len(ents) - 1):
            for b in range(a + 1, len(ents)):
                cooc.append(
                    {
                        "sujeito": ents[a].text,
                        "tipo_sujeito": ents[a].label_,
                        "verbo": "coocorre_com",
                        "objeto": ents[b].text,
                        "tipo_objeto": ents[b].label_,
                        "tipo": "coocorrencia",
                    }
                )
    return cooc


def extrair_relacoes_documento(doc) -> List[Dict]:
    """Combina SVO + co-ocorrencias para um documento."""
    triplas = extrair_triplas_svo(doc)
    if not triplas:
        triplas = extrair_coocorrencias_sentenca(doc)
    return triplas


def extrair_relacoes_corpus(documentos: List[str], nlp) -> pd.DataFrame:
    """
    Extrai relacoes de todos os documentos do corpus.
    """
    todos: List[Dict] = []
    for idx, texto in enumerate(documentos):
        doc = nlp(texto[:100_000])
        relacoes = extrair_relacoes_documento(doc)
        for rel in relacoes:
            rel["documento_id"] = idx
            todos.append(rel)
    if not todos:
        return pd.DataFrame(
            columns=[
                "documento_id",
                "sujeito",
                "tipo_sujeito",
                "verbo",
                "objeto",
                "tipo_objeto",
                "tipo",
            ]
        )
    return pd.DataFrame(todos)


def construir_arestas_grafo(relacoes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói DataFrame de arestas a partir das relacoes extraidas.
    Padrao do professor (linhas 703-706):
      df = pd.DataFrame(edges, columns=['source', 'target'])
      df['weight'] = 1.
    """
    if relacoes_df.empty:
        return pd.DataFrame(columns=["source", "target", "relation", "frequencia"])

    # Agrupa por par (source, target) e conta frequencia
    arestas: List[Dict] = []
    for _, row in relacoes_df.iterrows():
        arestas.append(
            {
                "source": row["sujeito"],
                "target": row["objeto"],
                "relation": row["verbo"],
            }
        )

    # Padrao do professor: Counter(chain(*edges)).most_common()
    # Aqui usamos groupby para contar frequencia de cada par
    df_arestas = pd.DataFrame(arestas)
    df_arestas["frequencia"] = 1.0  # peso inicial (linha 704 do professor)
    df_arestas = (
        df_arestas.groupby(["source", "target", "relation"], as_index=False)
        .agg(frequencia=("frequencia", "sum"))
    )
    df_arestas = df_arestas.sort_values("frequencia", ascending=False).reset_index(drop=True)
    return df_arestas


def filtrar_relacoes_relevantes(
    relacoes_df: pd.DataFrame, min_freq: int = 2
) -> pd.DataFrame:
    """Filtra arestas com frequencia abaixo do minimo."""
    if relacoes_df.empty:
        return relacoes_df
    return relacoes_df[relacoes_df["frequencia"] >= min_freq].reset_index(drop=True)

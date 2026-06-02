"""
Analise e visualizacao de entidades nomeadas com spaCy e displaCy.

Alinhado ao codigo do professor (linhas 648-663 e 776-792 do compilado_codigos.txt):
  nlp = spacy.load("pt_core_news_lg")
  for ent in doc.ents:
      if ent.label_ in ["PERSON", "ORG", "GPE"]:
          print(ent.text, ent.label_)
  displacy.render(doc, style="ent", jupyter=True)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd

from fase4_config import (
    AMOSTRA_DISPLACY,
    ENTIDADES_VALIDAS_GRAFO,
    MODELO_SPACY,
    MODELO_SPACY_FALLBACK,
)
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


def carregar_modelo_spacy():
    """Carrega pt_core_news_lg com fallback para pt_core_news_sm."""
    import spacy

    for modelo in (MODELO_SPACY, MODELO_SPACY_FALLBACK):
        try:
            nlp = spacy.load(modelo)
            logger.info("Modelo spaCy carregado: %s", modelo)
            return nlp
        except OSError:
            logger.warning("Modelo '%s' nao encontrado, tentando proximo...", modelo)
    raise RuntimeError(
        "Nenhum modelo spaCy disponivel. Execute: python -m spacy download pt_core_news_lg"
    )


def extrair_entidades_documento(doc) -> List[Dict]:
    """
    Extrai entidades de um doc spaCy.
    Padrao do professor (linhas 661-663): loop for ent in doc.ents
    """
    entidades = []
    for ent in doc.ents:
        entidades.append(
            {
                "texto": ent.text,
                "tipo": ent.label_,
                "inicio": ent.start_char,
                "fim": ent.end_char,
            }
        )
    return entidades


def extrair_entidades_corpus(documentos: List[str], nlp) -> pd.DataFrame:
    """Extrai entidades de todos os documentos e retorna DataFrame."""
    registros: List[Dict] = []
    for idx, texto in enumerate(documentos):
        doc = nlp(texto[:100_000])  # limite de seguranca
        entidades = extrair_entidades_documento(doc)
        for ent in entidades:
            ent["documento_id"] = idx
            registros.append(ent)
    if not registros:
        return pd.DataFrame(
            columns=["documento_id", "texto", "tipo", "inicio", "fim"]
        )
    return pd.DataFrame(registros)


def contar_entidades_por_tipo(entidades_df: pd.DataFrame) -> pd.DataFrame:
    """Conta frequencia de cada tipo de entidade."""
    if entidades_df.empty:
        return pd.DataFrame(columns=["tipo", "total"])
    return (
        entidades_df.groupby("tipo")
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )


def gerar_displacy_html(doc, caminho_saida: str) -> None:
    """
    Gera HTML displaCy de NER para um doc e salva em arquivo.
    Adapta displacy.render(jupyter=True) do professor para salvar em arquivo.
    """
    from spacy import displacy

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    html = displacy.render(doc, style="ent", jupyter=False, page=True)
    with open(caminho_saida, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.debug("displaCy HTML salvo: %s", caminho_saida)


def gerar_displacy_corpus(
    documentos: List[str], nlp, diretorio_saida: str
) -> None:
    """
    Gera HTMLs displaCy para uma amostra dos documentos com mais entidades.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    docs_com_ents: List[tuple] = []
    for idx, texto in enumerate(documentos):
        doc = nlp(texto[:100_000])
        n_ents = len(doc.ents)
        if n_ents > 0:
            docs_com_ents.append((n_ents, idx, doc))

    docs_com_ents.sort(key=lambda x: x[0], reverse=True)
    amostra = docs_com_ents[:AMOSTRA_DISPLACY]

    for rank, (n_ents, idx, doc) in enumerate(amostra, 1):
        nome_arquivo = os.path.join(diretorio_saida, f"amostra_{rank:02d}_doc{idx}.html")
        gerar_displacy_html(doc, nome_arquivo)

    logger.info("displaCy: %d HTMLs gerados em %s", len(amostra), diretorio_saida)


def consolidar_entidades_parquet_spacy(
    entidades_spacy_df: pd.DataFrame,
    parquet_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Consolida entidades extraidas pelo spaCy com as anotadas no parquet da Fase 1.
    """
    frames = []
    if parquet_df is not None and not parquet_df.empty:
        col_texto = "entidade" if "entidade" in parquet_df.columns else None
        col_tipo = "rotulo_entidade" if "rotulo_entidade" in parquet_df.columns else None
        if col_texto and col_tipo:
            sub = parquet_df[[col_texto, col_tipo]].dropna().copy()
            sub = sub[sub[col_texto].str.strip() != ""]
            sub = sub.rename(columns={col_texto: "texto", col_tipo: "tipo"})
            sub["fonte"] = "parquet_fase1"
            frames.append(sub)

    if not entidades_spacy_df.empty:
        sub2 = entidades_spacy_df[["texto", "tipo"]].copy()
        sub2["fonte"] = "spacy"
        frames.append(sub2)

    if not frames:
        return pd.DataFrame(columns=["texto", "tipo", "fonte"])

    consolidado = pd.concat(frames, ignore_index=True)
    consolidado = consolidado[consolidado["tipo"].isin(ENTIDADES_VALIDAS_GRAFO)]
    return consolidado

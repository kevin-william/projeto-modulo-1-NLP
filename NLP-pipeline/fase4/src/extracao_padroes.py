"""
Extracao de padroes textuais com expressoes regulares.

Alinhado ao codigo do professor (linhas 609-625 do compilado_codigos.txt):
  re.findall(email_regex, text)
  re.findall(url_regex,   text)
  re.findall(date_regex,  text)
  re.findall(cpf_regex,   text)
"""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd

from fase4_config import (
    REGEX_CODIGOS,
    REGEX_CPFS,
    REGEX_DATAS,
    REGEX_EMAILS,
    REGEX_URLS,
    REGEX_VALORES,
)


def extrair_emails(texto: str) -> List[str]:
    return re.findall(REGEX_EMAILS, texto)


def extrair_urls(texto: str) -> List[str]:
    return re.findall(REGEX_URLS, texto)


def extrair_datas(texto: str) -> List[str]:
    return re.findall(REGEX_DATAS, texto)


def extrair_cpfs(texto: str) -> List[str]:
    return re.findall(REGEX_CPFS, texto)


def extrair_valores_monetarios(texto: str) -> List[str]:
    return re.findall(REGEX_VALORES, texto)


def extrair_codigos(texto: str) -> List[str]:
    return re.findall(REGEX_CODIGOS, texto)


def extrair_todos_padroes(texto: str) -> Dict[str, List[Dict]]:
    """Extrai todos os padroes de um texto retornando valor e posicao."""
    resultado: Dict[str, List[Dict]] = {
        "email": [],
        "url": [],
        "data": [],
        "cpf": [],
        "valor_monetario": [],
        "codigo": [],
    }
    mapeamento = {
        "email": REGEX_EMAILS,
        "url": REGEX_URLS,
        "data": REGEX_DATAS,
        "cpf": REGEX_CPFS,
        "valor_monetario": REGEX_VALORES,
        "codigo": REGEX_CODIGOS,
    }
    for tipo, regex in mapeamento.items():
        for m in re.finditer(regex, texto):
            resultado[tipo].append(
                {
                    "valor": m.group(),
                    "posicao_inicio": m.start(),
                    "posicao_fim": m.end(),
                }
            )
    return resultado


def extrair_padroes_corpus(
    documentos: List[str], titulos: List[str]
) -> pd.DataFrame:
    """Extrai padroes de todos os documentos e retorna DataFrame consolidado."""
    registros: List[Dict] = []
    for idx, (texto, titulo) in enumerate(zip(documentos, titulos)):
        padroes = extrair_todos_padroes(texto)
        for tipo, ocorrencias in padroes.items():
            for oc in ocorrencias:
                registros.append(
                    {
                        "documento_id": idx,
                        "titulo": titulo,
                        "tipo_padrao": tipo,
                        "valor": oc["valor"],
                        "posicao_inicio": oc["posicao_inicio"],
                        "posicao_fim": oc["posicao_fim"],
                    }
                )
    return pd.DataFrame(
        registros,
        columns=[
            "documento_id",
            "titulo",
            "tipo_padrao",
            "valor",
            "posicao_inicio",
            "posicao_fim",
        ],
    )


def gerar_estatisticas_padroes(df: pd.DataFrame) -> pd.DataFrame:
    """Sumariza contagem por tipo de padrao."""
    if df.empty:
        return pd.DataFrame(columns=["tipo_padrao", "total"])
    return (
        df.groupby("tipo_padrao")
        .size()
        .reset_index(name="total")
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )

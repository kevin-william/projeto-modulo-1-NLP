"""
Distancia de Levenshtein para agrupar variacoes ortograficas de entidades.

Alinhado ao codigo do professor (linhas 627-646 do compilado_codigos.txt):
  import Levenshtein
  df['distance'] = df.apply(lambda x: Levenshtein.distance(x['a'], x['b']), axis=1)
  df.sort_values(by='distance').email.values[0]
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from fase4_config import MAX_LEVENSHTEIN_DISTANCE, NORMALIZAR_CASE, REMOVER_ACENTOS

# Tenta importar a biblioteca C; usa implementacao Python pura como fallback.
try:
    import Levenshtein as _lev_lib

    def _levenshtein(a: str, b: str) -> int:
        return _lev_lib.distance(a, b)

except ImportError:
    def _levenshtein(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + (0 if ca == cb else 1),
                    )
                )
            prev = curr
        return prev[-1]


_SUFIXOS_JURIDICOS = re.compile(
    r"\b(s\.?\s*a\.?|ltda\.?|eireli\.?|mei\.?|s\/a)\b", re.IGNORECASE
)
_ARTIGOS = re.compile(r"\b(de|da|do|das|dos|e|the|of|a|o)\b", re.IGNORECASE)


def normalizar_texto(texto: str) -> str:
    """Normaliza texto removendo acentos, case e sufixos juridicos."""
    s = texto.strip()
    if REMOVER_ACENTOS:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
    if NORMALIZAR_CASE:
        s = s.lower()
    s = _SUFIXOS_JURIDICOS.sub("", s)
    s = _ARTIGOS.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def calcular_distancia_levenshtein(str1: str, str2: str) -> int:
    return _levenshtein(str1, str2)


def calcular_similaridade_normalizada(str1: str, str2: str) -> float:
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(str1, str2)
    return 1.0 - dist / max_len


def agrupar_entidades_fuzzy(
    entidades: List[str], max_distancia: int = MAX_LEVENSHTEIN_DISTANCE
) -> Dict[str, List[str]]:
    """
    Agrupa entidades com distancia de Levenshtein <= max_distancia.
    Retorna mapeamento {entidade_canonica: [variantes]}.
    """
    normalizadas = [normalizar_texto(e) for e in entidades]
    grupos: Dict[int, List[int]] = {}  # indice_canonico -> [indices_variantes]
    visitados: List[bool] = [False] * len(entidades)

    for i in range(len(entidades)):
        if visitados[i]:
            continue
        grupos[i] = [i]
        visitados[i] = True
        for j in range(i + 1, len(entidades)):
            if visitados[j]:
                continue
            dist = _levenshtein(normalizadas[i], normalizadas[j])
            sim = calcular_similaridade_normalizada(normalizadas[i], normalizadas[j])
            if dist <= max_distancia or (dist == 3 and sim > 0.6):
                grupos[i].append(j)
                visitados[j] = True

    resultado: Dict[str, List[str]] = {}
    for canonico_idx, membros in grupos.items():
        canonica = entidades[canonico_idx]
        variantes = [entidades[m] for m in membros if m != canonico_idx]
        resultado[canonica] = variantes
    return resultado


def encontrar_mais_proximo(
    entrada: str, candidatos: List[str]
) -> Tuple[str, int, float]:
    """Encontra o candidato mais proximo usando Levenshtein."""
    if not candidatos:
        return ("", -1, 0.0)
    entrada_norm = normalizar_texto(entrada)
    df = pd.DataFrame({"candidato": candidatos, "entrada": [entrada] * len(candidatos)})
    df["norma_candidato"] = df["candidato"].apply(normalizar_texto)
    # Padrao do professor: df.apply(lambda x: Levenshtein.distance(...), axis=1)
    df["distance"] = df.apply(
        lambda x: _levenshtein(x["norma_candidato"], entrada_norm), axis=1
    )
    df = df.sort_values(by="distance")
    melhor = df.iloc[0]
    sim = calcular_similaridade_normalizada(melhor["norma_candidato"], entrada_norm)
    return (melhor["candidato"], int(melhor["distance"]), sim)


def normalizar_entidades(
    entidades: List[str], max_distancia: int = MAX_LEVENSHTEIN_DISTANCE
) -> pd.DataFrame:
    """
    Retorna DataFrame com colunas:
    entidade_original, entidade_canonica, distancia_levenshtein, similaridade, grupo_id
    """
    grupos = agrupar_entidades_fuzzy(entidades, max_distancia)
    registros: list = []
    for grupo_id, (canonica, variantes) in enumerate(grupos.items()):
        norm_can = normalizar_texto(canonica)
        registros.append(
            {
                "entidade_original": canonica,
                "entidade_canonica": canonica,
                "distancia_levenshtein": 0,
                "similaridade": 1.0,
                "grupo_id": grupo_id,
            }
        )
        for variante in variantes:
            norm_var = normalizar_texto(variante)
            dist = _levenshtein(norm_var, norm_can)
            sim = calcular_similaridade_normalizada(norm_var, norm_can)
            registros.append(
                {
                    "entidade_original": variante,
                    "entidade_canonica": canonica,
                    "distancia_levenshtein": dist,
                    "similaridade": sim,
                    "grupo_id": grupo_id,
                }
            )
    return pd.DataFrame(
        registros,
        columns=[
            "entidade_original",
            "entidade_canonica",
            "distancia_levenshtein",
            "similaridade",
            "grupo_id",
        ],
    )


def gerar_matriz_distancias(entidades: List[str]) -> pd.DataFrame:
    """Gera matriz N×N de distancias de Levenshtein normalizadas."""
    n = len(entidades)
    normalizadas = [normalizar_texto(e) for e in entidades]
    matriz = []
    for i in range(n):
        linha = []
        for j in range(n):
            linha.append(_levenshtein(normalizadas[i], normalizadas[j]))
        matriz.append(linha)
    return pd.DataFrame(matriz, index=entidades, columns=entidades)

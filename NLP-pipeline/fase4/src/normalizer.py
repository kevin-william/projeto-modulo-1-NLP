"""
Normalização e agrupamento de entidades por similaridade (Levenshtein).
Consolida: fuzzy_matching.py
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Constantes inline ─────────────────────────────────────────────────────
MAX_LEVENSHTEIN_DISTANCE = 2

logger = logging.getLogger(__name__)

# ── Levenshtein (lib C ou Python puro como fallback) ──────────────────────
try:
    import Levenshtein as _lev_lib

    def _lev(a: str, b: str) -> int:
        return _lev_lib.distance(a, b)

except ImportError:
    def _lev(a: str, b: str) -> int:  # type: ignore[misc]
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


_SUFIXOS = re.compile(
    r"\b(s\.?\s*a\.?|ltda\.?|eireli\.?|mei\.?|s\/a)\b", re.IGNORECASE
)
_ARTIGOS = re.compile(
    r"\b(de|da|do|das|dos|e|the|of|a|o)\b", re.IGNORECASE
)


def _normalizar(texto: str) -> str:
    """Remove acentos, lowercase e sufixos jurídicos para comparação."""
    s = unicodedata.normalize("NFKD", texto.strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _SUFIXOS.sub("", s)
    s = _ARTIGOS.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


class EntityNormalizer:
    """Normaliza entidades via distância de Levenshtein."""

    def __init__(self, threshold: int = MAX_LEVENSHTEIN_DISTANCE):
        self.threshold = threshold

    # ── NORMALIZAÇÃO ──────────────────────────────────────────────────────
    def normalize_entities(self, entities: List[str]) -> Dict[str, List[str]]:
        """Retorna: {forma_canônica: [variação1, variação2, ...]}"""
        normalizadas = [_normalizar(e) for e in entities]
        grupos: Dict[int, List[int]] = {}
        visitados = [False] * len(entities)

        for i in range(len(entities)):
            if visitados[i]:
                continue
            grupos[i] = [i]
            visitados[i] = True
            for j in range(i + 1, len(entities)):
                if visitados[j]:
                    continue
                d = _lev(normalizadas[i], normalizadas[j])
                if d <= self.threshold:
                    grupos[i].append(j)
                    visitados[j] = True

        resultado: Dict[str, List[str]] = {}
        for canon_idx, membros in grupos.items():
            canon = entities[canon_idx]
            variantes = [entities[m] for m in membros if m != canon_idx]
            resultado[canon] = variantes
        return resultado

    def get_canonical_mapping(
        self, groups: Dict[str, List[str]]
    ) -> Dict[str, str]:
        """Retorna: {variação: forma_canônica}"""
        mapping: Dict[str, str] = {}
        for canon, variantes in groups.items():
            mapping[canon] = canon
            for v in variantes:
                mapping[v] = canon
        return mapping

    def normalize_by_type(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona coluna 'canonical_form' agrupando por tipo NER."""
        if entities_df.empty:
            return entities_df

        col_text = (
            "text" if "text" in entities_df.columns
            else "texto"
        )
        col_type = (
            "label" if "label" in entities_df.columns
            else "tipo"
        )
        result = entities_df.copy()
        mapping: Dict[str, str] = {}

        for tipo in result[col_type].unique():
            ents = (
                result.loc[result[col_type] == tipo, col_text]
                .dropna()
                .unique()
                .tolist()
            )
            if ents:
                groups = self.normalize_entities(ents)
                mapping.update(self.get_canonical_mapping(groups))

        result["canonical_form"] = result[col_text].map(
            lambda x: mapping.get(x, x)
        )
        return result

    # ── ANÁLISE ───────────────────────────────────────────────────────────
    def calculate_distance_matrix(self, entities: List[str]) -> np.ndarray:
        """Matriz N×N de distâncias de Levenshtein. Útil para EDA."""
        n = len(entities)
        norms = [_normalizar(e) for e in entities]
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                d = _lev(norms[i], norms[j])
                matrix[i, j] = d
                matrix[j, i] = d
        return matrix

    def get_normalization_stats(
        self,
        original: List[str],
        groups: Dict[str, List[str]],
    ) -> Dict:
        """Retorna: {total_original, total_canonical, reduction_rate, avg_group_size}"""
        total_original = len(original)
        total_canonical = len(groups)
        sizes = [1 + len(v) for v in groups.values()]
        return {
            "total_original": total_original,
            "total_canonical": total_canonical,
            "reduction_rate": round(
                1 - total_canonical / max(total_original, 1), 4
            ),
            "avg_group_size": round(
                sum(sizes) / max(len(sizes), 1), 2
            ),
        }

    def export_mapping(
        self, mapping: Dict[str, str], output_path: str
    ) -> None:
        """Exporta mapeamento de normalização para CSV."""
        dir_out = os.path.dirname(output_path)
        if dir_out:
            os.makedirs(dir_out, exist_ok=True)
        rows = [
            {"entidade_original": k, "entidade_canonica": v}
            for k, v in mapping.items()
        ]
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Mapeamento exportado: %s", output_path)

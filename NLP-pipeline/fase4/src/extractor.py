"""
Extrator unificado: NER (spaCy) + Regex + Relações SVO.
Consolida: ner_analysis.py, extracao_padroes.py, relacoes.py
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Constantes inline ─────────────────────────────────────────────────────
MODELO_SPACY = "pt_core_news_lg"
MODELO_SPACY_FALLBACK = "pt_core_news_sm"

REGEX_EMAILS  = r"[^\s]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+"
REGEX_URLS    = r"https?://(?:www\.)?[a-zA-Z0-9\-_.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
REGEX_DATAS   = r"\b\d{2}/\d{2}/\d{4}\b"
REGEX_CPFS    = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
REGEX_VALORES = r"R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?"
REGEX_CODIGOS = r"\b[A-Z]{2,6}-\d{3,6}\b"

ENTIDADES_RELACAO = {"ORG", "PERSON", "GPE", "LOC", "PER"}

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extrator unificado: NER (spaCy) + Regex + Relações SVO."""

    def __init__(self, model: str = MODELO_SPACY):
        self._model_name = model
        self._nlp = None
        self._stats: Dict[str, int] = {
            "documentos": 0,
            "entidades": 0,
            "padroes": 0,
            "relacoes": 0,
        }

    # ── Carregamento ──────────────────────────────────────────────────────
    def _get_nlp(self):
        if self._nlp is None:
            self._nlp = self._carregar_modelo()
        return self._nlp

    def _carregar_modelo(self):
        import spacy

        for modelo in (self._model_name, MODELO_SPACY_FALLBACK):
            try:
                nlp = spacy.load(modelo)
                logger.info("Modelo spaCy carregado: %s", modelo)
                return nlp
            except OSError:
                logger.warning("Modelo '%s' nao encontrado.", modelo)
        raise RuntimeError(
            "Nenhum modelo spaCy disponivel. "
            "Execute: python -m spacy download pt_core_news_lg"
        )

    # ── NER ───────────────────────────────────────────────────────────────
    def extract_entities(self, text: str) -> List[Dict]:
        """Retorna: [{'text', 'label', 'start', 'end'}, ...]"""
        doc = self._get_nlp()(text[:100_000])
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def extract_entities_from_docs(self, docs: List[str]) -> pd.DataFrame:
        """Colunas: [doc_id, text, label, start, end]"""
        nlp = self._get_nlp()
        records: List[Dict] = []
        for doc_id, text in enumerate(docs):
            spacy_doc = nlp(text[:100_000])
            for ent in spacy_doc.ents:
                records.append(
                    {
                        "doc_id": doc_id,
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )
        self._stats["documentos"] = len(docs)
        self._stats["entidades"] = len(records)
        if not records:
            return pd.DataFrame(columns=["doc_id", "text", "label", "start", "end"])
        return pd.DataFrame(records)

    # ── REGEX ─────────────────────────────────────────────────────────────
    def extract_patterns(self, text: str) -> Dict[str, List[str]]:
        """Chaves: emails, urls, datas, valores, cpf, codigos."""
        return {
            "emails":  re.findall(REGEX_EMAILS, text),
            "urls":    re.findall(REGEX_URLS, text),
            "datas":   re.findall(REGEX_DATAS, text),
            "valores": re.findall(REGEX_VALORES, text),
            "cpf":     re.findall(REGEX_CPFS, text),
            "codigos": re.findall(REGEX_CODIGOS, text),
        }

    def extract_patterns_from_docs(
        self,
        docs: List[str],
        titles: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extrai padrões de todos os documentos e retorna DataFrame."""
        titles = titles or [f"doc_{i}" for i in range(len(docs))]
        records: List[Dict] = []
        for doc_id, (text, title) in enumerate(zip(docs, titles)):
            padroes = self.extract_patterns(text)
            for tipo, vals in padroes.items():
                for val in vals:
                    records.append(
                        {"doc_id": doc_id, "titulo": title, "tipo": tipo, "valor": val}
                    )
        self._stats["padroes"] = len(records)
        if not records:
            return pd.DataFrame(columns=["doc_id", "titulo", "tipo", "valor"])
        return pd.DataFrame(records)

    # ── SVO ───────────────────────────────────────────────────────────────
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Retorna: [(sujeito, verbo, objeto), ...]"""
        doc = self._get_nlp()(text[:100_000])
        triplas = self._svo_from_doc(doc)
        if not triplas:
            triplas = self._cooc_from_doc(doc)
        return [(t["sujeito"], t["verbo"], t["objeto"]) for t in triplas]

    def extract_relations_from_docs(self, docs: List[str]) -> pd.DataFrame:
        """Colunas: [doc_id, sujeito, verbo, objeto, tipo]"""
        nlp = self._get_nlp()
        records: List[Dict] = []
        for doc_id, text in enumerate(docs):
            spacy_doc = nlp(text[:100_000])
            triplas = self._svo_from_doc(spacy_doc)
            if not triplas:
                triplas = self._cooc_from_doc(spacy_doc)
            for t in triplas:
                t["doc_id"] = doc_id
                records.append(t)
        self._stats["relacoes"] = len(records)
        if not records:
            return pd.DataFrame(
                columns=["doc_id", "sujeito", "verbo", "objeto", "tipo"]
            )
        return pd.DataFrame(records)

    def _svo_from_doc(self, doc) -> List[Dict]:
        triplas: List[Dict] = []
        for sent in doc.sents:
            ents = [e for e in sent.ents if e.label_ in ENTIDADES_RELACAO]
            if len(ents) < 2:
                continue
            verb: Optional[str] = None
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ in (
                    "ROOT", "ccomp", "xcomp", "conj", "acl"
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
            for i in range(len(ents) - 1):
                triplas.append(
                    {
                        "sujeito": ents[i].text,
                        "verbo": verb,
                        "objeto": ents[i + 1].text,
                        "tipo": "svo",
                    }
                )
        return triplas

    def _cooc_from_doc(self, doc) -> List[Dict]:
        cooc: List[Dict] = []
        for sent in doc.sents:
            ents = [e for e in sent.ents if e.label_ in ENTIDADES_RELACAO]
            for a in range(len(ents) - 1):
                for b in range(a + 1, len(ents)):
                    cooc.append(
                        {
                            "sujeito": ents[a].text,
                            "verbo": "coocorre_com",
                            "objeto": ents[b].text,
                            "tipo": "coocorrencia",
                        }
                    )
        return cooc

    # ── UTILIDADES ────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict:
        """Estatísticas acumuladas de extração."""
        return dict(self._stats)

    def export_to_csv(
        self,
        entities: pd.DataFrame,
        patterns: pd.DataFrame,
        relations: pd.DataFrame,
        output_dir: str,
    ) -> None:
        """Exporta CSVs para o diretório de saída."""
        os.makedirs(output_dir, exist_ok=True)
        if not entities.empty:
            entities.to_csv(
                os.path.join(output_dir, "entidades_extraidas.csv"),
                index=False,
                encoding="utf-8",
            )
        if not patterns.empty:
            patterns.to_csv(
                os.path.join(output_dir, "padroes_extraidos.csv"),
                index=False,
                encoding="utf-8",
            )
        if not relations.empty:
            relations.to_csv(
                os.path.join(output_dir, "relacoes_extraidas.csv"),
                index=False,
                encoding="utf-8",
            )
        logger.info("CSVs exportados para: %s", output_dir)

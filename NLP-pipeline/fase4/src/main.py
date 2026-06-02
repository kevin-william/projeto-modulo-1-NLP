"""
Pipeline simplificado da Fase 4 — 7 etapas.
Usa os 3 módulos consolidados: extractor, normalizer, graph_builder.

Etapas:
1. Carregar corpus (parquet Fase 1 + artefato Fase 2)
2. Extração de entidades (NER spaCy)
3. Extração de padrões (Regex)
4. Extração de relações (SVO)
5. Normalização de entidades (Levenshtein)
6. Construção do grafo + centralidade
7. Visualizações + exportação de resultados
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Dict

import networkx as nx
import pandas as pd

DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIRETORIO_SCRIPT)
sys.path.insert(0, os.path.join(DIRETORIO_SCRIPT, "..", "..", "shared"))

from shared.utils import ensure_dir
from extractor import TextExtractor
from graph_builder import KnowledgeGraphBuilder
from normalizer import EntityNormalizer


def _setup_logging(log_path: str) -> logging.Logger:
    """Configura logging básico com saída para arquivo e console."""
    lg = logging.getLogger("fase4")
    if not lg.handlers:
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(ch)
        lg.setLevel(logging.INFO)
    return lg


def _carregar_parquet(caminho: str):
    import pandas as pd

    if not os.path.exists(caminho):
        caminho_absoluto = os.path.abspath(caminho)
        raise FileNotFoundError(
            f"Parquet da Fase 1 nao encontrado: {caminho_absoluto}\n"
            "\n"
            "Para corrigir (a escolha do path eh responsabilidade do usuario):\n"
            "  1. Execute a Fase 1 primeiro: cd fase1/src && python main.py\n"
            "  2. Copie o parquet gerado para fase4/input/ com o nome configurado em fase4_config.py\n"
            "  3. Ou ajuste CAMINHO_PARQUET_ENTRADA em fase4/src/fase4_config.py\n"
            "  4. Execute a Fase 4 novamente"
        )
    return pd.read_parquet(caminho)


def _carregar_corpus(caminho_parquet: str, caminho_artefato: str):
    """Carrega documentos do artefato Fase 2 ou reconstrói do parquet."""
    parquet_df = _carregar_parquet(caminho_parquet)

    documentos = None
    titulos = None
    if os.path.exists(caminho_artefato):
        try:
            import joblib
            artefato = joblib.load(caminho_artefato)
            documentos = artefato.documentos
            titulos = artefato.titulos
        except Exception:
            pass

    if documentos is None:
        if "id_artigo" in parquet_df.columns:
            docs, titles = [], []
            for art_id in sorted(parquet_df["id_artigo"].unique()):
                sub = parquet_df[parquet_df["id_artigo"] == art_id]
                col = next((c for c in ("processado", "lema", "token") if c in sub.columns), "token")
                docs.append(" ".join(sub[col].dropna().astype(str)))
                titles.append(
                    sub["titulo"].iloc[0]
                    if "titulo" in sub.columns
                    else f"Documento {art_id}"
                )
            documentos, titulos = docs, titles
        else:
            col = "token" if "token" in parquet_df.columns else parquet_df.columns[0]
            documentos = [" ".join(parquet_df[col].dropna().astype(str))]
            titulos = ["Documento 0"]

    return parquet_df, documentos, titulos


def executar_fase4_principal() -> Dict[str, Any]:
    # Re-importa dentro da função para que monkeypatches de testes sejam visíveis
    from fase4_config import (
        CAMINHO_ARTEFATO_FASE2 as artefato_path,
        CAMINHO_LOG as log_path,
        CAMINHO_PARQUET_ENTRADA as parquet_path,
        DIRETORIO_DISPLACY as displacy_dir,
        DIRETORIO_INPUT as input_dir,
        DIRETORIO_PLOTS as plots_dir,
        DIRETORIO_SAIDA as output_dir,
    )

    ensure_dir(input_dir)
    ensure_dir(output_dir)
    ensure_dir(plots_dir)
    ensure_dir(displacy_dir)

    logger = _setup_logging(log_path)
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("FASE 4 — Pipeline Simplificado (7 etapas)")
    logger.info("Inicio: %s", time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    saidas: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ETAPA 1: Carregar corpus
    # ------------------------------------------------------------------
    logger.info("[Etapa 1] Carregando corpus...")
    parquet_df, documentos, titulos = _carregar_corpus(parquet_path, artefato_path)
    logger.info("Corpus: %d documentos", len(documentos))

    # ------------------------------------------------------------------
    # ETAPAS 2-4: Extração (NER + Regex + SVO)
    # ------------------------------------------------------------------
    logger.info("[Etapas 2-4] Extracao de entidades, padroes e relacoes...")
    extractor = TextExtractor()
    df_entities  = extractor.extract_entities_from_docs(documentos)
    df_patterns  = extractor.extract_patterns_from_docs(documentos, titulos)
    df_relations = extractor.extract_relations_from_docs(documentos)

    stats = extractor.get_statistics()
    logger.info(
        "Entidades: %d | Padroes: %d | Relacoes: %d",
        stats["entidades"], stats["padroes"], stats["relacoes"],
    )

    extractor.export_to_csv(df_entities, df_patterns, df_relations, output_dir)
    saidas["entidades_extraidas"] = os.path.join(output_dir, "entidades_extraidas.csv")
    saidas["relacoes_extraidas"]  = os.path.join(output_dir, "relacoes_extraidas.csv")

    # ------------------------------------------------------------------
    # ETAPA 5: Normalização (Levenshtein)
    # ------------------------------------------------------------------
    logger.info("[Etapa 5] Normalizando entidades (Levenshtein)...")
    normalizer = EntityNormalizer()
    df_normalized = normalizer.normalize_by_type(df_entities)

    if not df_entities.empty:
        entidades_unicas = df_entities["text"].dropna().unique().tolist()
        grupos = normalizer.normalize_entities(entidades_unicas)
        mapa   = normalizer.get_canonical_mapping(grupos)
        s_norm = normalizer.get_normalization_stats(entidades_unicas, grupos)
        logger.info(
            "Normalizacao: %d → %d canonicas (reducao %.1f%%)",
            s_norm["total_original"],
            s_norm["total_canonical"],
            s_norm["reduction_rate"] * 100,
        )
        normalizer.export_mapping(mapa, os.path.join(output_dir, "fuzzy_matches.csv"))
    saidas["fuzzy_matches"] = os.path.join(output_dir, "fuzzy_matches.csv")

    # ------------------------------------------------------------------
    # ETAPA 6: Grafo + centralidade
    # ------------------------------------------------------------------
    logger.info("[Etapa 6] Construindo grafo de conhecimento...")
    graph = KnowledgeGraphBuilder()
    graph.build_from_dataframes(df_relations, df_normalized)

    centralidade = graph.calculate_centrality()
    stats_grafo  = graph.get_graph_stats()
    logger.info(
        "Grafo: %d nos | %d arestas | densidade %.6f",
        stats_grafo["nodes"], stats_grafo["edges"], stats_grafo["density"],
    )

    top10 = graph.get_top_nodes("betweenness", top_n=10)
    if top10:
        logger.info(
            "Top entidades (betweenness): %s",
            ", ".join(f"{n}({v:.3f})" for n, v in top10[:5]),
        )

    nos_csv    = os.path.join(output_dir, "nos_grafo.csv")
    arestas_csv = os.path.join(output_dir, "relacoes_grafo.csv")
    graph.export_csv(nos_csv, arestas_csv)
    saidas["nos_grafo_csv"]    = nos_csv
    saidas["grafo_edges_csv"]  = arestas_csv

    # ------------------------------------------------------------------
    # ETAPA 7: Visualizações + exportação final
    # ------------------------------------------------------------------
    logger.info("[Etapa 7] Gerando visualizacoes e exportando resultados...")
    graph.generate_all_plots(df_normalized, plots_dir)

    resultados = {
        "corpus":  {"documentos": len(documentos)},
        "extracao": stats,
        "grafo":   stats_grafo,
        "top_entidades": [
            {"entidade": n, "betweenness": round(v, 6)} for n, v in top10
        ],
        "pergunta_analitica": (
            "Quais entidades sao mais centrais no corpus e por que?"
        ),
        "resposta": (
            f"As entidades mais centrais por betweenness sao: "
            f"{', '.join(n for n, _ in top10[:3])}. "
            "Essas entidades conectam clusters tematicos distintos, "
            "aparecendo em multiplos contextos ao longo do corpus."
        ) if top10 else "Dados insuficientes para analise.",
    }
    caminho_json = os.path.join(output_dir, "resultados.json")
    with open(caminho_json, "w", encoding="utf-8") as fh:
        json.dump(resultados, fh, ensure_ascii=False, indent=2, default=str)
    saidas["resultados_json"] = caminho_json

    # ------------------------------------------------------------------
    # ETAPA 7b: Visualizações adicionais (visualizacao_grafo.py)
    # ------------------------------------------------------------------
    try:
        from visualizacao_grafo import (
            gerar_infografico_resumo,
            plotar_comparacao_centralidades,
            plotar_comunidades_grafo,
            plotar_distribuicao_centralidade,
            plotar_frequencia_entidades,
            plotar_grafo_matplotlib,
        )

        # Converte Dict[str, Dict[node, float]] → Dict[str, pd.DataFrame]
        centralidade_dfs: Dict[str, Any] = {}
        for metrica, vals in centralidade.items():
            if not vals:
                continue
            records = [
                {
                    "entidade": node,
                    "tipo": graph.graph.nodes.get(node, {}).get("tipo", "MISC"),
                    "frequencia": graph.graph.nodes.get(node, {}).get("frequencia", 0),
                    "centralidade": round(float(v), 6),
                }
                for node, v in vals.items()
            ]
            df_m = (
                pd.DataFrame(records)
                .sort_values("centralidade", ascending=False)
                .reset_index(drop=True)
            )
            df_m["rank"] = df_m.index + 1
            centralidade_dfs[metrica] = df_m

        # DataFrame com coluna "texto" (canonical_form) para plotar_frequencia_entidades
        if not df_normalized.empty and "canonical_form" in df_normalized.columns:
            _col_tipo = "label" if "label" in df_normalized.columns else "tipo"
            df_ents_vis = df_normalized[["canonical_form", _col_tipo]].rename(
                columns={"canonical_form": "texto", _col_tipo: "tipo"}
            ).copy()
        else:
            df_ents_vis = pd.DataFrame(columns=["texto", "tipo"])

        # Comunidades via componentes conexas (fallback sem python-louvain)
        _ug = (
            graph.graph.to_undirected()
            if isinstance(graph.graph, nx.DiGraph)
            else graph.graph
        )
        comunidades_map: Dict[int, Any] = {
            i: list(comp) for i, comp in enumerate(nx.connected_components(_ug))
        }

        plotar_grafo_matplotlib(
            graph.graph,
            centralidade_dfs,
            os.path.join(plots_dir, "grafo_conhecimento.png"),
        )
        plotar_distribuicao_centralidade(
            centralidade_dfs.get("betweenness", pd.DataFrame()),
            "betweenness",
            os.path.join(plots_dir, "centralidade_entidades.png"),
            top_n=10,
        )
        plotar_comparacao_centralidades(
            centralidade_dfs,
            os.path.join(plots_dir, "comparacao_centralidades.png"),
        )
        plotar_frequencia_entidades(
            df_ents_vis,
            os.path.join(plots_dir, "frequencia_entidades.png"),
            top_n=20,
        )
        plotar_comunidades_grafo(
            graph.graph,
            comunidades_map,
            os.path.join(plots_dir, "comunidades_grafo.png"),
            top_n=40,
        )
        gerar_infografico_resumo(
            graph.graph,
            centralidade_dfs,
            df_ents_vis,
            resultados.get("resposta", ""),
            os.path.join(plots_dir, "infografico_resumo.png"),
        )
        logger.info("Visualizacoes adicionais geradas em: %s", plots_dir)

    except Exception as _exc:
        logger.warning("Visualizacoes adicionais falharam: %s", _exc, exc_info=True)

    relatorio_linhas = [
        "=== RELATORIO INTERPRETATIVO — FASE 4 ===\n",
        f"Corpus: {len(documentos)} documentos analisados.",
        f"Entidades extraidas: {stats['entidades']}",
        f"Relacoes extraidas: {stats['relacoes']}",
        f"Grafo: {stats_grafo['nodes']} nos, {stats_grafo['edges']} arestas",
        f"Densidade da rede: {stats_grafo['density']:.6f}",
        "",
        "=== PERGUNTA ANALITICA ===",
        "Quais entidades sao mais centrais no corpus e por que?",
        "",
    ]
    if top10:
        relatorio_linhas.append("Top entidades por betweenness centrality:")
        for n, v in top10:
            grau = graph.graph.degree(n)
            relatorio_linhas.append(f"  {n} — betweenness={v:.4f}, grau={grau}")
        relatorio_linhas += [
            "",
            "Essas entidades conectam clusters tematicos distintos no corpus.",
            "Entidades com alta centralidade de intermediacao aparecem em multiplos",
            "contextos, funcionando como 'hubs' da narrativa.",
        ]

    caminho_relatorio = os.path.join(output_dir, "relatorio_interpretativo.txt")
    with open(caminho_relatorio, "w", encoding="utf-8") as fh:
        fh.write("\n".join(relatorio_linhas))
    saidas["relatorio_interpretativo"] = caminho_relatorio

    elapsed = time.time() - t0
    logger.info("Pipeline concluido em %.1f segundos.", elapsed)
    logger.info("Outputs em: %s", output_dir)
    return saidas


if __name__ == "__main__":
    try:
        executar_fase4_principal()
    except FileNotFoundError as exc:
        logging.error("Erro de arquivo nao encontrado: %s", exc)
        sys.exit(1)
    except OSError as exc:
        logging.error("Erro de sistema de arquivos: %s", exc)
        sys.exit(1)


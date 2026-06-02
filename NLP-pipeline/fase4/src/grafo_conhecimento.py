"""
Construcao e analise do grafo de conhecimento com NetworkX.

Alinhado ao codigo do professor (linhas 703-712 do compilado_codigos.txt):
  df = pd.DataFrame(edges, columns=['source', 'target'])
  df['weight'] = 1.
  df.to_csv('edges.csv', index=False)
  Counter(chain(*edges)).most_common()
"""

from __future__ import annotations

import os
from collections import Counter
from itertools import chain
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

from fase4_config import (
    ENTIDADES_VALIDAS_GRAFO,
    METRICAS_CENTRALIDADE,
    MINIMO_FREQUENCIA_ENTIDADE,
    NUMERO_MINIMO_NOS_GRAFO,
    TOP_CENTRALIDADE,
)
from logger import inicializar_sistema_log

logger = inicializar_sistema_log(__name__)


class GrafoConhecimento:
    def __init__(self, nome: str = "Grafo de Conhecimento") -> None:
        self.nome = nome
        self.grafo: nx.Graph = nx.Graph()
        self.centralidade: Dict[str, pd.DataFrame] = {}
        self.comunidades: Dict[int, List[str]] = {}
        self._freq_min = MINIMO_FREQUENCIA_ENTIDADE

    # ------------------------------------------------------------------
    # Construcao do grafo
    # ------------------------------------------------------------------

    def construir_grafo_entidade_entidade(self, arestas: pd.DataFrame) -> nx.Graph:
        """
        Constroi grafo entidade-entidade a partir de DataFrame de arestas.
        Estrutura: columns=['source', 'target', 'relation', 'frequencia']
        Padrao do professor (linhas 703-706).
        """
        if arestas.empty:
            logger.warning("DataFrame de arestas vazio — grafo sem arestas")
            return self.grafo
        self.grafo = nx.from_pandas_edgelist(
            arestas,
            source="source",
            target="target",
            edge_attr=["frequencia"] if "frequencia" in arestas.columns else True,
        )
        logger.info(
            "Grafo entidade-entidade: %d nos, %d arestas",
            self.grafo.number_of_nodes(),
            self.grafo.number_of_edges(),
        )
        return self.grafo

    def construir_grafo_entidade_documento(
        self, entidades_df: pd.DataFrame
    ) -> nx.Graph:
        """
        Constroi grafo bipartido entidade-documento.
        """
        g = nx.Graph()
        if entidades_df.empty:
            return g
        for _, row in entidades_df.iterrows():
            entidade = str(row.get("texto", "")).strip()
            doc_id = row.get("documento_id", 0)
            if not entidade:
                continue
            doc_node = f"DOC_{doc_id}"
            g.add_node(entidade, bipartite=0, tipo=row.get("tipo", "MISC"))
            g.add_node(doc_node, bipartite=1, tipo="documento")
            if g.has_edge(entidade, doc_node):
                g[entidade][doc_node]["weight"] = g[entidade][doc_node].get("weight", 0) + 1
            else:
                g.add_edge(entidade, doc_node, weight=1)
        return g

    def adicionar_nos_entidades(self, entidades_df: pd.DataFrame) -> None:
        """
        Adiciona nos do grafo baseado em entidades do DataFrame,
        respeitando frequencia minima.
        """
        if entidades_df.empty:
            return

        # Padrao do professor (linhas 711-712): Counter(chain(*edges)).most_common()
        freq = Counter(entidades_df["texto"].dropna().tolist())
        freq_min = self._freq_min

        for entidade, contagem in freq.most_common():
            if contagem < freq_min:
                continue
            tipo = entidades_df.loc[
                entidades_df["texto"] == entidade, "tipo"
            ].values
            tipo_ent = tipo[0] if len(tipo) > 0 else "MISC"
            if tipo_ent not in ENTIDADES_VALIDAS_GRAFO:
                continue
            if not self.grafo.has_node(entidade):
                self.grafo.add_node(entidade, tipo=tipo_ent, frequencia=contagem)
            else:
                self.grafo.nodes[entidade]["frequencia"] = contagem
                self.grafo.nodes[entidade]["tipo"] = tipo_ent

        self._validar_numero_nos_e_aplicar_fallback(entidades_df)

    def _validar_numero_nos(self) -> bool:
        return self.grafo.number_of_nodes() >= NUMERO_MINIMO_NOS_GRAFO

    def _validar_numero_nos_e_aplicar_fallback(
        self, entidades_df: pd.DataFrame
    ) -> None:
        if self._validar_numero_nos():
            return
        logger.warning(
            "Grafo com apenas %d nos (minimo: %d) — aplicando fallback",
            self.grafo.number_of_nodes(),
            NUMERO_MINIMO_NOS_GRAFO,
        )
        self._aplicar_fallback_nos(entidades_df)

    def _aplicar_fallback_nos(self, entidades_df: pd.DataFrame) -> None:
        """
        Reduz progressivamente freq_min e amplia tipos aceitos ate atingir
        NUMERO_MINIMO_NOS_GRAFO.
        """
        tipos_extras = {"DATE", "EVENT", "PRODUCT", "LAW", "NORP"}
        tipos_aceitos = set(ENTIDADES_VALIDAS_GRAFO) | tipos_extras

        for freq_min in (2, 1):
            if self._validar_numero_nos():
                break
            freq = Counter(entidades_df["texto"].dropna().tolist())
            for entidade, contagem in freq.most_common():
                if contagem < freq_min:
                    continue
                tipo_vals = entidades_df.loc[
                    entidades_df["texto"] == entidade, "tipo"
                ].values
                tipo_ent = tipo_vals[0] if len(tipo_vals) > 0 else "MISC"
                if tipo_ent not in tipos_aceitos:
                    continue
                if not self.grafo.has_node(entidade):
                    self.grafo.add_node(entidade, tipo=tipo_ent, frequencia=contagem)
            logger.info(
                "Fallback freq_min=%d: %d nos", freq_min, self.grafo.number_of_nodes()
            )

        # Ultimo recurso: inclui nós de documento
        if not self._validar_numero_nos() and not entidades_df.empty:
            logger.warning(
                "Incluindo nos de documento no grafo para atingir minimo de %d nos",
                NUMERO_MINIMO_NOS_GRAFO,
            )
            docs_ids = entidades_df.get("documento_id", pd.Series(dtype=int))
            for doc_id in docs_ids.dropna().unique():
                node_name = f"DOC_{int(doc_id)}"
                if not self.grafo.has_node(node_name):
                    self.grafo.add_node(node_name, tipo="documento", frequencia=1)
                if self._validar_numero_nos():
                    break

        if not self._validar_numero_nos():
            logger.error(
                "AVISO: grafo com apenas %d nos apos todos os fallbacks",
                self.grafo.number_of_nodes(),
            )

    # ------------------------------------------------------------------
    # Centralidade
    # ------------------------------------------------------------------

    def calcular_centralidade(self) -> Dict[str, pd.DataFrame]:
        """
        Calcula betweenness, degree, eigenvector e closeness centrality.
        """
        if self.grafo.number_of_nodes() == 0:
            logger.warning("Grafo vazio — centralidade nao calculada")
            return {}

        # Usa componente maior se grafo for desconexo
        if not nx.is_connected(self.grafo):
            maior_comp = max(nx.connected_components(self.grafo), key=len)
            g_sub = self.grafo.subgraph(maior_comp).copy()
        else:
            g_sub = self.grafo

        resultados: Dict[str, Dict] = {}
        try:
            resultados["betweenness"] = nx.betweenness_centrality(g_sub, normalized=True)
        except Exception as e:
            logger.warning("Betweenness falhou: %s", e)
            resultados["betweenness"] = {}

        resultados["degree"] = dict(nx.degree_centrality(g_sub))

        try:
            resultados["eigenvector"] = nx.eigenvector_centrality(g_sub, max_iter=500)
        except Exception:
            resultados["eigenvector"] = {}

        try:
            resultados["closeness"] = dict(nx.closeness_centrality(g_sub))
        except Exception:
            resultados["closeness"] = {}

        dfs: Dict[str, pd.DataFrame] = {}
        for metrica, valores in resultados.items():
            registros = []
            for node, valor in valores.items():
                attrs = self.grafo.nodes.get(node, {})
                registros.append(
                    {
                        "entidade": node,
                        "tipo": attrs.get("tipo", "MISC"),
                        "frequencia": attrs.get("frequencia", 0),
                        "centralidade": round(valor, 6),
                    }
                )
            df_metrica = pd.DataFrame(registros)
            if not df_metrica.empty:
                df_metrica = df_metrica.sort_values(
                    "centralidade", ascending=False
                ).reset_index(drop=True)
                df_metrica["rank"] = df_metrica.index + 1
            dfs[metrica] = df_metrica

        self.centralidade = dfs
        return dfs

    def obter_nos_mais_centrais(
        self, top_n: int = TOP_CENTRALIDADE, metrica: str = "betweenness"
    ) -> pd.DataFrame:
        if metrica not in self.centralidade:
            self.calcular_centralidade()
        df = self.centralidade.get(metrica, pd.DataFrame())
        return df.head(top_n)

    def obter_ranking_consolidado(self, top_n: int = TOP_CENTRALIDADE) -> pd.DataFrame:
        """Ranking consolidado como media dos ranks normalizados."""
        if not self.centralidade:
            self.calcular_centralidade()
        frames = []
        for metrica, df in self.centralidade.items():
            if df.empty:
                continue
            sub = df[["entidade", "rank"]].copy()
            sub = sub.rename(columns={"rank": f"rank_{metrica}"})
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=["entidade", "rank_medio"])
        df_merged = frames[0]
        for sub in frames[1:]:
            df_merged = df_merged.merge(sub, on="entidade", how="outer")
        rank_cols = [c for c in df_merged.columns if c.startswith("rank_")]
        df_merged["rank_medio"] = df_merged[rank_cols].mean(axis=1)
        return df_merged.sort_values("rank_medio").head(top_n).reset_index(drop=True)

    def obter_arestas_mais_frequentes(self, top_n: int = TOP_CENTRALIDADE) -> pd.DataFrame:
        if self.grafo.number_of_edges() == 0:
            return pd.DataFrame(columns=["source", "target", "frequencia"])
        registros = []
        for u, v, data in self.grafo.edges(data=True):
            registros.append(
                {
                    "source": u,
                    "target": v,
                    "frequencia": data.get("frequencia", data.get("weight", 1)),
                }
            )
        df = pd.DataFrame(registros)
        return df.sort_values("frequencia", ascending=False).head(top_n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Comunidades
    # ------------------------------------------------------------------

    def detectar_comunidades(self) -> Dict[int, List[str]]:
        """Detecta comunidades usando Louvain (python-louvain)."""
        if self.grafo.number_of_nodes() == 0:
            return {}
        try:
            import community as community_louvain

            if not nx.is_connected(self.grafo):
                maior_comp = max(nx.connected_components(self.grafo), key=len)
                g_sub = self.grafo.subgraph(maior_comp).copy()
            else:
                g_sub = self.grafo

            particao = community_louvain.best_partition(g_sub)
            agrupado: Dict[int, List[str]] = {}
            for node, com_id in particao.items():
                agrupado.setdefault(com_id, []).append(node)
            self.comunidades = agrupado
            logger.info("Comunidades detectadas: %d", len(agrupado))
            return agrupado
        except ImportError:
            logger.warning(
                "python-louvain nao instalado — deteccao de comunidades desabilitada"
            )
            # Fallback: cada componente conexa e uma comunidade
            comps = list(nx.connected_components(self.grafo))
            self.comunidades = {i: list(c) for i, c in enumerate(comps)}
            return self.comunidades

    # ------------------------------------------------------------------
    # Pergunta analitica
    # ------------------------------------------------------------------

    def responder_pergunta_analitica(self) -> str:
        """
        Responde: 'Quais entidades sao mais centrais no corpus e por que?'
        """
        if not self.centralidade:
            self.calcular_centralidade()

        ranking = self.obter_ranking_consolidado(top_n=5)
        if ranking.empty:
            return "Nao foi possivel calcular entidades centrais (grafo insuficiente)."

        linhas = [
            "=== PERGUNTA ANALITICA ===",
            "Quais entidades sao mais centrais no corpus e por que?\n",
        ]
        for _, row in ranking.iterrows():
            entidade = row["entidade"]
            rank_medio = row["rank_medio"]
            # Vizinhos no grafo
            vizinhos = list(self.grafo.neighbors(entidade))[:5]
            grau = self.grafo.degree(entidade)
            attrs = self.grafo.nodes.get(entidade, {})
            tipo = attrs.get("tipo", "MISC")
            freq = attrs.get("frequencia", "N/A")
            linhas.append(f"Entidade: {entidade}")
            linhas.append(f"  Tipo: {tipo} | Frequencia: {freq} | Grau: {grau}")
            linhas.append(f"  Rank medio de centralidade: {rank_medio:.2f}")
            if vizinhos:
                linhas.append(f"  Conectada com: {', '.join(vizinhos)}")
            linhas.append("")

        return "\n".join(linhas)

    def gerar_resumo_interpretativo(
        self,
        centralidade: Dict[str, pd.DataFrame],
        arestas: pd.DataFrame,
    ) -> str:
        """
        Gera texto interpretativo em linguagem acessivel para audiencia nao tecnica.
        """
        n_nos = self.grafo.number_of_nodes()
        n_arestas = self.grafo.number_of_edges()
        n_comunidades = len(self.comunidades)
        densidade = (
            nx.density(self.grafo) if n_nos > 1 else 0.0
        )

        top_ents = self.obter_ranking_consolidado(top_n=3)
        top_nomes = top_ents["entidade"].tolist() if not top_ents.empty else []

        linhas = [
            "=== RESUMO INTERPRETATIVO (AUDIENCIA NAO TECNICA) ===\n",
            "O que foi feito:",
            "  Analisamos um conjunto de textos e identificamos as entidades mais importantes",
            "  (pessoas, organizacoes, locais) e como elas se relacionam entre si.\n",
            f"Resultados em numeros:",
            f"  - Numero de entidades unicas identificadas: {n_nos}",
            f"  - Numero de conexoes entre entidades: {n_arestas}",
            f"  - Grupos tematicos encontrados: {n_comunidades}",
            f"  - Densidade da rede: {densidade:.3f} (quanto maior, mais interconectada)\n",
        ]
        if top_nomes:
            linhas.append(
                f"As entidades mais importantes do corpus sao: {', '.join(top_nomes)}."
            )
            linhas.append(
                "  Essas entidades aparecem com frequencia e conectam varios outros topicos,"
            )
            linhas.append(
                "  funcionando como 'hubs' da narrativa — ou seja, pontos de referencia"
            )
            linhas.append("  em torno dos quais os textos se organizam.\n")

        if not arestas.empty:
            top_arestas = arestas.head(3)
            linhas.append("Relacoes mais frequentes encontradas:")
            for _, row in top_arestas.iterrows():
                linhas.append(
                    f"  - '{row['source']}' se relaciona com '{row['target']}'"
                    f" ({int(row['frequencia'])} vezes)"
                )
            linhas.append("")

        linhas.append(
            "Limitacoes: a qualidade da analise depende da disponibilidade de entidades"
        )
        linhas.append(
            "  reconhecidas pelo modelo de linguagem. Textos com poucos nomes proprios"
        )
        linhas.append(
            "  podem gerar um grafo menos rico do que o esperado."
        )
        return "\n".join(linhas)

    # ------------------------------------------------------------------
    # Exportacao
    # ------------------------------------------------------------------

    def exportar_grafo(self, caminho_arestas: str, caminho_nos: str) -> None:
        """
        Exporta arestas e nos do grafo para CSV.
        Padrao do professor (linha 706): df.to_csv('edges.csv', index=False)
        """
        # Arestas
        registros_arestas = []
        for u, v, data in self.grafo.edges(data=True):
            registros_arestas.append(
                {
                    "source": u,
                    "target": v,
                    "frequencia": data.get("frequencia", data.get("weight", 1)),
                }
            )
        df_arestas = pd.DataFrame(registros_arestas)
        df_arestas.to_csv(caminho_arestas, index=False, encoding="utf-8")
        logger.info("Arestas exportadas: %s (%d linhas)", caminho_arestas, len(df_arestas))

        # Nos com metricas de centralidade
        registros_nos = []
        metricas_all = {m: self.centralidade.get(m, pd.DataFrame()) for m in METRICAS_CENTRALIDADE}
        for node, attrs in self.grafo.nodes(data=True):
            registro: Dict[str, Any] = {
                "entidade": node,
                "tipo": attrs.get("tipo", "MISC"),
                "frequencia": attrs.get("frequencia", 0),
                "grau": self.grafo.degree(node),
            }
            for metrica, df_m in metricas_all.items():
                if not df_m.empty and "entidade" in df_m.columns:
                    val_series = df_m.loc[df_m["entidade"] == node, "centralidade"]
                    registro[f"centralidade_{metrica}"] = (
                        val_series.values[0] if len(val_series) > 0 else 0.0
                    )
                else:
                    registro[f"centralidade_{metrica}"] = 0.0
            registros_nos.append(registro)

        df_nos = pd.DataFrame(registros_nos)
        df_nos.to_csv(caminho_nos, index=False, encoding="utf-8")
        logger.info("Nos exportados: %s (%d linhas)", caminho_nos, len(df_nos))

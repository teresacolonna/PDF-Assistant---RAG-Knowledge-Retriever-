"""graph.py
----------
Costruzione di un grafo di contesto a partire dai chunk recuperati
e utility per esplorare i vicini di un'entità.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
from langchain_core.documents import Document


def _extract_entities(text: str) -> list[str]:
    """Estrae parole chiave/entità semplici da un testo (token >= 4 caratteri, no stopword)."""
    stopwords = {
        "della", "delle", "degli", "dello", "nella", "nelle", "negli",
        "nello", "sulla", "sulle", "sugli", "dello", "questo", "questa",
        "questi", "queste", "anche", "come", "sono", "essere", "viene",
        "anno", "anni", "that", "with", "from", "this", "have", "been",
    }
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]{4,}", text.lower())
    return [t for t in tokens if t not in stopwords]


def build_graph(docs: list[Document]) -> nx.Graph:
    """Costruisce un grafo dove:
    - ogni nodo è un'entità (parola chiave) estratta dai chunk
    - un arco connette due entità che co-occorrono nello stesso chunk

    Ogni nodo memorizza il testo del primo chunk in cui appare.
    """
    G = nx.Graph()

    for doc in docs:
        text = doc.page_content
        entities = list(set(_extract_entities(text)))

        for entity in entities:
            if entity not in G:
                G.add_node(entity, text=text)

        # archi di co-occorrenza
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                e1, e2 = entities[i], entities[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    return G


def neighbors_of_entity(
    graph: nx.Graph,
    entity: str,
    top_k: int = 2,
) -> list[str]:
    """Restituisce i top_k vicini dell'entità nel grafo, ordinati per peso dell'arco.

    Se l'entità non è presente nel grafo, effettua una ricerca fuzzy cercando
    nodi che contengono la stringa dell'entità come sottostringa.
    """
    entity_lower = entity.lower()

    # ricerca esatta
    if entity_lower in graph:
        neighbors = sorted(
            graph.neighbors(entity_lower),
            key=lambda n: graph[entity_lower][n].get("weight", 1),
            reverse=True,
        )
        return neighbors[:top_k]

    # ricerca fuzzy (sottostringa)
    candidates = [n for n in graph.nodes if entity_lower in n or n in entity_lower]
    if not candidates:
        return []

    # prendi il candidato con più connessioni
    best = max(candidates, key=lambda n: graph.degree(n))
    neighbors = sorted(
        graph.neighbors(best),
        key=lambda n: graph[best][n].get("weight", 1),
        reverse=True,
    )
    return neighbors[:top_k]
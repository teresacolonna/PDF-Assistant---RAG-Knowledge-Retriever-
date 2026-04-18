"""rag_graph.py
---------------
Pipeline RAG iterativa implementata con LangGraph.

Nodi:
  - retrieval   : recupera i chunk dal vector store
  - generation  : genera una risposta con l'LLM
  - reflection  : valuta la risposta e decide se iterare o terminare

Flusso:
  __start__ → retrieval → generation → reflection → __end__
                                ↑______________|  (se serve un altro giro)
"""

from __future__ import annotations

import os
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langchain_openai import OpenAI
from langgraph.graph import StateGraph, END

from .vector_store import get_vector_store
from .retriever import retrieve_context
from .graph import build_graph, neighbors_of_entity

load_dotenv()

# ---------------------------------------------------------------------------
# Stato condiviso tra i nodi
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str               # domanda originale dell'utente
    current_question: str       # domanda corrente (può cambiare dopo reflection)
    context: str                # contesto recuperato dal vector store
    answer: str                 # risposta generata
    last_answer: str            # risposta del giro precedente (per rilevare convergenza)
    loop_count: int             # contatore iterazioni
    max_loops: int              # limite massimo di iterazioni
    k: int                      # numero di chunk da recuperare
    year_filter: str | None     # filtro anno opzionale


# ---------------------------------------------------------------------------
# Nodi
# ---------------------------------------------------------------------------

def retrieval_node(state: RAGState) -> dict:
    """Recupera i chunk dal vector store e costruisce il contesto."""
    vs = get_vector_store()

    retrieved, context_str = retrieve_context(
        vectorstore=vs,
        question=state["current_question"],
        k=state["k"],
        year_filter=state["year_filter"],
    )

    # costruiamo anche il grafo di co-occorrenza (per arricchire il contesto)
    if retrieved:
        docs_only = [doc for doc, _ in retrieved]
        nx_graph = build_graph(docs_only)

        # tentiamo di estendere il contesto con i vicini del concetto chiave
        followup = state.get("answer", "")
        if followup:
            neighbors = neighbors_of_entity(nx_graph, followup.split()[0], top_k=2)
            if neighbors:
                extra = "\n".join(nx_graph.nodes[n]["text"] for n in neighbors)
                context_str += "\n---\n" + extra

    return {"context": context_str}


def generation_node(state: RAGState) -> dict:
    """Genera una risposta usando il contesto recuperato."""
    llm = OpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    context = state["context"]

    if not context.strip():
        return {
            "answer": (
                "Mi dispiace, non ho trovato informazioni rilevanti nei report ISTAT "
                "per rispondere a questa domanda."
            )
        }

    if "Dummy content" in context:
        return {
            "answer": (
                "I documenti attualmente indicizzati sono ancora i file di prova. "
                "Sostituisci i PDF in data/raw con i report reali, "
                "ricostruisci il vector store e riprova."
            )
        }

    prompt = (
        "Sei un assistente esperto sui report ISTAT. "
        "Rispondi alla domanda utilizzando esclusivamente il contesto fornito, "
        "e cita sempre la fonte (ad es. 'Secondo il report 2021-2023, pag. 45').\n\n"
        f"CONTESTO:\n{context}\n\n"
        f"DOMANDA: {state['current_question']}\n"
        "RISPOSTA:"
    )

    result = llm.generate([prompt])
    answer = result.generations[0][0].text

    return {"answer": answer}


def reflection_node(state: RAGState) -> dict:
    """Valuta la risposta e decide se approfondire un concetto."""
    llm = OpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    answer = state["answer"]
    last_answer = state.get("last_answer", "")
    loop_count = state.get("loop_count", 0)

    # aggiorniamo il contatore e salviamo la risposta corrente come "ultima"
    updates: dict = {
        "last_answer": answer,
        "loop_count": loop_count + 1,
    }

    # chiediamo all'LLM quale concetto approfondire
    followup_prompt = (
        "Dalla risposta precedente, indica in una sola parola il concetto o entità "
        "più importante da approfondire. Rispondi 'NONE' se non ci sono altri spunti.\n\n"
        f"RISPOSTA PRECEDENTE:\n{answer}\n\n"
    )
    followup = llm.generate([followup_prompt]).generations[0][0].text.strip()

    if followup.upper() != "NONE":
        updates["current_question"] = (
            f"Inquadra meglio il concetto '{followup}' "
            f"in relazione a: {state['question']}"
        )

    return updates



# Routing: dopo reflection decidiamo se tornare a generation o terminare


def should_continue(state: RAGState) -> str:
    """Ritorna 'generation' per iterare, 'end' per terminare."""
    if state["loop_count"] >= state["max_loops"]:
        return "end"
    if state["answer"] == state.get("last_answer", ""):
        return "end"
    # se la domanda corrente non è cambiata non ha senso iterare
    if state["current_question"] == state["question"] and state["loop_count"] > 0:
        return "end"
    return "generation"



# Costruzione del grafo LangGraph


def build_rag_graph() -> StateGraph:
    """Costruisce e compila il grafo LangGraph."""
    builder = StateGraph(RAGState)

    builder.add_node("retrieval", retrieval_node)
    builder.add_node("generation", generation_node)
    builder.add_node("reflection", reflection_node)

    builder.set_entry_point("retrieval")
    builder.add_edge("retrieval", "generation")
    builder.add_edge("generation", "reflection")

    builder.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "generation": "generation",
            "end": END,
        },
    )

    return builder.compile()


# Funzione principale da chiamare da rag_pipeline.py


def run_rag_graph(
    question: str,
    k: int = 3,
    year_filter: str | None = None,
    max_loops: int = 2,
    save_png: bool = False,
) -> str:
    """Esegue la pipeline RAG con LangGraph e restituisce la risposta finale."""
    graph = build_rag_graph()

    # salva il grafo come PNG
    if save_png:
        with open("graph_rag.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print("Grafo salvato in graph_rag.png")

    initial_state: RAGState = {
        "question": question,
        "current_question": question,
        "context": "",
        "answer": "",
        "last_answer": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "k": k,
        "year_filter": year_filter,
    }

    final_state = graph.invoke(initial_state)
    return final_state["answer"]

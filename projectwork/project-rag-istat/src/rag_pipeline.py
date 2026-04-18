"""rag_pipeline.py
------------------
Orchestrazione del flusso RAG:
1. recupera/crea il vector store
2. esegue il retrieval
3. costruisce il prompt
4. invoca l'LLM per la generazione
"""

from __future__ import annotations

from dotenv import load_dotenv
import os

from langchain_openai import OpenAI

from .vector_store import get_vector_store
from .retriever import retrieve_context
from .graph import build_graph, neighbors_of_entity

load_dotenv()


def answer_question(
    question: str,
    k: int = 3,
    year_filter: str | None = None,
    rebuild_store: bool = False,
) -> str:
    """Esegue una singola richiesta RAG e restituisce il testo generato.

    La risposta viene formulata affinché l'LLM citi esplicitamente la fonte
    (anno/pagina) quando utilizza informazioni dai chunk.
    """

    vs = get_vector_store(rebuild=rebuild_store)


    score_thr = os.getenv("MIN_SCORE", None)
    if score_thr is not None:
        try:
            score_thr = float(score_thr)
        except ValueError:
            score_thr = None

    retrieved, context_str = retrieve_context(
        vectorstore=vs,
        question=question,
        k=k,
        year_filter=year_filter,
    )

    # fallback rapido se non ho trovato niente di utile (lista vuota o contesto vuoto)
    if not retrieved or not context_str.strip():
        return (
            "Mi dispiace, non ho trovato informazioni rilevanti nei report ISTAT "
            "per rispondere a questa domanda."
        )

    # se i chunk contengono ancora i testi dummy, segnaliamo che bisogna sostituire i PDF
    if "Dummy content" in context_str:
        return (
            "I documenti attualmente indicizzati sono ancora i file di prova. "
            "Sostituisci i PDF in data/raw con i report reali, ricostruisci il vector store "
            "e riprova."
        )

    # costruiamo un prompt semplice ma con istruzioni di citazione
    prompt = (
        "Sei un assistente esperto sui report ISTAT. "
        "Rispondi alla domanda utilizzando esclusivamente il contesto fornito, "
        "e cita sempre la fonte (ad es. 'Secondo il report 2021-2023, pag. 45').\n\n"
        f"CONTESTO:\n{context_str}\n\n"
        f"DOMANDA: {question}\n"
        "RISPOSTA:"   # il modello completa qui
    )

    llm = OpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))
    result = llm.generate([prompt])
    return result.generations[0][0].text


def answer_question_with_graph(
    question: str,
    k: int = 3,
    year_filter: str | None = None,
    rebuild_store: bool = False,
    max_loops: int = 2,
) -> str:
    """Esegui una pipeline RAG arricchita da un grafo di contesto e un ciclo iterativo.

    In ogni iterazione viene costruito un grafo dai chunk recuperati e il modello viene
    interrogato; un"follow‑up" automatico chiede al modello quale concetto approfondire
    e, se rilevante, il contesto viene esteso con i vicini nel grafo. Il ciclo termina
    quando la risposta non cambia o non ci sono ulteriori concetti da esplorare.
    """

    vs = get_vector_store(rebuild=rebuild_store)
    llm = OpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

    last_answer = None
    current_question = question

    for loop in range(max_loops):
        retrieved, context_str = retrieve_context(
            vectorstore=vs,
            question=current_question,
            k=k,
            year_filter=year_filter,
        )

        if not retrieved or not context_str.strip():
            return (
                "Mi dispiace, non ho trovato informazioni rilevanti nei report ISTAT "
                "per rispondere a questa domanda."
            )

        if "Dummy content" in context_str:
            return (
                "I documenti attualmente indicizzati sono ancora i file di prova. "
                "Sostituisci i PDF in data/raw con i report reali, ricostruisci il vector store "
                "e riprova."
            )

        # costruzione del grafo dai chunk (estraiamo solo i documenti dalle tuple)
        docs_only = [doc for doc, _ in retrieved]
        graph = build_graph(docs_only)

        prompt = (
            "Sei un assistente esperto sui report ISTAT. "
            "Rispondi alla domanda utilizzando esclusivamente il contesto fornito, "
            "e cita sempre la fonte (ad es. 'Secondo il report 2021-2023, pag. 45').\n\n"
            f"CONTESTO:\n{context_str}\n\n"
            f"DOMANDA: {current_question}\n"
            "RISPOSTA:"
        )
        result = llm.generate([prompt])
        answer = result.generations[0][0].text

        if answer == last_answer:
            break
        last_answer = answer

        # chiediamo un concetto da approfondire
        followup_prompt = (
            "Dalla risposta precedente, indica in una sola parola il concetto o entità "
            "più importante da approfondire. Rispondi 'NONE' se non ci sono altri spunti.\n\n"
            f"RISPOSTA PRECEDENTE:\n{answer}\n\n"
        )
        followup = llm.generate([followup_prompt]).generations[0][0].text.strip()
        if followup.upper() == "NONE":
            break

        neighbors = neighbors_of_entity(graph, followup, top_k=2)
        if neighbors:
            extra_texts = "\n".join(graph.nodes[n]["text"] for n in neighbors)
            context_str += "\n" + extra_texts
            current_question = (
                f"Inquadra meglio il concetto '{followup}' in relazione a: {question}"
            )
        else:
            break

    return last_answer or ""

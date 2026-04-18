"""retriever.py
-----------
Contiene le funzioni di retrieval usate dall'esercizio.
"""

from __future__ import annotations

from typing import Any, Tuple, List


def retrieve_context(
    vectorstore: Any,
    question: str,
    k: int = 3,
    year_filter: str | None = None,
) -> Tuple[List[tuple[Any, float]], str]:
    """Recupera i top-k chunk più rilevanti e costruisce la stringa di contesto.

    Parameters
    ----------
    vectorstore:
        Un oggetto che implementa ``similarity_search_with_score`` (es. Chroma).
    question:
        La domanda/termine di ricerca.
    k:
        Numero di chunk da prelevare.
    year_filter:
        Filtro opzionale sui metadati ``anni`` (es. "2021-2023").

    Returns
    -------
    retrieved_docs:
        Lista di tuple (Document, score).
    context_str:
        Tutti i testi concatenati, pronti per essere iniettati nel prompt.
    """

    if not question.strip():
        raise ValueError("La domanda non può essere vuota")
    if k <= 0:
        raise ValueError("k deve essere >0")

    search_kwargs: dict[str, Any] = {"k": k}
    if year_filter:

        search_kwargs["filter"] = {"anni": year_filter}

    retrieved: list[tuple[Any, float]] = vectorstore.similarity_search_with_score(
        question,
        **search_kwargs,
    )

    chunks: list[str] = []
    for doc, score in retrieved:
        fonte = doc.metadata.get("fonte", "?")
        pagina = doc.metadata.get("pagina", "?")
        chunks.append(f"{fonte} (pag. {pagina}): {doc.page_content}")
    context_str = "\n---\n".join(chunks)
    return retrieved, context_str

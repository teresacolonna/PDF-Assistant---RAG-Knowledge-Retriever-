"""Simple FastAPI wrapper around the RAG pipeline."""

from fastapi import FastAPI, Query
from src.rag_pipeline import answer_question, answer_question_with_graph

app = FastAPI(title="RAG ISTAT API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/query")
def query(
    q: str = Query(..., description="The question to ask"),
    k: int = Query(3, ge=1, description="Number of chunks to retrieve"),
    year: str | None = Query(None, description="Optional year filter (e.g. '2021-2023')"),
    graph: bool = Query(False, description="Use graph+loop reasoning"),
):
    """Perform a RAG query and return the model's answer."""
    if graph:
        answer = answer_question_with_graph(question=q, k=k, year_filter=year)
    else:
        answer = answer_question(question=q, k=k, year_filter=year)
    return {"question": q, "answer": answer}

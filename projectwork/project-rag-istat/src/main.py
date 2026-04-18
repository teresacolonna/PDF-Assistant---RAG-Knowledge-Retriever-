"""CLI di utilità per interrogare il sistema RAG.
"""

import argparse
from src.rag_pipeline import answer_question, answer_question_with_graph
from src.vector_store import get_vector_store


def main():
    parser = argparse.ArgumentParser(description="Semplice shell RAG per i report ISTAT")
    parser.add_argument("--query", "-q", required=True, help="Domanda da porre")
    parser.add_argument("--k", type=int, default=3, help="Numero di chunk da recuperare")
    parser.add_argument("--year", "-y", help="Filtro per anni (es. '2020-2022')")
    parser.add_argument("--rebuild", action="store_true", help="Rigenera il vector store")
    parser.add_argument("--graph", action="store_true", help="Usa il ciclo iterativo con grafo per il ragionamento")
    args = parser.parse_args()

    if args.rebuild:
        # ricrea esplicitamente il vector store
        get_vector_store(rebuild=True)

    if args.graph:
        answer = answer_question_with_graph(
            question=args.query,
            k=args.k,
            year_filter=args.year,
        )
    else:
        answer = answer_question(
            question=args.query,
            k=args.k,
            year_filter=args.year,
        )

    print("\n=== RISPOSTA ===\n")
    print(answer)


if __name__ == "__main__":
    main()

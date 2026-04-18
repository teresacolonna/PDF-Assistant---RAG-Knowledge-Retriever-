"""
embeddings.py
-------------
Step 2 del progetto RAG ISTAT.
Prende i chunk dall'ingestion, genera gli embeddings con OpenAI
e li salva su ChromaDB (database vettoriale locale).
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .ingestion import run_ingestion

load_dotenv()

CHROMA_DB_PATH = os.path.join("data", "chroma_db")  
EMBEDDING_MODEL = "text-embedding-3-small"          


def build_vector_store():
    """
    Pipeline completa:
    1. Estrae i chunk dai PDF (ingestion)
    2. Genera gli embeddings con OpenAI
    3. Salva tutto su ChromaDB
    """

    print("Avvio ingestion dei PDF...")
    chunks = run_ingestion()
    print(f"Chunk pronti: {len(chunks)}")

    print("\n Preparazione dati per ChromaDB...")
    testi = [chunk["text"] for chunk in chunks]
    metadati = [chunk["metadata"] for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"\n Inizializzazione embedding model: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

 
    print(f"\n Creazione ChromaDB in: {CHROMA_DB_PATH}")
    print("   (Questa operazione chiama le API OpenAI, attendere...)")

    vector_store = Chroma.from_texts(
        texts=testi,
        embedding=embeddings,
        metadatas=metadati,
        ids=ids,
        persist_directory=CHROMA_DB_PATH
    )

    print(f"\n ChromaDB creato e salvato!")
    print(f"   Documenti indicizzati: {vector_store._collection.count()}")
    return vector_store


def test_retrieval(vector_store, query: str, k: int = 3, year_filter: str | None = None):
    """
    Test rapido: fa una query sul vector store e mostra i risultati.

    Parameters
    ----------
    vector_store:
        Istanza di Chroma (o altro) già inizializzato.
    query:
        Termine di ricerca.
    k:
        Numero di risultati da restituire.
    year_filter:
        Se specificato, applica un filtro sui metadati "anni".
    """
    print(f"\n{'='*60}")
    print(f"🔍 TEST RETRIEVAL")
    print(f"   Query: '{query}'")
    if year_filter:
        print(f"   Filtro anni: {year_filter}")
    print(f"   Top-{k} risultati:")
    print(f"{'='*60}")

    if year_filter:
        results = vector_store.similarity_search(query, k=k, filter={"anni": year_filter})
    else:
        results = vector_store.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\n--- Risultato {i+1} ---")
        print(f"Fonte: {doc.metadata.get('fonte')} | Pagina: {doc.metadata.get('pagina')}")
        print(f"Testo: {doc.page_content[:300]}...")


if __name__ == "__main__":

    vs = build_vector_store()

    test_retrieval(vs, "emissioni di CO2 e gas climalteranti")
    test_retrieval(vs, "spesa per la protezione dell'ambiente")
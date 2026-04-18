"""vector_store.py
---------------
Help utilities per caricare / costruire il vector store Chroma.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from .embeddings import build_vector_store, CHROMA_DB_PATH, EMBEDDING_MODEL

load_dotenv()


def get_vector_store(rebuild: bool = False) -> Chroma:
    """Restituisce un Chroma vector store.

    Se esiste già una directory persistente e ``rebuild`` è False,
    carica il DB esistente; altrimenti chiama ``build_vector_store()``
    che rigenera gli embeddings da zero.

    Parameters
    ----------
    rebuild:
        Forza la ricostruzione completa (ignorando eventuale store salvato).
    """

    if rebuild or not os.path.isdir(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        return build_vector_store()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

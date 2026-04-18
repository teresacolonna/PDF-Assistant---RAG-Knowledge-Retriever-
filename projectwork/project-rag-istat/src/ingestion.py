"""
ingestion.py
------------
Step 1 del progetto RAG ISTAT.
Estrae il testo dai 3 PDF, lo divide in chunk e aggiunge metadati.
"""

import os


try:
    import fitz  
except ImportError as e:
    raise ImportError(
        "La libreria 'fitz' (PyMuPDF) non è installata. "
        "Esegui `pip install pymupdf` o controlla il tuo environment.`"
    ) from e

from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FILES = {
    "2020-2022": os.path.join("data", "raw", "report_2020_2022.pdf"),
    "2021-2023": os.path.join("data", "raw", "report_2021_2023.pdf"),
    "2022-2024": os.path.join("data", "raw", "report_2022_2024.pdf"),
}

CHUNK_SIZE = 700       
CHUNK_OVERLAP = 100  


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Estrae il testo da un PDF pagina per pagina.
    Restituisce una lista di dict con testo e numero di pagina.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        text = text.strip()
        if text:  
            pages.append({
                "text": text,
                "page": page_num + 1  
            })
    doc.close()
    return pages


def split_into_chunks(pages: list[dict], anni: str) -> list[dict]:
    """
    Divide il testo estratto in chunk più piccoli.
    Aggiunge metadati: anni di riferimento, numero di pagina, fonte.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for page_data in pages:
        testi_divisi = splitter.split_text(page_data["text"])
        for i, testo in enumerate(testi_divisi):
            chunks.append({
                "text": testo,
                "metadata": {
                    "anni": anni,                         
                    "pagina": page_data["page"],          
                    "fonte": f"ISTAT Report {anni}",      
                    "chunk_index": i,                     
                }
            })
    return chunks


def run_ingestion() -> list[dict]:
    """
    Esegue l'ingestion completa di tutti i PDF.
    Restituisce tutti i chunk con metadati pronti per l'embedding.
    """
    all_chunks = []

    for anni, pdf_path in PDF_FILES.items():
        print(f"\nElaborazione: Report {anni}")
        print(f"   File: {pdf_path}")

        if not os.path.exists(pdf_path):
            print(f"File non trovato, skip.")
            continue

        pages = extract_text_from_pdf(pdf_path)
        print(f"Pagine estratte: {len(pages)}")

        chunks = split_into_chunks(pages, anni)
        print(f"Chunk generati: {len(chunks)}")

        all_chunks.extend(chunks)

    print(f"\n Totale chunk da tutti i report: {len(all_chunks)}")
    return all_chunks

def preview_chunks(chunks: list[dict], n: int = 3):
    """Stampa n chunk di esempio per verificare il risultato."""
    print(f"\n{'='*60}")
    print(f"ANTEPRIMA DEI PRIMI {n} CHUNK")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Metadati: {chunk['metadata']}")
        print(f"Testo ({len(chunk['text'])} caratteri):")
        print(chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'])


if __name__ == "__main__":
    chunks = run_ingestion()
    preview_chunks(chunks, n=3)
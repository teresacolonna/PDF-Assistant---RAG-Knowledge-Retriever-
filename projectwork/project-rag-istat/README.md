# RAG ISTAT Project

Questo repository contiene un semplice pipeline RAG per interrogare i report ISTAT.

## Stato attuale

- **Ingestion**: `src/ingestion.py` estrae testo da PDF e genera chunk con metadati.
- **Embeddings**: `src/embeddings.py` crea un vector store Chroma usando OpenAI.
- **Struttura cartelle**: `data/raw` contiene i PDF, `data/chroma_db` il database vettoriale.
- **Requisiti**: `requirements.txt` include `langchain`, `chromadb`, `pymupdf`, `dotenv`.

I moduli di retrieval, pipeline e evaluazione sono stati aggiunti in `src/`.

## Come usare

1. creare un file `.env` con la variabile `OPENAI_API_KEY`.
2. installare le dipendenze:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. (opzionale) generare PDF dummy per test:

   ```powershell
   python -m src.create_dummy_pdfs
   ```

4. costruire il vector store e fare un esempio di query:

   ```powershell
   python -m src.main -q "emissioni di CO2" --rebuild
   ```
   Se vuoi sfruttare il nuovo meccanismo iterativo basato su un grafo di
   contesto, aggiungi la flag `--graph`:

   ```powershell
   python -m src.main -q "emissioni di CO2" --graph
   ```
   La prima esecuzione recuperera i chunk rilevanti, costruire un semplice grafo
   sulle parole condivise e, dopo ogni risposta, chiedere al modello quale concetto
   approfondire; il processo si ripete finche la risposta non converge.
5. eseguire i test automatici:

   ```powershell
   python -m src.evaluation
   ```

## Note

- Le query possono essere filtrate per anno fornendo `-y "2021-2023"`.
- La risposta generata dovrebbe citare le fonti (anno/pagina) dove possibile.
- L'implementazione è minimale; per metriche più sofisticate si può integrare RAGAS (vedi `src/evaluation.py`).

## Dockerizzazione 

Per ottenere un ambiente riproducibile e facilità di deployment, è disponibile un `Dockerfile` e
un `docker-compose.yml`.

1. Costruisci l'immagine:
   ```powershell
   docker build -t rag-istat .
   ```
2. Avvia il servizio API (monta i dati e l'.env):
   ```powershell
   docker-compose up --build
   ```
   il server sarà raggiungibile su `http://localhost:8000`.
3. Esempio di query HTTP:
   ```bash
   curl "http://localhost:8000/query?q=emissioni%20CO2&k=3"
   ```
4. Se preferisci l'interfaccia a linea di comando dentro il container:
   ```bash
   docker run --env-file .env -v "$PWD/data:/app/data" rag-istat \
       python -m src.main -q "domanda qui"
   ```

## Sostituzione dei PDF e rigenerazione

1. Copia i report ISTAT reali nella cartella `data/raw` (sovrascrivi o rimuovi i dummy).
   - I nomi correnti sono `report_2020_2022.pdf`, `report_2021_2023.pdf`, `report_2022_2024.pdf`.
   - In alternativa adatta il dizionario `PDF_FILES` in `src/ingestion.py`.
2. Ricostruisci il vector store per indicizzare i nuovi documenti:
   ```powershell
   python -m src.main -q "qualcosa" --rebuild
   # oppure con Docker:
   docker run --env-file .env -v "$PWD/data:/app/data" rag-istat \
       python -m src.main -q "qualcosa" --rebuild
   ```
   Se il sistema risponde suggerendo di sostituire i PDF, significa che il vecchio
   contenuto dummy è ancora presente.
3. Lancia nuovamente la valutazione per verificare le risposte aggiornate:
   ```powershell
   python -m src.evaluation
   ```

> È possibile regolare il parametro `MIN_SCORE` (punteggio massimo ammissibile
> per considerare un documento rilevante) impostandolo nel file `.env`.
> Valori più bassi stringono il filtro; un documento distante verrà ignorato e
> il bot risponderà con il messaggio di fallback.

## Grafi e ciclo di ragionamento

Un nuovo modulo `src/graph.py` costruisce un grafo elementare in cui i
chunk sono nodi collegati se condividono un certo numero di parole. La
funzione `answer_question_with_graph` in `src/rag_pipeline.py` sfrutta questo
strumento per effettuare un semplice loop di reasoning (modello -> followup ->
recupero aggiuntivo).

L'opzione 

```bash
python -m src.main -q "domanda" --graph
```

attiva la variante iterativa; la logica può essere replicata anche in
un eventuale servizio HTTP o bot, basta richiamare direttamente
`answer_question_with_graph`.

## Interfaccia web e Telegram

Oltre all'API FastAPI (endpoint `/query`), è possibile interrogare il sistema anche via
Telegram. Imposta `TELEGRAM_TOKEN` nel `.env` e avvia il bot:

```powershell
python -m src.telegram_bot
```

Poi invia messaggi al bot (es. "emissioni CO2"), riceverai le risposte generate.

## Valutazione avanzata (facoltativa)

`src/evaluation.py` contiene ancora la funzione `compute_ragas_metrics` ma puoi ignorarla
se non vuoi installare `ragas`. In caso contrario, l'installazione richiede alcuni pacchetti
nativi e può fallire; il comando di test gestisce il fallback.

Esegui lo script dopo la costruzione del DB:

```powershell
python -m src.evaluation
```

## Interfaccia web

Un piccolo server FastAPI (`src/app.py`) espone un endpoint `/query` per interrogare il
modello via HTTP. È eseguito automaticamente dal container Docker ma può essere lanciato
anche localmente con:

```powershell
uvicorn src.app:app --reload
```

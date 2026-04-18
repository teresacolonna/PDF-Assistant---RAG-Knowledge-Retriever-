"""Simple Telegram bot that forwards questions to the RAG pipeline."""
import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Imposta TELEGRAM_TOKEN nel tuo .env prima di avviare il bot")

from src.rag_pipeline import answer_question, answer_question_with_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 4000


async def send_long_message(update: Update, text: str):
    """Invia un messaggio lungo spezzandolo in chunk da MAX_CHUNK_SIZE caratteri."""
    if len(text) <= MAX_CHUNK_SIZE:
        await update.message.reply_text(text)
        return

    # Spezza il testo in chunk rispettando gli spazi (no tagli a metà parola)
    chunks = []
    while len(text) > MAX_CHUNK_SIZE:
        # Cerca l'ultimo spazio o newline entro il limite
        split_pos = text.rfind("\n", 0, MAX_CHUNK_SIZE)
        if split_pos == -1:
            split_pos = text.rfind(" ", 0, MAX_CHUNK_SIZE)
        if split_pos == -1:
            split_pos = MAX_CHUNK_SIZE  # fallback: taglia secco

        chunks.append(text[:split_pos].strip())
        text = text[split_pos:].strip()

    if text:
        chunks.append(text)

    for i, chunk in enumerate(chunks, 1):
        # Aggiunge indicatore di parte solo se ci sono più chunk
        if len(chunks) > 1:
            header = f"[{i}/{len(chunks)}]\n"
            await update.message.reply_text(header + chunk)
        else:
            await update.message.reply_text(chunk)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ciao! Inviami una domanda e ti risponderò usando il sistema RAG ISTAT."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    raw = update.message.text.strip()
    if not raw:
        return

    # support prefisso "graph:" per attivare il ciclo iterativo
    graph_mode = False
    question = raw
    if raw.lower().startswith("graph:"):
        graph_mode = True
        question = raw.split("\n", 1)[0].partition(":")[2].strip()

    await update.message.reply_text("Sto pensando...")

    try:
        if graph_mode:
            answer = answer_question_with_graph(question=question)
        else:
            answer = answer_question(question=question)

        await send_long_message(update, answer)

    except Exception as e:
        logger.exception("Errore nel pipeline")
        await update.message.reply_text("Si è verificato un errore, riprova più tardi.")


if __name__ == "__main__":
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Imposta TELEGRAM_TOKEN nel tuo .env prima di avviare il bot")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Avvio bot Telegram...")
    app.run_polling()
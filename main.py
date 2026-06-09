import os
import subprocess
import time
import socket
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama

app = FastAPI(title="Aarav Local RAG Engine")

# DB Configuration Parameters for Jio Allocation
DB_PARAMS = {
    "dbname": "aarav_vector_db",
    "user": "postgres",
    "password": "",  # Set your password if configured
    "host": "localhost",
    "port": "5432"
}

LLM_MODEL = "llama3.2"
EMBED_MODEL = "nomic-embed-text"

def bootstrap_ollama():
    """Checks if the local Ollama port is open. If closed, automates engine boot."""
    # Check if port 11434 is bound/listening
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        is_running = s.connect_ex(('127.0.0.1', 11434)) == 0

    if not is_running:
        print("[*] Ollama daemon is offline. Initializing automatic background boot thread...")
        try:
            # Spawn the systemd background task natives cleanly
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Allow time slice for the socket allocation
        except Exception as e:
            print(f"[-] Auto-boot failed. Attempting alternative systemd invoke: {e}")
            subprocess.Popen(["sudo", "systemctl", "start", "ollama"])
            time.sleep(3)

    # Automatically check and pre-fetch the necessary model layers if missing
    try:
        local_models = [m['model'] for m in ollama.list().get('models', [])]
        if f"{EMBED_MODEL}:latest" not in local_models and EMBED_MODEL not in local_models:
            print(f"[*] Fetching required embedding matrix: {EMBED_MODEL}")
            ollama.pull(EMBED_MODEL)
        if f"{LLM_MODEL}:latest" not in local_models and LLM_MODEL not in local_models:
            print(f"[*] Fetching required synthesis matrix: {LLM_MODEL}")
            ollama.pull(LLM_MODEL)
        print("[+] Ollama execution layer online and verified.")
    except Exception as e:
        print(f"[-] Target verification error: {e}. Confirm manually via 'ollama list'")

# Run the host automation routine
bootstrap_ollama()

def init_db():
    """Initializes extension registers and core vector tracking tables."""
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_store (
            id serial PRIMARY KEY,
            filename text,
            content text,
            embedding vector(768)
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

init_db()

class IngestPayload(BaseModel):
    filename: str
    text_content: str

class QueryPayload(BaseModel):
    prompt: str

def chunk_text(text: str, chunk_size=500, chunk_overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - chunk_overlap
    return chunks

@app.post("/ingest")
async def ingest_document(payload: IngestPayload):
    try:
        chunks = chunk_text(payload.text_content)

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)  # Registers explicit vector type casting parameters

        for chunk in chunks:
            response = ollama.embed(model=EMBED_MODEL, input=chunk)
            vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

            cursor.execute(
                "INSERT INTO document_store (filename, content, embedding) VALUES (%s, %s, %s);",
                (payload.filename, chunk, vector)
            )

        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "chunks_ingested": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pipeline(payload: QueryPayload):
    try:
        response = ollama.embed(model=EMBED_MODEL, input=payload.prompt)
        query_vector = response['embeddings'][0] if 'embeddings' in response else response['embedding']

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()
        register_vector(conn)  # CRITICAL: Forces mapping translation parameters

        # Pull top 3 slices closest to the calculated coordinate map
        cursor.execute(
            "SELECT content FROM document_store ORDER BY embedding <=> %s::vector LIMIT 3;",
            (query_vector,)
        )
        records = cursor.fetchall()
        cursor.close()
        conn.close()

        if not records:
            return {"answer": "No records found in database layers. Upload files first."}

        context_payload = "\n---\n".join([row[0] for row in records])

        system_instructions = (
            "You are an advanced technical intelligence assistant. Synthesize an optimal answer "
            "strictly utilizing this verified context layer. If the context is insufficient, state it clearly.\n\n"
            f"=== VERIFIED CONTEXT BLOCK ===\n{context_payload}\n==============================="
        )

        chat_res = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": payload.prompt}
            ]
        )

        return {"answer": chat_res['message']['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# mem_sync_server.py
import os, json, time
from fastapi import FastAPI, Header, HTTPException, UploadFile, File
from pydantic import BaseModel
import chromadb

AUTH_TOKEN = os.getenv("ELYSIA_SYNC_TOKEN", "changeme")
DB_PATH = os.getenv("ELYSIA_DB_PATH", "./chroma_db")
COLLECTION = os.getenv("ELYSIA_COLLECTION", "persona_memory")

app = FastAPI()
client = chromadb.PersistentClient(path=DB_PATH)
coll = client.get_or_create_collection(name=COLLECTION)

def _ingest_record(r):
    text = r.get("text","")
    speaker = r.get("speaker","unknown")
    tid = r.get("turn_id", f"sys-{int(r.get('ts', time.time()))}")
    rid = f"{speaker}_{tid}_{abs(hash(text))}"
    coll.add(
        documents=[text],
        metadatas=[{"speaker": speaker, "turn_id": tid, "ts": r.get("ts")}],
        ids=[rid]
    )

@app.post("/import-ndjson")
async def import_ndjson(x_auth: str = Header(None), file: UploadFile = File(...)):
    if x_auth != AUTH_TOKEN:
        raise HTTPException(401, "bad auth")
    dedupe = set()
    n_ok = 0
    content = (await file.read()).decode("utf-8", "ignore").splitlines()
    for line in content:
        if not line.strip(): continue
        try:
            r = json.loads(line)
            h = r.get("hash")
            if not h or h in dedupe:
                continue
            dedupe.add(h)
            _ingest_record(r)
            n_ok += 1
        except Exception:
            continue
    return {"ingested": n_ok}

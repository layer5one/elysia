import chromadb
import uuid
import logging

# NEW: journaling
import json, hashlib, time, os
JOURNAL_DIR = os.environ.get("ELYSIA_JOURNAL_DIR", "./mem_journal")
os.makedirs(JOURNAL_DIR, exist_ok=True)

def _journal_path():
    day = time.strftime("%Y-%m-%d")
    return os.path.join(JOURNAL_DIR, f"{day}.ndjson")

def _hash_entry(d: dict) -> str:
    canon = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

def _append_journal(entry: dict):
    # idempotent-ish: embed content hash so the home ingester can dedupe
    entry["hash"] = _hash_entry(entry)
    with open(_journal_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

class ChromaMemoryService:
    """A memory service using ChromaDB for persistent conversational memory."""

    def __init__(self, db_path="./chroma_db", collection_name="persona_memory"):
        """
        Initializes the ChromaDB memory service.
        :param db_path: The directory to persist the database.
        :param collection_name: The name of the collection to store memories.
        """
        logging.info("Initializing ChromaMemoryService...")
        try:
            self._client = chromadb.PersistentClient(path=db_path)
            self._collection = self._client.get_or_create_collection(name=collection_name)
            logging.info(f"ChromaDB client initialized. Using collection '{collection_name}'.")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB: {e}. Ensure SQLite version is >= 3.35.")
            raise

    def add_memory(self, user_input: str, assistant_response: str):
        """
        Adds a conversational turn to the memory.
        :param user_input: The text of the user's input.
        :param assistant_response: The text of the assistant's full response.
        """
        turn_id = str(uuid.uuid4())
        self._collection.add(
            documents=[
                f"User said: {user_input}",
                f"Assistant responded: {assistant_response}"
            ],
            metadatas=[
                {"speaker": "user", "turn_id": turn_id, "ts": time.time()},
                {"speaker": "assistant", "turn_id": turn_id, "ts": time.time()}
            ],
            ids=[f"user_{turn_id}", f"assistant_{turn_id}"]
        )
        logging.info(f"Added memory for turn {turn_id}.")

        # NEW: journal both sides of the turn (append-only, NDJSON)
        _append_journal({
            "type":"turn", "ts": time.time(), "turn_id": turn_id,
            "speaker":"user", "text": user_input
        })
        _append_journal({
            "type":"turn", "ts": time.time(), "turn_id": turn_id,
            "speaker":"assistant", "text": assistant_response
        })

    def add_system_memory(self, system_note: str):
        """Adds a system-level memory, like a self-reflection."""
        note_id = str(uuid.uuid4())
        self._collection.add(
            documents=[system_note],
            metadatas=[{"speaker": "system", "ts": time.time()}],
            ids=[f"system_{note_id}"]
        )
        logging.info(f"Added system memory: '{system_note}'")

        # NEW: journal system notes too
        _append_journal({
            "type":"system", "ts": time.time(),
            "speaker":"system", "text": system_note
        })

    def retrieve_relevant_memories(self, query: str, n_results: int = 5) -> list[str]:
        """
        Retrieves the most relevant memories for a given query.
        :param query: The user's current query.
        :param n_results: The number of results to retrieve.
        :return: A list of the most relevant document strings.
        """
        results = self._collection.query(query_texts=[query], n_results=n_results)

        nested_docs = results.get('documents', [])
        if not nested_docs:
            return []

        retrieved_docs = nested_docs[0]
        logging.info(f"Retrieved {len(retrieved_docs)} memories for query '{query}'.")
        return retrieved_docs

if __name__ == '__main__':
    # Example usage
    memory = ChromaMemoryService()
    memory.add_memory("What's your name?", "I am a conversational AI.")
    memory.add_memory("What can you do?", "I can answer questions and remember our conversations.")

    relevant_memories = memory.retrieve_relevant_memories("What are your capabilities?")
    print("\nRelevant Memories:")
    for mem in relevant_memories:
        print(f"- {mem}")

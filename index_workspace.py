#!/usr/bin/env python3
"""
Semantic search indexer for ~/.openclaw/workspace/projects/
Embeds file chunks and stores in SQLite with sqlite-vec.
"""

import os
import sys
import sqlite3
import struct
from pathlib import Path

# NOTE: `sqlite_vec` and `sentence_transformers` are heavy, optional-at-import
# dependencies. They are imported lazily inside the functions that need them so
# the pure helpers (chunk_text, serialize_vector, deserialize_vector) can be
# imported and unit-tested without pulling in the model/runtime stack.

# Defaults can be overridden via environment variables without editing code.
WORKSPACE_DIR = Path(
    os.environ.get(
        "SEMANTIC_SEARCH_WORKSPACE",
        Path.home() / ".openclaw" / "workspace" / "projects",
    )
)
DB_PATH = Path(
    os.environ.get(
        "SEMANTIC_SEARCH_DB",
        Path.home() / "semantic-search" / "workspace.db",
    )
)
CHUNK_SIZE = 500
EXTENSIONS = {".md", ".txt", ".py", ".js", ".html"}
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into ~size char chunks at word boundaries."""
    chunks = []
    text = text.strip()
    while len(text) > size:
        split_at = text.rfind(" ", 0, size)
        if split_at == -1:
            split_at = size
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def serialize_vector(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def deserialize_vector(data: bytes) -> list[float]:
    n = len(data) // 4
    return list(struct.unpack(f"{n}f", data))


def build_db(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL
        )
    """)
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        )
    """)
    db.commit()
    return db


def index_workspace(workspace: Path = WORKSPACE_DIR, db_path: Path = DB_PATH):
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME} (CPU)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    db = build_db(db_path)

    # Clear existing data
    db.execute("DELETE FROM chunks")
    db.execute("DELETE FROM chunk_vectors")
    db.commit()

    files_indexed = 0
    total_chunks = 0
    skipped = 0

    file_list = []
    for root, dirs, files in os.walk(workspace):
        # Skip hidden dirs and node_modules
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
        for fname in files:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in EXTENSIONS:
                file_list.append(fpath)

    print(f"Found {len(file_list)} files to index...")

    for fpath in file_list:
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                continue

            chunks = chunk_text(text, CHUNK_SIZE)
            if not chunks:
                continue

            rel_path = str(fpath.relative_to(Path.home()))
            embeddings = model.encode(chunks, show_progress_bar=False)

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                cur = db.execute(
                    "INSERT INTO chunks (file_path, chunk_index, content) VALUES (?, ?, ?)",
                    (rel_path, i, chunk)
                )
                chunk_id = cur.lastrowid
                db.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_vector(emb.tolist()))
                )
                total_chunks += 1

            db.commit()
            files_indexed += 1

            if files_indexed % 10 == 0:
                print(f"  Indexed {files_indexed}/{len(file_list)} files, {total_chunks} chunks...")

        except Exception as e:
            print(f"  SKIP {fpath}: {e}", file=sys.stderr)
            skipped += 1

    db.close()
    print(f"\nDone. Files indexed: {files_indexed}, Chunks: {total_chunks}, Skipped: {skipped}")
    print(f"Database: {db_path}")
    return files_indexed, total_chunks


def search(query: str, top_k: int = 5, db_path: Path = DB_PATH) -> list[dict]:
    """Search the index and return top_k results with file path and snippet."""
    import sqlite_vec
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    query_emb = model.encode([query])[0]
    query_bytes = serialize_vector(query_emb.tolist())

    db = sqlite3.connect(str(db_path))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    rows = db.execute("""
        SELECT c.file_path, c.content, cv.distance
        FROM chunk_vectors cv
        JOIN chunks c ON c.id = cv.chunk_id
        WHERE cv.embedding MATCH ?
          AND k = ?
        ORDER BY cv.distance
    """, (query_bytes, top_k)).fetchall()

    db.close()

    results = []
    for file_path, content, distance in rows:
        results.append({
            "file": file_path,
            "snippet": content[:200].replace("\n", " "),
            "score": round(1 - distance, 4)
        })
    return results


if __name__ == "__main__":
    index_workspace()

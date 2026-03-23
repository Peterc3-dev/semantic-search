# Semantic Search

A local semantic search engine that indexes text files using sentence-transformer embeddings and stores them in SQLite with vector search.

## Features

- Indexes `.md`, `.txt`, `.py`, `.js`, and `.html` files from a workspace directory
- Generates 384-dimensional embeddings with `all-MiniLM-L6-v2`
- Stores chunks and vectors in SQLite via `sqlite-vec`
- Fast nearest-neighbor search from the command line

## Usage

### Index a workspace

```bash
python index_workspace.py
```

### Search

```bash
python search.py "your query here"
```

Returns the top 5 matching file chunks ranked by cosine similarity.

## Configuration

Edit the constants at the top of `index_workspace.py`:

| Variable | Default | Description |
|---|---|---|
| `WORKSPACE_DIR` | `~/.openclaw/workspace/projects/` | Directory to index |
| `DB_PATH` | `~/semantic-search/workspace.db` | SQLite database location |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `EXTENSIONS` | `.md .txt .py .js .html` | File types to index |

## Tech Stack

- Python 3.11+
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search
- SQLite

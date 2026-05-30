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
python search.py "your query here" --db /path/to/workspace.db --top-k 10
```

Returns the top matching file chunks (5 by default) ranked by cosine similarity.

## Configuration

The workspace and database paths can be set via environment variables (no code
edits needed):

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_SEARCH_WORKSPACE` | `~/.openclaw/workspace/projects/` | Directory to index |
| `SEMANTIC_SEARCH_DB` | `~/semantic-search/workspace.db` | SQLite database location |

The remaining constants are edited at the top of `index_workspace.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `EXTENSIONS` | `.md .txt .py .js .html` | File types to index |

## Development

```bash
pip install -r requirements.txt   # runtime deps
pip install ruff pytest           # lint + test tooling
ruff check .
pytest -q tests
```

The unit tests cover only pure helpers (chunking, vector (de)serialization) and
run without the model/runtime stack — `sentence-transformers` and `sqlite-vec`
are imported lazily inside the functions that use them. CI
(`.github/workflows/ci.yml`) runs lint + these tests on Python 3.11 and 3.12.

## Tech Stack

- Python 3.11+
- [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for vector search
- SQLite

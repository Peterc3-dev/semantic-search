#!/usr/bin/env python3
"""
CLI semantic search tool.

Usage:
    python search.py "query string"
    python search.py "query string" --db /path/to/workspace.db --top-k 10
"""

import argparse
import sys
from pathlib import Path

# Allow running from any directory: import the sibling module by its real
# location rather than a hardcoded home path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from index_workspace import DB_PATH, search  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over an indexed workspace."
    )
    parser.add_argument("query", nargs="+", help="Search query text.")
    parser.add_argument(
        "--db",
        type=Path,
        default=DB_PATH,
        help=f"Path to the SQLite index (default: {DB_PATH}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    args = parser.parse_args()

    query = " ".join(args.query)
    db_path = args.db

    print(f"Searching for: {query!r}\n")

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("Run: python index_workspace.py")
        sys.exit(1)

    results = search(query, top_k=args.top_k, db_path=db_path)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['file']}  (score: {r['score']})")
        print(f"    {r['snippet']}")
        print()


if __name__ == "__main__":
    main()

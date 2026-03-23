#!/usr/bin/env python3
"""
CLI semantic search tool.
Usage: python ~/semantic-search/search.py "query string"
"""

import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path.home() / "semantic-search"))
from index_workspace import search, DB_PATH


def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py <query>")
        print("Example: python search.py 'IDOR vulnerability'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    db_path = Path(sys.argv[-1]) if "--db" in sys.argv else DB_PATH

    print(f"Searching for: {query!r}\n")

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run: python ~/semantic-search/index_workspace.py")
        sys.exit(1)

    results = search(query, top_k=5)

    if not results:
        print("No results found.")
        return

    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['file']}  (score: {r['score']})")
        print(f"    {r['snippet']}")
        print()


if __name__ == "__main__":
    main()

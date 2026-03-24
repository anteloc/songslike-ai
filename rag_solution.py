"""
RAG (Retrieval-Augmented Generation) for song similarity search — LlamaIndex + ChromaDB

Install dependencies:
    pip install llama-index llama-index-vector-stores-chroma chromadb openai tiktoken tqdm

Set your API key:
    export OPENAI_API_KEY="your-key-here"

Usage:
    python rag_solution.py create <name>
    python rag_solution.py index  <name> <directory>
    python rag_solution.py query  <name> "upbeat celtic music" [--top-k 5]
    python rag_solution.py retrieve <name> "upbeat celtic music" [--top-k 5]
    python rag_solution.py estimate <directory>
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import chromadb
import tiktoken
from tqdm import tqdm
from llama_index.core import VectorStoreIndex, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext


# ── 1. Setup ──────────────────────────────────────────────────────────────────

def create_index(collection_name: str = "my_rag_db") -> VectorStoreIndex:
    """Create (or reconnect to) a persistent ChromaDB-backed vector index."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(collection_name)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build an empty index backed by the vector store
    index = VectorStoreIndex([], storage_context=storage_context)
    return index


# ── 2. Indexing ───────────────────────────────────────────────────────────────

TOKEN_TO_DESC: dict[str, str] = {
    "bpm:slow": "slow tempo",
    "bpm:mid":  "mid tempo",
    "bpm:fast": "fast tempo",
    "DOOM":     "high energy",
    "voom":     "moderate energy",
    "hmm":      "low energy",
    "tsee":     "bright treble-heavy sound",
    "tsing":    "bright upper-midrange sound",
    "bwoom":    "dark bass-heavy sound",
    "bzzra":    "noisy distorted timbre",
    "ooh":      "pure tonal character",
    "tak-tak":  "strong rhythmic percussive attacks",
    "ahhh":     "smooth sustained character",
    "shaa":     "broad spectral spread",
    "weee":     "high-pitched dominant frequency",
    "dum":      "bass dominant frequency",
    "shhh":     "silence",
    "mmm":      "uncharacterized sound",
}

_FP_TOKEN_RE = re.compile(r"\{([^}]+)\}")


def fp_to_prose(fp_text: str) -> str:
    """Convert a .fp.txt fingerprint to a single natural-language description."""
    all_tokens: list[str] = []
    for match in _FP_TOKEN_RE.finditer(fp_text):
        all_tokens.extend(match.group(1).split())

    if not all_tokens:
        return ""

    counts = Counter(all_tokens)
    total = len(all_tokens)

    # Extract unique lyric lines; skip NOTE block, index numbers, timestamps
    in_note_block = False
    lyrics: list[str] = []
    seen: set[str] = set()
    for line in fp_text.splitlines():
        line = line.strip()
        if line == "NOTE":
            in_note_block = True
            continue
        if in_note_block:
            if not line:
                in_note_block = False
            continue
        if not line or line.isdigit() or "-->" in line:
            continue
        clean = _FP_TOKEN_RE.sub("", line).strip()
        if clean and clean not in seen:
            seen.add(clean)
            lyrics.append(clean)

    characteristics = [
        TOKEN_TO_DESC[token]
        for token, count in counts.most_common()
        if count / total >= 0.05 and token in TOKEN_TO_DESC
    ]

    prose = "Music characteristics: " + ", ".join(characteristics) + "."
    if lyrics:
        prose += "\nLyrics: " + " / ".join(lyrics[:20])
    return prose


def index_directory(directory: str, index: VectorStoreIndex) -> None:
    """Convert .fp.txt files to prose and index one document per file."""
    paths = [p for p in sorted(Path(directory).iterdir()) if p.is_file()]
    documents = []
    for path in paths:
        prose = fp_to_prose(path.read_text(encoding="utf-8", errors="ignore"))
        if not prose:
            continue
        documents.append(Document(
            text=prose,
            id_=str(path),
            metadata={"file_path": str(path), "file_name": path.name},
        ))

    for doc in tqdm(documents, desc="Indexing", unit="doc"):
        index.insert(doc)
    print(f"Indexed {len(documents)} document(s) from '{directory}'")


# ── 3. Estimation ─────────────────────────────────────────────────────────────

_ENCODER = tiktoken.get_encoding("cl100k_base")


def estimate_directory(directory: str) -> None:
    """Print per-file and total token counts in tokens / kt / Mt."""
    rows = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        n = len(_ENCODER.encode(path.read_text(encoding="utf-8", errors="ignore")))
        rows.append((path.name, n))

    total = sum(n for _, n in rows)
    width = max(len(name) for name, _ in rows) if rows else 0

    for name, n in rows:
        print(f"{name:<{width}}  {n:>10,} tokens  {n/1_000:>8.3f} kt  {n/1_000_000:>10.6f} Mt")

    print("-" * (width + 42))
    print(f"{'Total':<{width}}  {total:>10,} tokens  {total/1_000:>8.3f} kt  {total/1_000_000:>10.6f} Mt")


# ── 4. Querying ───────────────────────────────────────────────────────────────

def query(question: str, index: VectorStoreIndex, top_k: int = 3) -> str:
    """Ask a question; returns an LLM answer grounded in your documents."""
    engine = index.as_query_engine(similarity_top_k=top_k)
    response = engine.query(question)
    return str(response)


def retrieve(question: str, index: VectorStoreIndex, top_k: int = 3) -> list[str]:
    """Return the raw matching chunks without LLM synthesis (pure retrieval)."""
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)
    return [node.get_content() for node in nodes]


# ── 5. CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG search for similar songs, for examples run: python rag_solution.py <command> --help")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fmt = argparse.RawDescriptionHelpFormatter

    p_create = sub.add_parser("create", help="Create a new empty named index.",
                               formatter_class=fmt,
                               epilog="example:\n  python rag_solution.py create songs")
    p_create.add_argument("name")

    p_index = sub.add_parser("index", help="Populate index from a directory (creates if needed).",
                              formatter_class=fmt,
                              epilog="example:\n  python rag_solution.py index songs ./srt_files/")
    p_index.add_argument("name")
    p_index.add_argument("directory")

    p_query = sub.add_parser("query", help="Ask a question; returns an LLM-synthesized answer.",
                              formatter_class=fmt,
                              epilog="example:\n  python rag_solution.py query songs \"upbeat celtic music\" --top-k 5")
    p_query.add_argument("name")
    p_query.add_argument("question")
    p_query.add_argument("--top-k", type=int, default=3, metavar="K")

    p_retrieve = sub.add_parser("retrieve", help="Return raw matching chunks without LLM synthesis.",
                                 formatter_class=fmt,
                                 epilog="example:\n  python rag_solution.py retrieve songs \"upbeat celtic music\" --top-k 5")
    p_retrieve.add_argument("name")
    p_retrieve.add_argument("question")
    p_retrieve.add_argument("--top-k", type=int, default=3, metavar="K")
    p_retrieve.add_argument("--full", "-f", action="store_true", help="Print the full chunk instead of the first 120 characters.")
    p_retrieve.add_argument("--metadata", "-m", action="store_true", help="Output only metadata for each result in JSON format.")

    p_estimate = sub.add_parser("estimate", help="Estimate token counts for files in a directory.",
                                 formatter_class=fmt,
                                 epilog="example:\n  python rag_solution.py estimate ./srt_files/")
    p_estimate.add_argument("directory")

    args = parser.parse_args()

    if args.cmd == "estimate":
        estimate_directory(args.directory)
        return

    index = create_index(args.name)

    if args.cmd == "create":
        print(f"Index '{args.name}' ready.")
    elif args.cmd == "index":
        index_directory(args.directory, index)
    elif args.cmd == "query":
        print(query(args.question, index, args.top_k))
    elif args.cmd == "retrieve":
        if args.metadata:
            nodes = index.as_retriever(similarity_top_k=args.top_k).retrieve(args.question)
            print(json.dumps([node.metadata for node in nodes], indent=2))
        else:
            for i, chunk in enumerate(retrieve(args.question, index, args.top_k), 1):
                print(f"[{i}] {chunk if args.full else chunk[:120]}")


if __name__ == "__main__":
    main()

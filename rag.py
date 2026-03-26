"""
RAG (Retrieval-Augmented Generation) for song similarity search — LlamaIndex + ChromaDB

Install dependencies:
    pip install llama-index llama-index-vector-stores-chroma chromadb openai tiktoken tqdm

Set your API key:
    export OPENAI_API_KEY="your-key-here"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DUAL-INDEX ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each song is indexed TWICE, in two separate ChromaDB collections:

  <name>_acoustic  — built from .mus.txt files (musical descriptors, SRT format)
  <name>_lyric     — built from .lrc.txt files (plain lyrics companions)

At retrieval time both indexes are queried independently, then results
are merged by file stem using a weighted score:

  final_score = alpha * acoustic_score + (1 - alpha) * lyric_score

Default alpha = 0.6  (acoustic slightly dominant).
Songs with no lyrics ("instrumental") still get a lyric score from their
NOTE header (Tempo/Key/Mode), so they are never silently penalised.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED DIRECTORY LAYOUT FOR INDEXING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use --mus-dir and --lrc-dir to point at separate directories, OR keep both
file types together in a single directory (--dir).  Files are matched by
their stem (the filename without extension):

  104.AC_DC-Thunderstruck.mus.txt   ←→   104.AC_DC-Thunderstruck.lrc.txt

If a song has a .mus.txt but no matching .lrc.txt (or vice-versa) it is
still indexed in the available collection — the missing collection simply
contributes a score of 0.0 for that song at retrieval time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Index from a single directory containing both .mus.txt and .lrc.txt
  python rag.py index <index_base> --dir ./songs/

  # Index from separate directories
  python rag.py index <index_base> --mus-dir ./mus/ --lrc-dir ./lrcs/

  # Index using a custom database path
  python rag.py index <index_base> --dir ./songs/ --db ./my_db

  # Retrieve similar songs using a .mus.txt file as query
  # (companion .lrc.txt with the same stem is used automatically if present)
  python rag.py retrieve <index_base> --file song.mus.txt --top-k 10

  # Retrieve with custom acoustic/lyric weight (0.0–1.0, default 0.6)
  python rag.py retrieve <index_base> --file song.mus.txt --alpha 0.7

  # Retrieve using a plain text question
  python rag.py retrieve <index_base> "fast minor key hard rock"

  # Estimate token counts for a directory
  python rag.py estimate --dir ./songs/
"""

import argparse
import json
import re

from pathlib import Path

import chromadb
import tiktoken
from tqdm import tqdm

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Prevent LlamaIndex from chunking song documents — the default 1024-token limit
# splits long acoustic arcs into partial fragments that fail to self-match at query time.
# 10 240 tokens comfortably holds even the longest .mus.txt or .lrc.txt as one chunk.
Settings.chunk_size = 10_240


# ── 1. Index setup ────────────────────────────────────────────────────────────

def _make_index(collection_name: str, db_path: str) -> VectorStoreIndex:
    """Create (or reconnect to) one named ChromaDB-backed vector index."""
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection    = chroma_client.get_or_create_collection(collection_name)
    vector_store  = ChromaVectorStore(chroma_collection=collection)
    storage_ctx   = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex([], storage_context=storage_ctx)


def get_indexes(index_base: str, db_path: str) -> tuple[VectorStoreIndex, VectorStoreIndex]:
    """Return (acoustic_index, lyric_index) for the given base name and database path."""
    return _make_index(f"{index_base}_acoustic", db_path), _make_index(f"{index_base}_lyric", db_path)


# ── 2. Parsing helpers ────────────────────────────────────────────────────────

# Matches "Tempo: ...", "Key: ...", "Mode: ...", "Instrumentation: ..." lines in the NOTE block
_NOTE_GLOBAL_RE = re.compile(r"^(Tempo|Key|Mode|Instrumentation):\s*(.+)$")

# Used in mus_to_acoustic_prose to give instrumentation an extra repetition
_INSTRUMENTATION_RE = re.compile(r"^Instrumentation:\s*(.+)$", re.MULTILINE)

# Matches the timestamp line in SRT segments: "HH:MM:SS,ms --> ..."
_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d+\s+-->")

# Metadata prefixes to strip from the retrieval query (avoids self-matching by name)
_METADATA_PREFIXES = ("Artist:", "Title:", "Album:", "NOTE")

# Stop-words filtered from lyric prose so grammatical glue words don't inflate similarity
_LYRIC_STOPWORDS = {
    "the", "a", "an", "i", "you", "we", "they", "he", "she", "it",
    "and", "or", "but", "in", "on", "at", "to", "of", "for", "with",
    "is", "was", "are", "be", "been", "have", "had", "has", "do", "did",
    "not", "so", "if", "when", "all", "just", "that", "this", "what",
    "my", "your", "our", "their", "me", "him", "her", "us", "up", "out",
    "no", "oh", "yeah", "ooh", "ahh", "it's", "i'm", "you're", "we're",
    "don't", "can't", "won't", "gonna", "gotta", "wanna", "ain't",
}


def _parse_note_block(text: str) -> list[str]:
    """
    Extract Tempo/Key/Mode descriptors from the NOTE block.

    Values are already plain text (e.g. "mid tempo", "A", "minor") — they are
    returned as-is, except Key which gets a " key" suffix for clarity.
    """
    descs: list[str] = []
    in_note = False
    for line in text.splitlines():
        s = line.strip()
        if s == "NOTE":
            in_note = True
            continue
        if in_note:
            if not s:
                break
            m = _NOTE_GLOBAL_RE.match(s)
            if m:
                field, value = m.group(1), m.group(2).strip()
                descs.append(f"{value} key" if field == "Key" else value)
    return descs


def _clean_lyric_line(line: str) -> str:
    """
    Return a lyric line with stop-words removed, lowercased, keeping only
    alphabetic tokens.  Returns empty string if nothing meaningful remains.
    """
    words = re.findall(r"[a-z']+", line.lower())
    meaningful = [w for w in words if re.sub(r"[^a-z]", "", w) not in _LYRIC_STOPWORDS]
    return " ".join(meaningful)


def _strip_metadata_from_prose(prose: str) -> str:
    """Remove artist/title/album/NOTE lines from a prose string (used at query time)."""
    return "\n".join(
        line for line in prose.splitlines()
        if not any(line.strip().startswith(p) for p in _METADATA_PREFIXES)
    )


# ── 3a. Acoustic prose ────────────────────────────────────────────────────────

def mus_to_acoustic_prose(mus_text: str) -> str:
    """
    Convert a .mus.txt file to natural-language prose for the acoustic index.

    The NOTE block provides song-level properties (Tempo/Key/Mode), repeated
    twice so the embedding model weights them more heavily.

    Each SRT segment's descriptor line is preserved in order and joined as a
    temporal arc ("Arc: ...") separated by " — ".  Keeping the full chronological
    sequence makes each song's acoustic document unique: two songs may share the
    same vocabulary yet differ in *when* they are loud, bright, sparse, etc.
    This ensures a query built from the same file always self-matches near 1.0.
    """
    global_descs = _parse_note_block(mus_text)

    # Walk lines after the NOTE block; collect each descriptor line as one segment.
    # Segment format: <integer> / <timestamp> / <comma-separated descriptors> / <blank>
    in_note   = False
    note_done = False
    segments: list[str] = []

    for line in mus_text.splitlines():
        s = line.strip()
        if s == "NOTE":
            in_note = True
            continue
        if in_note:
            if not s:
                in_note   = False
                note_done = True
            continue
        if not note_done or not s:
            continue
        if s.isdigit() or _TIMESTAMP_RE.match(s):
            continue
        # Descriptor line — normalise commas to spaces to keep prose clean
        terms = " ".join(t.strip() for t in s.split(",") if t.strip())
        segments.append(terms)

    parts: list[str] = []

    if global_descs:
        g = ", ".join(global_descs)
        parts.append(f"Song: {g}.")
        parts.append(f"Overall feel: {g}.")

    if segments:
        parts.append("Arc: " + " — ".join(segments) + ".")

    # Instrumentation gets a third repetition — it's the primary timbre discriminant
    # and would otherwise appear only twice (inside "Song:" and "Overall feel:").
    m = _INSTRUMENTATION_RE.search(mus_text)
    if m:
        parts.append(f"Instrumentation: {m.group(1).strip()}.")

    return "\n".join(parts)


# ── 3b. Lyric prose ───────────────────────────────────────────────────────────

def lrc_to_lyric_prose(lrc_text: str) -> str:
    """
    Convert a .lrc.txt file to natural-language prose for the lyric index.

    Song-level properties (Tempo/Key/Mode) from the NOTE header are included
    so that even purely-lyric queries still incorporate some acoustic context.
    Lyric lines are stop-word filtered so that grammatical glue words don't
    inflate similarity between thematically unrelated songs.

    Songs with no lyrics (containing only "instrumental") produce a short prose
    with just the song-level properties — they will score 0.0 on lyric overlap
    but are never silently excluded from merged results.
    """
    global_descs = _parse_note_block(lrc_text)

    # Collect lyric lines (everything after the blank line that ends the NOTE block)
    in_note   = False
    note_done = False
    raw_lines: list[str] = []

    for line in lrc_text.splitlines():
        s = line.strip()
        if s == "NOTE":
            in_note = True
            continue
        if in_note:
            if not s:
                in_note   = False
                note_done = True
            continue
        if not note_done:
            continue
        if s:
            raw_lines.append(s)

    parts: list[str] = []

    if global_descs:
        g = ", ".join(global_descs)
        parts.append(f"Song characteristics: {g}.")

    if raw_lines and raw_lines != ["instrumental"]:
        cleaned = [_clean_lyric_line(line) for line in raw_lines]
        meaningful = [c for c in cleaned if c]
        if meaningful:
            parts.append("Lyrics: " + " / ".join(meaningful))

    return "\n".join(parts)


# ── 4. Indexing ───────────────────────────────────────────────────────────────

def _stem(path: Path) -> str:
    """
    Return a canonical song stem from a file path, stripping both the file
    extension and the type suffix (.mus or .lrc).

    Examples
    --------
    104.AC_DC-Thunderstruck.mus.txt  ->  104.AC_DC-Thunderstruck
    104.AC_DC-Thunderstruck.lrc.txt  ->  104.AC_DC-Thunderstruck
    """
    name = path.name
    for suffix in (".mus.txt", ".lrc.txt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def index_directories(
    acoustic_index: VectorStoreIndex,
    lyric_index:    VectorStoreIndex,
    mus_dir:  Path | None,
    lrc_dir:  Path | None,
) -> None:
    """
    Populate the acoustic and lyric indexes from their respective directories.

    mus_dir — directory containing .mus.txt files  (musical descriptors)
    lrc_dir — directory containing .lrc.txt files  (lyrics companions)

    Either directory may be None if only one type is being re-indexed.
    Files that match neither pattern are silently skipped.
    """
    if mus_dir:
        mus_paths = sorted(p for p in mus_dir.iterdir() if p.is_file() and p.name.endswith(".mus.txt"))
        mus_docs  = []
        for path in mus_paths:
            prose = mus_to_acoustic_prose(path.read_text(encoding="utf-8", errors="ignore"))
            if prose:
                meta = {"file_path": str(path), "file_name": path.name, "stem": _stem(path)}
                mus_docs.append(Document(
                    text=prose,
                    id_=str(path),
                    metadata=meta,
                    # Keep metadata out of the embedding — only the prose is embedded,
                    # so query text (same prose, no metadata) matches the stored vector exactly.
                    excluded_embed_metadata_keys=list(meta.keys()),
                ))
        for doc in tqdm(mus_docs, desc="Indexing acoustic", unit="doc"):
            acoustic_index.insert(doc)
        print(f"Indexed {len(mus_docs)} acoustic document(s) from '{mus_dir}'")

    if lrc_dir:
        lrc_paths = sorted(p for p in lrc_dir.iterdir() if p.is_file() and p.name.endswith(".lrc.txt"))
        lrc_docs  = []
        for path in lrc_paths:
            prose = lrc_to_lyric_prose(path.read_text(encoding="utf-8", errors="ignore"))
            if prose:
                meta = {"file_path": str(path), "file_name": path.name, "stem": _stem(path)}
                lrc_docs.append(Document(
                    text=prose,
                    id_=str(path),
                    metadata=meta,
                    excluded_embed_metadata_keys=list(meta.keys()),
                ))
        for doc in tqdm(lrc_docs, desc="Indexing lyrics  ", unit="doc"):
            lyric_index.insert(doc)
        print(f"Indexed {len(lrc_docs)} lyric document(s) from '{lrc_dir}'")


# ── 5. Retrieval & merging ────────────────────────────────────────────────────

def _retrieve_with_scores(
    index: VectorStoreIndex,
    query: str,
    top_k: int,
) -> list[tuple[str, float, dict]]:
    """
    Retrieve top_k nodes from index and return
    [(stem, score, metadata), …] sorted by descending score.
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    results = []
    for node in nodes:
        meta  = node.metadata or {}
        stem_ = meta.get("stem") or _stem(Path(meta.get("file_name", "unknown")))
        score = float(node.score) if node.score is not None else 0.0
        results.append((stem_, score, meta))
    return results


def retrieve_merged(
    acoustic_index: VectorStoreIndex,
    lyric_index:    VectorStoreIndex,
    acoustic_query: str,
    lyric_query:    str,
    top_k: int,
    alpha: float = 0.6,
) -> list[dict]:
    """
    Query both indexes and merge results by song stem.

    final_score = alpha * acoustic_score + (1 - alpha) * lyric_score

    Returns a list of dicts sorted by descending final_score:
      {"stem": …, "acoustic_score": …, "lyric_score": …,
       "final_score": …, "metadata": …}

    alpha=1.0  → pure acoustic retrieval
    alpha=0.0  → pure lyric retrieval
    alpha=0.6  → default: acoustic slightly dominant
    """
    # Fetch more candidates than top_k from each index so that the merge
    # doesn't miss songs that rank just outside top_k in one index but
    # would rank highly after combining scores.
    fetch_k = max(top_k * 3, 30)

    acoustic_results = _retrieve_with_scores(acoustic_index, acoustic_query, fetch_k)
    lyric_results    = _retrieve_with_scores(lyric_index,    lyric_query,    fetch_k)

    # Normalise scores within each result set to [0, 1]
    def _normalise(results: list[tuple[str, float, dict]]) -> dict[str, tuple[float, dict]]:
        if not results:
            return {}
        scores = [s for _, s, _ in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo if hi > lo else 1.0
        return {stem: ((s - lo) / span, meta) for stem, s, meta in results}

    acoustic_norm = _normalise(acoustic_results)
    lyric_norm    = _normalise(lyric_results)

    all_stems = set(acoustic_norm) | set(lyric_norm)
    merged: list[dict] = []
    for stem in all_stems:
        a_score, a_meta = acoustic_norm.get(stem, (0.0, {}))
        l_score, l_meta = lyric_norm.get(stem,    (0.0, {}))
        meta = a_meta if a_meta else l_meta
        final = alpha * a_score + (1.0 - alpha) * l_score
        merged.append({
            "stem":           stem,
            "acoustic_score": round(a_score, 4),
            "lyric_score":    round(l_score, 4),
            "final_score":    round(final,   4),
            "metadata":       meta,
        })

    merged.sort(key=lambda x: x["final_score"], reverse=True)
    return merged[:top_k]


# ── 6. Estimation ─────────────────────────────────────────────────────────────

_ENCODER = tiktoken.get_encoding("cl100k_base")


def estimate_directory(directory: Path) -> None:
    rows = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        n = len(_ENCODER.encode(path.read_text(encoding="utf-8", errors="ignore")))
        rows.append((path.name, n))

    if not rows:
        print("No files found.")
        return

    total = sum(n for _, n in rows)
    width = max(len(name) for name, _ in rows)
    for name, n in rows:
        print(f"{name:<{width}} {n:>10,} tokens {n/1_000:>8.3f} kt {n/1_000_000:>10.6f} Mt")
    print("-" * (width + 42))
    print(f"{'Total':<{width}} {total:>10,} tokens {total/1_000:>8.3f} kt {total/1_000_000:>10.6f} Mt")


# ── 7. CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-index RAG search for similar songs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    fmt = argparse.RawDescriptionHelpFormatter

    # index
    p_index = sub.add_parser(
        "index",
        help="Populate acoustic and/or lyric indexes from directories (creates if needed).",
        formatter_class=fmt,
        epilog=(
            "examples:\n"
            "  # single dir with both .mus.txt and .lrc.txt files\n"
            "  python rag.py index songs --dir ./songs/\n\n"
            "  # separate dirs\n"
            "  python rag.py index songs --mus-dir ./mus/ --lrc-dir ./lrcs/\n\n"
            "  # custom db path\n"
            "  python rag.py index songs --dir ./songs/ --db ./my_db\n"
        ),
    )
    p_index.add_argument("index_base")
    p_index.add_argument("--dir",     metavar="DIR",     help="Directory containing both .mus.txt and .lrc.txt files")
    p_index.add_argument("--mus-dir", metavar="MUS_DIR", help="Directory containing .mus.txt files only")
    p_index.add_argument("--lrc-dir", metavar="LRC_DIR", help="Directory containing .lrc.txt files only")
    p_index.add_argument("--db",      metavar="DB",      default="./songs_db",
                         help="ChromaDB storage path (default: ./songs_db)")

    # retrieve
    p_retrieve = sub.add_parser(
        "retrieve",
        help="Return merged similar songs ranked by combined acoustic+lyric score.",
        formatter_class=fmt,
        epilog=(
            "examples:\n"
            "  python rag.py retrieve songs --file song.mus.txt --top-k 10\n"
            "  python rag.py retrieve songs --file song.mus.txt --alpha 0.8\n"
            "  python rag.py retrieve songs \"fast minor key hard rock\" --top-k 5\n"
            "  python rag.py retrieve songs --file song.mus.txt --db ./my_db\n"
        ),
    )
    p_retrieve.add_argument("index_base")
    p_retrieve.add_argument("question", nargs="?", default=None,
                             help="Free-text query (used for both acoustic and lyric indexes)")
    p_retrieve.add_argument("--file",  "-F", metavar="FILE",
                             help="Use a .mus.txt file as the query; companion .lrc.txt is used automatically if present")
    p_retrieve.add_argument("--top-k", type=int, default=10, metavar="K")
    p_retrieve.add_argument(
        "--alpha", type=float, default=0.6, metavar="A",
        help="Acoustic weight 0.0–1.0 (default 0.6). 1.0=pure acoustic, 0.0=pure lyric.",
    )
    p_retrieve.add_argument("--metadata", "-m", action="store_true",
                             help="Output metadata JSON instead of score table")
    p_retrieve.add_argument("--scores",   "-s", action="store_true",
                             help="Show individual acoustic/lyric scores alongside final score")
    p_retrieve.add_argument("--db",      metavar="DB",      default="./songs_db",
                             help="ChromaDB storage path (default: ./songs_db)")

    # estimate
    p_estimate = sub.add_parser("estimate", help="Estimate token counts for files in a directory.",
                                 formatter_class=fmt,
                                 epilog="example:\n  python rag.py estimate --dir ./songs/")
    p_estimate.add_argument("--dir", metavar="DIR", required=True)

    args = parser.parse_args()

    # ── estimate (no index needed) ────────────────────────────────────────
    if args.cmd == "estimate":
        estimate_directory(Path(args.dir))
        return

    acoustic_index, lyric_index = get_indexes(args.index_base, args.db)

    # ── index ─────────────────────────────────────────────────────────────
    if args.cmd == "index":
        if args.dir:
            d = Path(args.dir)
            index_directories(acoustic_index, lyric_index, mus_dir=d, lrc_dir=d)
        else:
            mus_dir = Path(args.mus_dir) if args.mus_dir else None
            lrc_dir = Path(args.lrc_dir) if args.lrc_dir else None
            if not mus_dir and not lrc_dir:
                parser.error("index requires --dir, --mus-dir, or --lrc-dir")
            index_directories(acoustic_index, lyric_index, mus_dir=mus_dir, lrc_dir=lrc_dir)
        return

    # ── retrieve ──────────────────────────────────────────────────────────
    if args.cmd == "retrieve":
        if args.file:
            mus_path   = Path(args.file)
            mus_text   = mus_path.read_text(encoding="utf-8")
            acoustic_query = _strip_metadata_from_prose(mus_to_acoustic_prose(mus_text))

            # Derive the companion .lrc.txt path from the .mus.txt path and use
            # it for the lyric query if it exists; otherwise fall back to the
            # acoustic query (the NOTE header alone still provides useful signal).
            lrc_path = mus_path.with_suffix("").with_suffix(".lrc.txt")
            if lrc_path.exists():
                lrc_text    = lrc_path.read_text(encoding="utf-8")
                lyric_query = _strip_metadata_from_prose(lrc_to_lyric_prose(lrc_text))
            else:
                lyric_query = acoustic_query

        elif args.question:
            acoustic_query = args.question
            lyric_query    = args.question
        else:
            p_retrieve.error("provide either a question or --file")

        results = retrieve_merged(
            acoustic_index, lyric_index,
            acoustic_query, lyric_query,
            top_k=args.top_k,
            alpha=args.alpha,
        )

        if args.metadata:
            print(json.dumps([r["metadata"] for r in results], indent=2))
        elif args.scores:
            print(f"{'#':<4} {'final':>7} {'acou':>7} {'lyric':>7}  stem")
            print("-" * 60)
            for i, r in enumerate(results, 1):
                print(
                    f"{i:<4} {r['final_score']:>7.4f} "
                    f"{r['acoustic_score']:>7.4f} {r['lyric_score']:>7.4f}  "
                    f"{r['stem']}"
                )
        else:
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['stem']}")


if __name__ == "__main__":
    main()

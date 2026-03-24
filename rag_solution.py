"""
Simple RAG (Retrieval-Augmented Generation) with LlamaIndex + ChromaDB

Install dependencies:
    pip install llama-index llama-index-vector-stores-chroma chromadb openai

Set your API key:
    export OPENAI_API_KEY="your-key-here"
"""

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
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

def index_directory(directory: str, index: VectorStoreIndex) -> None:
    """Load every file in a folder and add it to the index."""
    documents = SimpleDirectoryReader(directory).load_data()
    for doc in documents:
        index.insert(doc)
    print(f"✅ Indexed {len(documents)} document(s) from '{directory}'")


def index_text(text: str, index: VectorStoreIndex, doc_id: str = None) -> None:
    """Add a raw text string directly to the index."""
    doc = Document(text=text, id_=doc_id or text[:40])
    index.insert(doc)
    print(f"✅ Indexed text snippet: '{text[:60]}...'")


# ── 3. Querying ───────────────────────────────────────────────────────────────

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


# ── 4. Main demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Build / reconnect to the index ---
    index = create_index("demo")

    # --- Add some documents (pick one approach) ---

    # Option A: index every file in a folder
    # index_directory("./my_docs", index)

    # Option B: index raw text snippets
    index_text("The Eiffel Tower is located in Paris, France. It was built in 1889.", index)
    index_text("Python was created by Guido van Rossum and first released in 1991.", index)
    index_text("RAG stands for Retrieval-Augmented Generation. It combines search with LLMs.", index)

    # --- Query with LLM answer ---
    print("\n🔍 Query: 'Where is the Eiffel Tower?'")
    answer = query("Where is the Eiffel Tower?", index)
    print(f"💬 Answer: {answer}")

    # --- Pure retrieval (no LLM) ---
    print("\n📄 Raw chunks for 'What is RAG?':")
    chunks = retrieve("What is RAG?", index)
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk[:120]}")

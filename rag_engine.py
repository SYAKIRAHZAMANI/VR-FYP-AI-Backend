import os
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# 1. Setup Vector Database
if not os.path.exists("coaching_rules.txt"):
    print("Error: coaching_rules.txt not found!")
    exit()

PERSIST_DIR = "./chroma_db"
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Only re-embed if no persisted DB exists yet. Delete the chroma_db/ folder
# manually if you ever change coaching_rules.txt and need to force a rebuild.
if os.path.exists(PERSIST_DIR):
    print("[rag_engine] Loading existing vector store from disk (fast)...")
    t0 = time.time()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
    print(f"[rag_engine] Loaded in {time.time() - t0:.2f}s")
else:
    print("[rag_engine] No existing vector store found - building from scratch (slow, one-time cost)...")
    loader = TextLoader("coaching_rules.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=150,      # Smaller chunks for high precision
        chunk_overlap=20     # Overlap so sentences don't get cut in half
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[rag_engine] Embedding {len(chunks)} chunks...")

    t0 = time.time()
    vectorstore = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=PERSIST_DIR
    )
    print(f"[rag_engine] Built and persisted in {time.time() - t0:.2f}s")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 2. Setup LLM
llm = ChatOllama(model="llama3.2:1b", num_predict=40)

# Shared cache - imported by intelligent_agent.py's nodes
feedback_cache = {}
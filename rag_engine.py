import os
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

# ---------------------------------------------------------------------------
# SETTINGS YOU MAY WANT TO CHANGE
# ---------------------------------------------------------------------------
# Tips are generated ONCE at server start-up, so speed does not matter.
# Use the biggest model your PC can run. Pull it first, e.g.:  ollama pull llama3.2:3b
LLM_MODEL = "llama3.2:3b"      # try "llama3.1:8b" if you have 8 GB+ of free RAM/VRAM
PERSIST_DIR = "./chroma_db"
# ---------------------------------------------------------------------------

# 1. Setup Vector Database
if not os.path.exists("coaching_rules.txt"):
    print("Error: coaching_rules.txt not found!")
    exit()

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Only re-embed if no persisted DB exists yet. DELETE the chroma_db/ folder
# whenever you change coaching_rules.txt OR the splitter settings below.
if os.path.exists(PERSIST_DIR):
    print("[rag_engine] Loading existing vector store from disk (fast)...")
    t0 = time.time()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings_model)
    print(f"[rag_engine] Loaded in {time.time() - t0:.2f}s")
else:
    print("[rag_engine] No existing vector store found - building from scratch (slow, one-time cost)...")
    loader = TextLoader("coaching_rules.txt", encoding="utf-8")
    docs = loader.load()

    # coaching_rules.txt = blocks of "SOURCE: ...\n<paragraph>" separated by a blank line.
    # Split on the BLANK LINE so every chunk is one complete "SOURCE + advice" block.
    # (Splitting on a single newline with chunk_size=150 separated every SOURCE title
    #  from its paragraph, so title-only chunks with NO advice could be retrieved.)
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_documents(docs)
    print(f"[rag_engine] Embedding {len(chunks)} chunks...")

    t0 = time.time()
    vectorstore = Chroma.from_documents(
        chunks, embeddings_model, persist_directory=PERSIST_DIR
    )
    print(f"[rag_engine] Built and persisted in {time.time() - t0:.2f}s")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Setup LLM  (low temperature = stays on topic; 60 tokens is plenty for one sentence)
llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=60)

# Shared cache - imported by intelligent_agent.py's nodes
feedback_cache = {}
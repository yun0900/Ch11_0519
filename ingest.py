"""
Step 1-3 of the slide A.5 exercise:
  1. Load CLRS Ch11 text from data/
  2. Split into 200-token chunks
  3. SHA-256 dedup, embed with all-MiniLM-L6-v2, persist to Chroma (hnswlib)

Run:  python ingest.py
"""

import hashlib
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "clrs_ch11"


def load_docs():
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"[load]   讀入 {len(docs)} 份文件,共 {sum(len(d.page_content) for d in docs)} 字元")
    return docs


def chunk_docs(docs, chunk_size=200, chunk_overlap=20):
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
    )
    chunks = splitter.split_documents(docs)
    print(f"[chunk]  切成 {len(chunks)} 塊(每塊 {chunk_size} token,overlap {chunk_overlap})")
    return chunks


def dedup_by_sha256(chunks):
    """投影片 A.5 的『內容雜湊 SHA-256』:重複內容跳過 embedding。"""
    seen, unique, dup = set(), [], 0
    for c in chunks:
        h = hashlib.sha256(c.page_content.encode("utf-8")).hexdigest()
        if h in seen:
            dup += 1
            continue
        seen.add(h)
        c.metadata["sha256"] = h
        c.metadata["chunk_id"] = f"clrs11-{len(unique):04d}"
        unique.append(c)
    print(f"[dedup]  SHA-256 命中 {dup} 塊重複,剩 {len(unique)} 塊送入 embedding")
    return unique


def build_index(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name=COLLECTION,
    )
    print(f"[chroma] 已寫入 {DB_DIR}(collection={COLLECTION},筆數={db._collection.count()})")
    return db


def main():
    docs = load_docs()
    if not docs:
        raise SystemExit(f"data/ 找不到 .md 檔,請放入 CLRS Ch11 文字後重跑。({DATA_DIR})")
    chunks = chunk_docs(docs)
    chunks = dedup_by_sha256(chunks)
    build_index(chunks)
    print("\n下一步:python query.py  或  python eval.py")


if __name__ == "__main__":
    main()

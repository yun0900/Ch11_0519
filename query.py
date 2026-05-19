"""
Step 4: 互動式查詢,觀察 top-k 召回。

Run:  python query.py
      然後輸入問題,例如:
        How does chaining resolve collisions?
        除法雜湊如何運作?
        Ctrl+C 離開。
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent
DB_DIR = ROOT / "chroma_db"
COLLECTION = "clrs_ch11"
TOP_K = 5


def open_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


def main():
    db = open_db()
    n = db._collection.count()
    if n == 0:
        raise SystemExit("Chroma 是空的,請先跑 python ingest.py")
    print(f"已載入 {n} 筆。輸入查詢,Ctrl+C 離開。\n")

    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        # Chroma 回傳的是「距離」,越小越像 (cosine distance)
        results = db.similarity_search_with_score(q, k=TOP_K)
        for i, (doc, dist) in enumerate(results, 1):
            preview = " ".join(doc.page_content.split())[:140]
            cid = doc.metadata.get("chunk_id", "?")
            print(f"  #{i}  dist={dist:.4f}  id={cid}\n      {preview}…")
        print()


if __name__ == "__main__":
    main()

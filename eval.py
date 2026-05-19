"""
固定一組查詢,自動跑 recall@3,讓同學量化觀察召回率。

每個 case 是 (query, expected_keyword) — 只要 top-3 有任何一塊內文包含
expected_keyword(忽略大小寫),就算命中。同時刻意混入中文 query,
讓同學感受 all-MiniLM-L6-v2 在跨語言上的強弱。

Run:  python eval.py
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT = Path(__file__).parent
DB_DIR = ROOT / "chroma_db"
COLLECTION = "clrs_ch11"

CASES = [
    # 英文 — 與資料同語言,應該幾乎全中
    ("What is a direct-address table?",          "direct-address"),
    ("How does chaining resolve collisions?",    "chaining"),
    ("Explain the division method for hashing.", "division"),
    ("What is universal hashing?",               "universal"),
    ("How does linear probing handle collisions?", "linear probing"),
    ("What is double hashing?",                  "double hashing"),
    ("Define perfect hashing.",                  "perfect hashing"),
    ("What is the load factor in a hash table?", "load factor"),
    # 中文 — 跨語言檢索測試
    ("什麼是雜湊衝突?",                            "collision"),
    ("除法雜湊法怎麼運作?",                        "division"),
    ("什麼是完美雜湊?",                            "perfect hashing"),
]


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    if db._collection.count() == 0:
        raise SystemExit("Chroma 是空的,請先跑 python ingest.py")

    hits = 0
    print(f"{'結果':<4}{'query':<48}{'expected':<20}{'top-1 dist'}")
    print("-" * 90)
    for q, expect in CASES:
        results = db.similarity_search_with_score(q, k=3)
        joined = " ".join(d.page_content.lower() for d, _ in results)
        ok = expect.lower() in joined
        hits += int(ok)
        flag = "✓" if ok else "✗"
        top1 = f"{results[0][1]:.4f}" if results else "n/a"
        q_show = (q[:46] + "…") if len(q) > 47 else q
        print(f" {flag}   {q_show:<48}{expect:<20}{top1}")

    total = len(CASES)
    print("-" * 90)
    print(f"recall@3 = {hits}/{total} = {hits/total:.0%}")
    print("\n觀察重點:")
    print("  1. 英文 query 應幾乎全中,因為文件就是英文。")
    print("  2. 中文 query 命中率較低 — all-MiniLM-L6-v2 主要訓練於英文。")
    print("     若要更好的跨語言效果,可改用 paraphrase-multilingual-MiniLM-L12-v2。")


if __name__ == "__main__":
    main()

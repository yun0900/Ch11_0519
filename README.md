<h1>eval.py 執行結果</h1>
<img width="554" height="263" alt="image" src="https://github.com/user-attachments/assets/917b5656-67a0-46a0-8983-63d7d7401d23" />
<h1>Top-3 對照表</h1>
<img width="554" height="150" alt="image" src="https://github.com/user-attachments/assets/e7fc889f-7ea4-4467-8396-a168c2328391" />
<img width="554" height="154" alt="image" src="https://github.com/user-attachments/assets/4ef29f49-751f-4e0d-baab-b40cbac543b7" />
<img width="554" height="156" alt="image" src="https://github.com/user-attachments/assets/2bfa1cef-5a4b-44c3-b828-a5654b949686" />
<img width="554" height="159" alt="image" src="https://github.com/user-attachments/assets/a024afdd-39f6-4489-aadc-fdeb48c8424e" />

<h1>vLLM 與 Chroma 的異同比較</h1>
# vLLM 與 Chroma 架構技術對比

## 📌 共通點
### 核心目的都是為了「省空間與省算力（去重與快取）」
* **vLLM** 透過雜湊（如 `hash(Block)`）來識別哪些前置提示詞（Prompt Tokens）已經被計算過 KV Cache，藉此直接重用（Automatic Prefix Caching），省下重複計算 Prefill 的算力。
* **Chroma** 在 `ingest.py` 階段利用 SHA-256 雜湊文本，識別出完全相同的重複文本區塊（Chunks）並直接過濾，省下重複呼叫 Embedding 模型的算力與儲存空間。

### 都存在「多層級雜湊/映射」的架構
> 兩者都不是只用一個單純的 Hash Table。
> * **vLLM** 內部有 Token 到實體記憶體物理區塊（Physical Blocks）的雜湊映射表。
> * **Chroma** 內部則如前述，結合了「SHA-256（去重）+ LSH（檢索）+ 倒排索引（過濾）」的三層雜湊骨幹。

### 底層都必須處理「衝突（Collision）」或近似碰撞的課題
為了維持系統的穩定度與精準度，兩個系統都必須處理雜湊碰撞：
* **vLLM** 在維護 Block 雜湊表時必須有嚴格的動態配置與衝突防範。
* **Chroma** 在向量分桶時，更需要依賴高維空間的數學設計來定義什麼是「合理的碰撞」。

---

## 🔍 差異點

| 比較維度 | vLLM | Chroma |
| :--- | :--- | :--- |
| **雜湊的哲學** | **傳統精準雜湊（極力打散）**<br>遵循 CLRS 課本的傳統雜湊哲學。它的目的是追求均勻打散，不希望兩個不同的 Token 序列發生碰撞，必須維持 $O(1)$ 的精準映射，一分一毫都不能錯。 | **位置敏感雜湊 LSH（刻意聚集）**<br>採用語意幾何雜湊。它的目的是刻意讓語意接近的向量發生碰撞並落入同一個桶子（Bucket）。在 Chroma 裡，高維向量「撞在同一個區域」才是檢索加速的關鍵。 |
| **處理的數據對象** | **離散的 Token 序列**<br>雜湊的對象是離散的、確定性的整數 ID 序列（也就是 Prompt 的 Token IDs）。 | **連續的高維實數向量**<br>向量檢索層雜湊的對象是經由深度學習模型產出、由成百上千個浮點數組成的高維連續向量空間（Dense Vectors）。 |
| **生命週期與動態性** | **記憶體內的短暫快取**<br>雜湊表（KV Cache Manager）完全存在於 GPU/CPU 記憶體（RAM）中，生命週期隨著動態推論、LRU 快取淘汰機制、或是伺服器關閉而消失，屬於高頻、極速的記憶體管理。 | **硬碟中的長期持久化索引**<br>雜湊與索引（Chroma DB）主要是為了硬碟持久化儲存（Persistence）設計，一旦透過 `ingest.py` 建立好索引並寫入磁碟，這個雜湊結構就會長期存在，供後續多次啟動 `query.py` 進行靜態知識庫的讀取。 |

# A.5 同學動手: 用 LangChain + Chroma 建小型 RAG

對應投影片 **A.5 RAG 與向量資料庫的雜湊骨幹**(Ch11 p.79)。

這個小實驗會把 CLRS Chapter 11(雜湊表)當語料,跑完 RAG 檢索流程:

1. 把 CLRS Ch11 切成 200-token chunks
2. 用 `all-MiniLM-L6-v2` 產生 embedding
3. 存入 Chroma 向量資料庫(底層 hnswlib)
4. 查詢並觀察 top-k 召回結果

專案也把投影片提到的「**SHA-256 內容雜湊**」做進去,讓同學看到重複片段不會被重複算 embedding。

---

## Anaconda Prompt CLI 詳細操作步驟

> 這份教學假設你使用 Windows 11,並且全程在 **Anaconda Prompt** 裡輸入指令。
> 請不要混用其他終端機,初學時最容易因此卡住。

### Step 0. 環境: 用 conda 建立專用環境

建議環境:

- Windows 11
- Anaconda Prompt
- Python 3.11
- conda 環境名稱: `rag-ch11`

請在 **Anaconda Prompt** 執行以下指令:

```bat
conda create -n rag-ch11 python=3.11 -y
conda activate rag-ch11
cd alg_ch11_p79_ex
python -m pip install --upgrade pip
pip install -r requirements.txt
```

上面是本專案的環境建立方式。  
如果你是第一次操作,請繼續照下面的拆解步驟一步一步做。

### Step 1. 打開 Anaconda Prompt

1. 按鍵盤 **Win**。
2. 搜尋 `Anaconda Prompt`。
3. 點開黑色視窗。
4. 你應該會看到類似下面的提示符:

```bat
(base) C:\Users\你的名字>
```

看到 `(base)` 代表 conda 已經啟動。  
如果沒有看到 `(base)`,請確認你開的是 **Anaconda Prompt**。

先確認 conda 能用:

```bat
conda --version
```

正常會看到類似:

```bat
conda 24.x.x
```

如果出現 `conda is not recognized`,代表 Anaconda/Miniconda 沒裝好,請先安裝 Anaconda 或 Miniconda 後再回來。

---

### Step 2. 用 conda 建立一個乾淨的環境

這裡的 `rag-ch11` 是我們替這個實驗取的 conda 環境名稱。你可以把它想成這個專案專用的小房間,套件都裝在裡面,不會污染其他 Python 專案。

在 Anaconda Prompt 輸入:

```bat
conda create -n rag-ch11 python=3.11 -y
```

這行指令的意思是:

```text
conda create   建立新環境
-n rag-ch11    環境名字叫 rag-ch11
python=3.11    使用 Python 3.11
-y             自動回答 yes
```

第一次建立會下載一些基本套件,請等它跑完。成功時,結尾通常會看到:

```bat
done
#
# To activate this environment, use
#
#     conda activate rag-ch11
```

如果你想確認環境真的建好了,輸入:

```bat
conda env list
```

列表裡應該會看到 `rag-ch11`。

---

### Step 3. 啟動 rag-ch11 環境

輸入:

```bat
conda activate rag-ch11
```

成功後,提示符前面會從 `(base)` 變成 `(rag-ch11)`:

```bat
(rag-ch11) C:\Users\你的名字>
```

這一步非常重要。後面所有 `python`、`pip`、`langchain`、`chromadb` 都會裝在這個環境裡。

確認目前 Python 版本:

```bat
python --version
```

應該看到:

```bat
Python 3.11.x
```

再確認 `python` 和 `pip` 都指向 conda 環境:

```bat
where python
where pip
```

正常情況下,第一個路徑應該包含:

```bat
anaconda3\envs\rag-ch11
```

或:

```bat
miniconda3\envs\rag-ch11
```

如果還是指到別的 Python,請關掉視窗,重新打開 Anaconda Prompt,再跑一次:

```bat
conda activate rag-ch11
```

---

### Step 4. 切到本專案資料夾

進入專案資料夾:

```bat
cd alg_ch11_p79_ex
```

列出檔案:

```bat
dir
```

你應該看得到:

```bat
README.md
requirements.txt
ingest.py
query.py
eval.py
data
```

如果看不到這些檔案,代表你還沒切到正確資料夾,請回頭檢查 `cd` 指令。

---

### Step 5. 更新 pip

在安裝專案套件前,先把 pip 更新到比較新的版本:

```bat
python -m pip install --upgrade pip
```

正常會看到 `Successfully installed pip-...`。  
如果它說 `Requirement already satisfied`,也沒問題,代表原本就夠新。

---

### Step 6. 安裝 requirements.txt 裡的套件

輸入:

```bat
pip install -r requirements.txt
```

這一步會安裝:

- `langchain`
- `chromadb`
- `sentence-transformers`
- `tiktoken`
- 以及它們需要的相依套件

第一次安裝可能需要 2 到 10 分鐘,中間看到大量下載訊息是正常的。

安裝完成後,檢查套件是否能被 Python 載入:

```bat
python -c "import langchain, chromadb, sentence_transformers, tiktoken; print('ok')"
```

看到下面這行就代表安裝成功:

```bat
ok
```

如果出現 `ModuleNotFoundError`,通常是你忘了啟動環境。請確認提示符前面有 `(rag-ch11)`,再重新跑:

```bat
pip install -r requirements.txt
```

---

### Step 7. 建立 Chroma 向量索引

現在開始跑投影片流程的第 1 到第 3 步:

1. 讀取 `data\clrs_ch11.md`
2. 切 chunks
3. SHA-256 去重
4. 算 embedding
5. 寫入 Chroma

輸入:

```bat
python ingest.py
```

第一次執行時,程式會自動下載 embedding 模型 `sentence-transformers/all-MiniLM-L6-v2`,大約 90 MB。看到下載進度請耐心等待。

成功時會看到類似:

```bat
[load]   讀入 1 份文件,共 5832 字元
[chunk]  切成 8 塊(每塊 200 token,overlap 20)
[dedup]  SHA-256 命中 0 塊重複,剩 8 塊送入 embedding
[chroma] 已寫入 alg_ch11_p79_ex\chroma_db(collection=clrs_ch11,筆數=8)

下一步:python query.py  或  python eval.py
```

跑完後,專案資料夾會多一個:

```bat
chroma_db
```

這就是本機向量資料庫。想重建索引時,刪掉它再跑 `python ingest.py` 即可。

---

### Step 8. 互動查詢 top-k 結果

輸入:

```bat
python query.py
```

你會看到:

```bat
已載入 8 筆。輸入查詢,Ctrl+C 離開。

query>
```

接著可以輸入英文問題:

```bat
How does chaining resolve collisions?
```

預期會看到 top-5 檢索結果:

```bat
#1  dist=0.4123  id=clrs11-0002
    In chaining, every slot T[j] points to a doubly linked list...
```

`dist` 是 cosine distance,意思是距離:

```text
dist 越小 = 越相似 = 排名越前面
```

你也可以繼續試:

```bat
What is universal hashing?
Define perfect hashing.
Explain the division method for hashing.
除法雜湊如何運作?
```

想離開互動查詢時,按:

```bat
Ctrl+C
```

---

### Step 9. 跑固定測試,觀察 recall@3

輸入:

```bat
python eval.py
```

這支程式會跑 11 條固定 query,每條檢查 top-3 裡面有沒有命中預期關鍵字。

最後會看到類似:

```bat
recall@3 = 9/11 = 82%
```

觀察重點:

- 英文 query 通常比較容易命中,因為語料本身是英文。
- 中文 query 可能掉幾題,因為 `all-MiniLM-L6-v2` 主要偏英文。
- 如果想改善中文檢索,可以把 embedding 模型換成 `paraphrase-multilingual-MiniLM-L12-v2`。

---

### Step 10. 看 SHA-256 去重效果

先複製一份相同語料:

```bat
copy data\clrs_ch11.md data\clrs_ch11_copy.md
```

刪掉舊索引:

```bat
rmdir /s /q chroma_db
```

重新建立索引:

```bat
python ingest.py
```

這次你應該會看到 `[dedup]` 那行出現重複命中。意思是:內容完全一樣的 chunk 被 SHA-256 偵測出來,所以不會重複送去算 embedding。

看完後,把剛剛複製的檔案刪掉:

```bat
del data\clrs_ch11_copy.md
```

再把索引重建回乾淨狀態:

```bat
rmdir /s /q chroma_db
python ingest.py
```

---

### Step 11. 下次重開電腦後怎麼繼續

下次不用重建環境,只要:

```bat
conda activate rag-ch11
cd alg_ch11_p79_ex
```

然後直接查詢:

```bat
python query.py
```

或跑評估:

```bat
python eval.py
```

只要 `chroma_db` 還在,就不用重新跑 `python ingest.py`。

---

### Step 12. 全部清掉,重新來過

如果你想把這次實驗全部清掉,先退出目前環境:

```bat
conda deactivate
```

刪掉 conda 環境:

```bat
conda env remove -n rag-ch11 -y
```

再刪掉向量資料庫:

```bat
cd alg_ch11_p79_ex
rmdir /s /q chroma_db
```

確認環境刪掉了:

```bat
conda env list
```

列表裡不應該再看到 `rag-ch11`。

---

## 常見錯誤排除

### 1. `conda` 找不到

你可能開錯終端機,或 Anaconda/Miniconda 沒安裝好。請重新打開 **Anaconda Prompt**。

### 2. `python` 版本不是 3.11

先確認環境已啟動:

```bat
conda activate rag-ch11
python --version
```

如果還是不對,刪掉環境重建:

```bat
conda deactivate
conda env remove -n rag-ch11 -y
conda create -n rag-ch11 python=3.11 -y
conda activate rag-ch11
```

### 3. `ModuleNotFoundError`

通常是套件沒有裝在目前環境。請確認提示符前面有 `(rag-ch11)`,再跑:

```bat
pip install -r requirements.txt
```

### 4. `Chroma 是空的,請先跑 python ingest.py`

代表你還沒建立向量索引。請先跑:

```bat
python ingest.py
```

### 5. `pip install` 卡在 `chromadb` 或編譯錯誤

先更新 pip:

```bat
python -m pip install --upgrade pip setuptools wheel
```

再重裝:

```bat
pip install -r requirements.txt
```

如果仍然失敗,可改用 conda-forge 先裝 Chroma:

```bat
conda install -c conda-forge chromadb -y
pip install -r requirements.txt
```

### 6. 下載模型很慢

第一次跑 `python ingest.py` 或 `python query.py` 會下載 embedding 模型。請保持網路連線,等它完成。下載完成後,下次通常會使用快取,不會重新下載。

---

## 對應投影片的觀察

| 投影片重點 | 在這個專案怎麼看到 |
| --- | --- |
| 把 query 轉 embedding,檢索 top-k 再給 LLM | `query.py` 的 `similarity_search_with_score` |
| 多層雜湊並存: chunk_id 去重、embedding 分桶、metadata inverted index | `metadata={"chunk_id", "sha256", "source"}`,Chroma 內部用 hnswlib 做 ANN 分桶 |
| 內容雜湊 SHA-256 避免重複收錄 | `ingest.py` 的 `dedup_by_sha256` |
| Pinecone 公開 10K QPS、p95 < 50 ms | 本機 Chroma 在少量資料上單查通常很快;規模上去才需要 Pinecone / Weaviate |

## 進階小實驗

- **改 chunk 大小**: 把 `ingest.py` 的 `chunk_size` 從 200 改成 50 或 800,看 `eval.py` 的 recall@3 怎麼變。記得改完先 `rmdir /s /q chroma_db`,再 `python ingest.py` 重建索引。
- **換 embedding 模型**: 把 `all-MiniLM-L6-v2` 換成 `paraphrase-multilingual-MiniLM-L12-v2`,中文 query 召回通常會提升。
- **拿掉 normalize_embeddings**: 把 `encode_kwargs` 改成 `{"normalize_embeddings": False}`,觀察距離分數分佈差異。
- **換成真正的 CLRS 文字**: 把 `data\clrs_ch11.md` 換成你自己從 CLRS PDF 擷取的章節文字,其他流程不用改。

## 檔案結構

```text
alg_ch11_p79_ex/
├── README.md             # 本檔
├── requirements.txt
├── data/
│   └── clrs_ch11.md      # 範例語料,可換成你自己的
├── ingest.py             # 步驟 1-3: 切 chunks -> SHA-256 dedup -> embed -> 存 Chroma
├── query.py              # 步驟 4: 互動查詢
├── eval.py               # 步驟 4: 固定 query 跑 recall@3
└── chroma_db/            # 跑完 python ingest.py 才會出現,可整個刪掉重建
```

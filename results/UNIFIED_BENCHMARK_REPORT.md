# Unified & Reproducible RAG Chunking Benchmark Report

## 1. Methodological & Economic Configuration
- **Corpus**: 115 Technical & Regulatory PDF Documents converted via Docling to structured Markdown (4 categories, 24,088,389 characters, 2,246 reconstructed tables comprising 27,773 tabular rows).
- **Evaluation Benchmark**: Strictly balanced **120 queries** (**60 `table_numeric`** and **60 `narrative_text`**; 4 categories × 6 subtypes × 5 queries = 120).
- **Unified Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions, cosine similarity, `$0.02 / 1M tokens`) used for **both** Semantic Chunking (sentence-level breakpoint detection) **and** ChromaDB vector indexing/retrieval across all 6 strategies.
- **Unified LLM Generator & RAGAS Judge**: OpenAI `gpt-6-luna` (`reasoning_effort='low'`, **Full Top-5 Context Judge**, Input: **`$0.10 / 1M tokens`**, Output: **`$0.50 / 1M tokens`**).
- **Retrieval Architecture**: Pure bi-encoder cosine similarity search in ChromaDB (`hnsw:space = cosine`, no cross-encoder re-ranker).

## 2. Corpus Chunking & Ingestion Footprint (Table 4.1)

| Strategy | Total Chunks | Avg Tokens | Median Tokens | Table Chunks | Chunking Time (s) | Peak RAM (MB) | Vector DB (MB) | Ingest Cost ($) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fixed-128** | 44,812 | 127.9 | 128.0 | 10,660 | 18.15s | 103.28 | 262.57 | $0.1146 |
| **Fixed-512** | 11,152 | 509.8 | 512.0 | 3,448 | 9.45s | 50.03 | 65.34 | $0.1137 |
| **Fixed-1024** | 5,931 | 1016.4 | 1024.0 | 2,235 | 8.23s | 49.62 | 34.75 | $0.1206 |
| **Recursive** | 12,710 | 404.3 | 450.0 | 1,928 | 29.76s | 50.88 | 74.47 | $0.1028 |
| **Structural** | 18,345 | 271.8 | 221.0 | 2,718 | 20.05s | 52.61 | 107.49 | $0.0997 |
| **Semantic** | 11,659 | 411.1 | 409.0 | 1,642 | 282.49s | 869.25 | 68.31 | $0.1883 |

## 3. Tier 1 Retrieval Metrics Across 120 Balanced Queries (Table 4.2)

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Token IoU | Table R@5 (n=60) | Narrative R@5 (n=60) | Modality Gap (Δ) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fixed-128** | 0.5167 | 0.7083 | **0.7833** | 0.8833 | **0.6331** | 0.5200 | 0.7333 | 0.8333 | -0.1000 |
| **Fixed-512** | 0.6333 | 0.8083 | **0.8417** | 0.9333 | **0.7350** | 0.3652 | 0.9000 | 0.7833 | +0.1167 |
| **Fixed-1024** | 0.5500 | 0.7167 | **0.8167** | 0.9167 | **0.6591** | 0.2512 | 0.8000 | 0.8333 | -0.0333 |
| **Recursive** | 0.6750 | 0.8083 | **0.8333** | 0.8750 | **0.7481** | 0.4195 | 0.8000 | 0.8667 | -0.0667 |
| **Structural** | 0.7250 | 0.8667 | **0.8917** | 0.9500 | **0.8006** | 0.6047 | 0.9000 | 0.8833 | +0.0167 |
| **Semantic** | 0.4750 | 0.7000 | **0.7500** | 0.8667 | **0.6113** | 0.3274 | 0.7333 | 0.7667 | -0.0334 |

## 4. Granular Subtype Recall@5 Breakdown (20 Queries per Subtype, Table 4.3)

| Strategy | Exact Cell (n=20) | Cross Comp (n=20) | Range Thresh (n=20) | Scope Def (n=20) | Proc Steps (n=20) | Reg Oblig (n=20) |
|:---|---:|---:|---:|---:|---:|---:|
| **Fixed-128** | 0.800 | 0.800 | 0.600 | 0.850 | 0.850 | 0.800 |
| **Fixed-512** | 0.900 | 0.900 | 0.900 | 0.800 | 0.700 | 0.850 |
| **Fixed-1024** | 0.800 | 0.800 | 0.800 | 0.900 | 0.800 | 0.800 |
| **Recursive** | 0.800 | 0.850 | 0.750 | 0.900 | 0.850 | 0.850 |
| **Structural** | 0.950 | 0.900 | 0.850 | 0.950 | 0.900 | 0.800 |
| **Semantic** | 0.700 | 0.700 | 0.800 | 0.900 | 0.700 | 0.700 |

## 5. Tier 2 Generative RAGAS Evaluation via `gpt-6-luna` — Full Top-5 Context (Table 4.4)

| Strategy | Context Precision@5 | Faithfulness (All) | Table Faith (n=60) | Narrative Faith (n=60) | Answer Relevance | AQS |
|:---|---:|---:|---:|---:|---:|---:|
| **Fixed-128** | 0.6048 | **0.9533** | 0.9150 | 0.9915 | **0.8502** | 0.9018 |
| **Fixed-512** | 0.6923 | **0.9947** | 0.9997 | 0.9897 | **0.8960** | 0.9454 |
| **Fixed-1024** | 0.6249 | **0.9885** | 0.9867 | 0.9903 | **0.8258** | 0.9072 |
| **Recursive** | 0.7222 | **0.9773** | 0.9670 | 0.9877 | **0.8509** | 0.9141 |
| **Structural** | 0.7660 | **0.9914** | 0.9933 | 0.9895 | **0.8863** | 0.9388 |
| **Semantic** | 0.5864 | **0.9678** | 0.9593 | 0.9763 | **0.8408** | 0.9043 |

## 6. Operational Cost (`gpt-6-luna`: `$0.10/M` In, `$0.50/M` Out), Pareto Frontier & ICER (Tables 4.5 & 4.6)

| Strategy | Context Tok (k=5) | Retrieval (ms) | Total Latency (ms) | Cost / 1k Queries ($) | Empirical API Cost / 1k ($) | Quality $Q_m$ | Cost Score $C_m$ | Pareto Status | ICER ($\Delta \text{Cost} / \Delta Q_m$) |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|:---|
| **Fixed-128** | 639.9 | 20.2 | 1450.0 | **$0.1395** | $0.1130 | **0.7649** | 0.4676 | **Pareto Optimal** | Baseline Frontier |
| **Fixed-512** | 2542.0 | 33.3 | 1401.8 | **$0.3297** | $0.2929 | **0.8319** | 0.6862 | **Dominated** | N/A (Dominated) |
| **Fixed-1024** | 5071.1 | 15.8 | 1318.7 | **$0.5826** | $0.5368 | **0.7830** | 0.9728 | **Dominated** | N/A (Dominated) |
| **Recursive** | 2012.2 | 15.0 | 1301.1 | **$0.2767** | $0.2420 | **0.8264** | 0.6016 | **Dominated** | N/A (Dominated) |
| **Structural** | 1672.8 | 26.2 | 1327.6 | **$0.2428** | $0.2092 | **0.8672** | 0.5664 | **Pareto Optimal** | $1.01 per +1.0 Qm |
| **Semantic** | 2713.8 | 25.6 | 1357.0 | **$0.3469** | $0.3136 | **0.7513** | 0.6975 | **Dominated** | N/A (Dominated) |
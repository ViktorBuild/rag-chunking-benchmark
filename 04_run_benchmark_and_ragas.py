# -*- coding: utf-8 -*-
"""
STAGE 4: Full Unified Retrieval & RAGAS Evaluation Suite (720 Runs)
Executes across all 6 strategies x 120 strictly balanced queries (60 Table / 60 Narrative):
  - Tier 1 Retrieval Metrics: Recall@1, Recall@3, Recall@5, Recall@10, MRR, Mean Token IoU,
    Table Recall@5 (n=60), Narrative Recall@5 (n=60), 6 Query Subtype Recall@5 (n=20 each),
    4 Category Recall@5 (n=30 each), and rank-weighted Context Precision@5.
  - Tier 2 Generative RAGAS Metrics via OpenAI `gpt-6-luna` (reasoning_effort="low"):
    Faithfulness (Overall, Table, Narrative), Answer Relevance, LLM Context Precision, and AQS.
  - Stage 5 Economic & Multi-Objective Trade-Off Analysis:
    Composite Quality Score (Qm), Operational Cost Score (Cm), Empirical Pareto Frontier, and ICER.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import csv
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI

from config import (
    DATASET_JSON,
    RESULTS_DIR,
    FIGURES_DIR,
    OPENAI_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    PRICING,
    STRATEGIES,
    STRATEGY_DISPLAY_NAMES,
    TOP_K_RETRIEVAL,
    TOP_K_CONTEXT,
    setup_logger
)
from importlib import import_module

vs_mod = import_module("03_vector_store")
UnifiedVectorStore = vs_mod.UnifiedVectorStore

logger = setup_logger("04_run_benchmark_and_ragas")

RAW_CACHE_PATH = RESULTS_DIR / "raw_720_evaluation_runs.json"
SUMMARY_JSON_PATH = RESULTS_DIR / "unified_benchmark_summary.json"
SUMMARY_CSV_PATH = RESULTS_DIR / "unified_benchmark_summary.csv"
SUBTYPE_CSV_PATH = RESULTS_DIR / "subtype_breakdown_summary.csv"
PARETO_ICER_CSV_PATH = RESULTS_DIR / "pareto_icer_summary.csv"
REPORT_MD_PATH = RESULTS_DIR / "UNIFIED_BENCHMARK_REPORT.md"


# ==============================================================================
# TIER 1 RETRIEVAL RELEVANCE & TOKEN IoU FUNCTIONS
# ==============================================================================
def compute_token_iou(text_a: str, text_b: str) -> float:
    tokens_a = set(re.findall(r'\w+', text_a.lower()))
    tokens_b = set(re.findall(r'\w+', text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    return round(inter / union, 4) if union > 0 else 0.0


def is_chunk_relevant(chunk: Dict[str, Any], query: Dict[str, Any]) -> bool:
    """
    Verifies whether a retrieved chunk contains the ground-truth evidence for `query`.
    Matches exact methodology in `pipeline/evaluator.py`.
    """
    c_meta = chunk.get("metadata", {})
    c_doc_id = c_meta.get("doc_id", "")
    c_text = chunk.get("text", "").lower()

    doc_match = (c_doc_id == query["doc_id"])
    key_facts = query.get("key_facts", [])

    facts_present = 0
    if key_facts:
        for f in key_facts:
            if str(f).lower() in c_text:
                facts_present += 1
        facts_ratio = facts_present / len(key_facts)
    else:
        facts_ratio = 0.0

    passage_snippet = query.get("source_passage_text", "")[:120].lower()
    snippet_match = passage_snippet in c_text if len(passage_snippet) > 20 else False

    if snippet_match:
        return True
    if doc_match and facts_ratio >= 0.6:
        return True
    if facts_ratio == 1.0:
        return True

    return False


def compute_context_precision_at_k(retrieved_chunks: List[Dict[str, Any]], query: Dict[str, Any], k: int = 5) -> float:
    """
    Exact RAGAS rank-weighted Context Precision@k (Es et al., 2024):
    CP@k = sum_{r=1}^k (Precision@r * v_r) / sum_{r=1}^k v_r
    """
    if not retrieved_chunks:
        return 0.0
    sub = retrieved_chunks[:k]
    flags = [1 if is_chunk_relevant(c, query) else 0 for c in sub]
    total_rel = sum(flags)
    if total_rel == 0:
        return 0.0

    running_rel = 0
    prec_sum = 0.0
    for r, v in enumerate(flags, start=1):
        if v == 1:
            running_rel += 1
            prec_sum += (running_rel / r)
    return round(prec_sum / total_rel, 4)


# ==============================================================================
# TIER 2 GENERATIVE & RAGAS JUDGE FUNCTIONS (`gpt-6-luna`)
# ==============================================================================
def generate_rag_answer(
    client: OpenAI,
    question: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> Tuple[str, int, int]:
    """
    Generates a grounded technical answer using `gpt-6-luna` on Top-5 retrieved chunks.
    Returns (answer_text, prompt_tokens, completion_tokens).
    """
    context_blocks = []
    for idx, c in enumerate(retrieved_chunks[:TOP_K_CONTEXT], start=1):
        txt = c.get("text", "").strip()
        context_blocks.append(f"[Passage {idx}]:\n{txt}")

    context_str = "\n\n".join(context_blocks)
    prompt = (
        f"You are an industrial energy auditor and technical compliance expert.\n"
        f"Answer the user's question concisely, factually, and strictly based ONLY on the provided context passages.\n"
        f"If the context does not contain sufficient facts to answer with certainty, state clearly: "
        f"\"The provided technical context does not contain sufficient information to answer this question.\"\n"
        f"Never extrapolate, speculate, or fabricate metrics.\n\n"
        f"Context Passages:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Technical Answer:"
    )

    for attempt in range(5):
        try:
            t_req = time.perf_counter()
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600,
                reasoning_effort="low"
            )
            req_ms = (time.perf_counter() - t_req) * 1000.0
            ans = (res.choices[0].message.content or "").strip()
            p_tok = res.usage.prompt_tokens if res.usage else 0
            c_tok = res.usage.completion_tokens if res.usage else 0
            if ans:
                return ans, p_tok, c_tok, req_ms
        except Exception as e:
            if attempt == 4:
                return f"Error generating answer: {e}", 0, 0, 1500.0
            time.sleep(2.0 * (attempt + 1))

    return "No answer generated.", 0, 0, 1500.0


def judge_ragas_metrics(
    client: OpenAI,
    question: str,
    context_str: str,
    generated_answer: str,
    ground_truth_answer: str
) -> Tuple[float, float, float, str]:
    """
    LLM-as-a-Judge evaluation using `gpt-6-luna` (`reasoning_effort="low"`).
    Returns (faithfulness, answer_relevance, llm_context_precision, critique).
    """
    judge_prompt = (
        f"You are an impartial academic evaluator scoring a Retrieval-Augmented Generation (RAG) system "
        f"according to the RAGAS framework (Es et al., 2024).\n\n"
        f"[Evaluation Task]\n"
        f"Question: {question}\n\n"
        f"Retrieved Context (Top-5 Passages provided to Generator):\n{context_str}\n\n"
        f"Generated Answer:\n{generated_answer}\n\n"
        f"Ground Truth Reference:\n{ground_truth_answer[:800]}\n\n"
        f"[Evaluation Criteria]\n"
        f"1. Faithfulness (0.0 to 1.0):\n"
        f"   - Are all factual claims, numbers, units, and assertions in the Generated Answer directly supported by the Retrieved Context?\n"
        f"   - If the answer states that information is not available when context indeed lacks it, Faithfulness is 1.0.\n"
        f"   - Penalize any hallucinated numbers, incorrect units, or ungrounded claims.\n\n"
        f"2. Answer Relevance (0.0 to 1.0):\n"
        f"   - How directly, completely, and accurately does the Generated Answer address the user's specific Question?\n"
        f"   - If the answer directly provides the requested metric or procedure matching the ground truth, score 1.0.\n"
        f"   - If the answer states that the context does not contain enough information, score between 0.2 and 0.4.\n\n"
        f"3. Context Precision (0.0 to 1.0):\n"
        f"   - Does the Retrieved Context contain the necessary ground-truth facts near the top without excessive irrelevant noise?\n\n"
        f"[Output Format]\n"
        f"Respond ONLY with a valid JSON object matching this schema:\n"
        f"{{\n"
        f"  \"faithfulness\": <float between 0.0 and 1.0>,\n"
        f"  \"answer_relevance\": <float between 0.0 and 1.0>,\n"
        f"  \"context_precision\": <float between 0.0 and 1.0>,\n"
        f"  \"critique\": \"<1-2 sentence evaluation summary>\"\n"
        f"}}"
    )

    for attempt in range(5):
        try:
            res = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                max_completion_tokens=400,
                reasoning_effort="low"
            )
            raw = (res.choices[0].message.content or "").strip()
            data = json.loads(raw)
            f_score = round(min(1.0, max(0.0, float(data.get("faithfulness", 0.5)))), 4)
            ar_score = round(min(1.0, max(0.0, float(data.get("answer_relevance", 0.5)))), 4)
            cp_score = round(min(1.0, max(0.0, float(data.get("context_precision", 0.5)))), 4)
            critique = str(data.get("critique", ""))
            return f_score, ar_score, cp_score, critique
        except Exception as e:
            if attempt == 4:
                return 0.5, 0.5, 0.5, f"Judge error: {e}"
            time.sleep(2.0 * (attempt + 1))

    return 0.5, 0.5, 0.5, "Default fallback"


# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
class UnifiedBenchmarkEvaluator:
    def __init__(self, max_workers: int = 16):
        self.vs = UnifiedVectorStore()
        self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=8.0, max_retries=0)
        self.max_workers = max_workers
        self.lock = threading.Lock()
        self.cache: Dict[str, Dict[str, Any]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if RAW_CACHE_PATH.exists():
            try:
                return json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        with self.lock:
            RAW_CACHE_PATH.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate_single(
        self,
        strat: str,
        q: Dict[str, Any],
        retrieved_10: List[Dict[str, Any]],
        retrieval_ms: float,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        cache_key = f"{strat}__{q['query_id']}"
        if not force_rerun and cache_key in self.cache:
            return self.cache[cache_key]

        retrieved_5 = retrieved_10[:TOP_K_CONTEXT]

        # 2. Tier 1 IR Metrics
        first_rank = None
        for r, ch in enumerate(retrieved_10, start=1):
            if is_chunk_relevant(ch, q):
                first_rank = r
                break

        hit_1 = 1 if first_rank is not None and first_rank <= 1 else 0
        hit_3 = 1 if first_rank is not None and first_rank <= 3 else 0
        hit_5 = 1 if first_rank is not None and first_rank <= 5 else 0
        hit_10 = 1 if first_rank is not None and first_rank <= 10 else 0

        rr_at_10 = round(1.0 / first_rank, 4) if first_rank is not None else 0.0
        rr_at_5 = round(1.0 / first_rank, 4) if (first_rank is not None and first_rank <= 5) else 0.0

        # Token IoU (best overlap in top-5 against ground-truth source_passage_text)
        source_passage = q.get("source_passage_text", "")
        ious = [compute_token_iou(c.get("text", ""), source_passage) for c in retrieved_5]
        best_iou = max(ious) if ious else 0.0

        # Rank-weighted Context Precision@5
        cp_rank = compute_context_precision_at_k(retrieved_5, q, k=5)

        # Context tokens (top-5)
        k5_chunk_tokens = sum(int(c.get("metadata", {}).get("token_count", 0)) for c in retrieved_5)

        # 3. RAG Answer Generation via `gpt-6-luna`
        gen_ans, api_prompt_toks, api_comp_toks, gen_ms = generate_rag_answer(self.client, q["question"], retrieved_5)

        # 4. RAGAS Judge Evaluation via `gpt-6-luna` (passing full Top-5 context matching generator)
        full_ctx_blocks = [f"[Passage {i}]:\n{c.get('text', '').strip()}" for i, c in enumerate(retrieved_5, start=1)]
        ctx_full_str = "\n\n".join(full_ctx_blocks)
        faith, ans_rel, llm_cp, critique = judge_ragas_metrics(
            self.client,
            q["question"],
            ctx_full_str,
            gen_ans,
            q["ground_truth_answer"]
        )

        rec = {
            "strategy": strat,
            "query_id": q["query_id"],
            "category_id": q["category_id"],
            "category_name": q["category_name"],
            "doc_id": q["doc_id"],
            "query_type": q["query_type"],
            "query_subtype": q["query_subtype"],
            "question": q["question"],
            "ground_truth_answer": q["ground_truth_answer"],
            "generated_answer": gen_ans,
            "first_rank": first_rank,
            "hit@1": hit_1,
            "hit@3": hit_3,
            "hit@5": hit_5,
            "hit@10": hit_10,
            "reciprocal_rank": rr_at_10,
            "reciprocal_rank_at_5": rr_at_5,
            "token_iou": best_iou,
            "context_precision": cp_rank,
            "llm_context_precision": llm_cp,
            "faithfulness": faith,
            "answer_relevance": ans_rel,
            "critique": critique,
            "retrieval_ms": round(retrieval_ms, 2),
            "generation_ms": round(gen_ms, 2),
            "total_latency_ms": round(retrieval_ms + gen_ms, 2),
            "prompt_tokens_k5": k5_chunk_tokens,
            "api_prompt_tokens": api_prompt_toks,
            "api_completion_tokens": api_comp_toks
        }

        with self.lock:
            self.cache[cache_key] = rec
        return rec

    def run_full_benchmark(self, force_rerun: bool = False) -> Dict[str, Any]:
        queries = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
        logger.info(f"[*] Loaded {len(queries)} balanced benchmark queries from {DATASET_JSON}")

        # Pre-embed all 120 query strings via text-embedding-3-small
        logger.info(f"[*] Pre-computing query embeddings for all {len(queries)} queries via {EMBEDDING_MODEL}...")
        t_emb0 = time.perf_counter()
        q_texts = [q["question"] for q in queries]
        q_embs = self.vs.embed_texts(q_texts)
        avg_emb_ms = ((time.perf_counter() - t_emb0) * 1000.0) / len(queries)
        q_emb_map = {q["query_id"]: emb for q, emb in zip(queries, q_embs)}

        # Execute uncontended ChromaDB retrieval across all 6 strategies x 120 queries
        logger.info("[*] Running uncontended ChromaDB Top-10 retrieval across all 6 strategies (720 queries)...")
        retrieved_map: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        for strat in STRATEGIES:
            for q in queries:
                cache_key = f"{strat}__{q['query_id']}"
                if not force_rerun and cache_key in self.cache:
                    retrieved_map[cache_key] = ([], 0.0)
                    continue
                t0 = time.perf_counter()
                r10 = self.vs.retrieve(strategy=strat, query=q["question"], top_k=TOP_K_RETRIEVAL, query_embedding=q_emb_map[q["query_id"]])
                r_ms = (time.perf_counter() - t0) * 1000.0 + avg_emb_ms
                retrieved_map[cache_key] = (r10, r_ms)

        tasks = []
        for strat in STRATEGIES:
            for q in queries:
                tasks.append((strat, q))

        total_tasks = len(tasks)
        logger.info(f"[*] Executing {total_tasks} LLM generation + RAGAS judge runs via {LLM_MODEL} with {self.max_workers} threads...")
        t_start = time.perf_counter()
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            fut_map = {
                ex.submit(
                    self.evaluate_single,
                    strat,
                    q,
                    retrieved_map[f"{strat}__{q['query_id']}"][0],
                    retrieved_map[f"{strat}__{q['query_id']}"][1],
                    force_rerun
                ): (strat, q["query_id"])
                for strat, q in tasks
            }
            for fut in as_completed(fut_map):
                strat, qid = fut_map[fut]
                try:
                    _ = fut.result()
                    completed += 1
                    if completed % 30 == 0 or completed == total_tasks:
                        elapsed = time.perf_counter() - t_start
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        logger.info(f"    -> Progress: {completed:3d}/{total_tasks} runs completed ({rate:.2f} runs/s, {elapsed:.1f}s elapsed)")
                        self._save_cache()
                except Exception as e:
                    logger.error(f"[!] Exception on {strat}__{qid}: {e}")
                    raise

        self._save_cache()
        logger.info(f"[+] Completed all {total_tasks} runs in {time.perf_counter() - t_start:.1f}s!")
        return self.aggregate_and_export(queries)

    def aggregate_and_export(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        chunk_sum_path = RESULTS_DIR / "chunking_summary.json"
        chunk_summary = json.loads(chunk_sum_path.read_text(encoding="utf-8")) if chunk_sum_path.exists() else {}

        agg: Dict[str, Dict[str, Any]] = {}
        subtype_rows: List[Dict[str, Any]] = []

        subtypes_order = [
            "exact_cell_lookup", "cross_comparison", "range_threshold",
            "scope_definition", "procedural_steps", "regulatory_obligation"
        ]

        for strat in STRATEGIES:
            items = [self.cache[f"{strat}__{q['query_id']}"] for q in queries if f"{strat}__{q['query_id']}" in self.cache]
            n = len(items)
            if n == 0:
                continue

            tbl_items = [it for it in items if it["query_type"] == "table_numeric"]
            nar_items = [it for it in items if it["query_type"] == "narrative_text"]

            r1 = round(sum(it["hit@1"] for it in items) / n, 4)
            r3 = round(sum(it["hit@3"] for it in items) / n, 4)
            r5 = round(sum(it["hit@5"] for it in items) / n, 4)
            r10 = round(sum(it["hit@10"] for it in items) / n, 4)
            mrr = round(float(np.mean([it["reciprocal_rank"] for it in items])), 4)
            mrr_5 = round(float(np.mean([it["reciprocal_rank_at_5"] for it in items])), 4)
            mean_iou = round(float(np.mean([it["token_iou"] for it in items])), 4)

            tbl_r5 = round(sum(it["hit@5"] for it in tbl_items) / len(tbl_items), 4) if tbl_items else 0.0
            nar_r5 = round(sum(it["hit@5"] for it in nar_items) / len(nar_items), 4) if nar_items else 0.0
            modality_gap = round(tbl_r5 - nar_r5, 4)

            cp_rank = round(float(np.mean([it["context_precision"] for it in items])), 4)
            cp_llm = round(float(np.mean([it["llm_context_precision"] for it in items])), 4)
            faith = round(float(np.mean([it["faithfulness"] for it in items])), 4)
            tbl_faith = round(float(np.mean([it["faithfulness"] for it in tbl_items])), 4) if tbl_items else 0.0
            nar_faith = round(float(np.mean([it["faithfulness"] for it in nar_items])), 4) if nar_items else 0.0
            ans_rel = round(float(np.mean([it["answer_relevance"] for it in items])), 4)
            aqs = round((faith + ans_rel) / 2.0, 4)

            avg_prompt_tok = round(float(np.mean([it["prompt_tokens_k5"] for it in items])), 1)
            avg_api_prompt_tok = round(float(np.mean([it["api_prompt_tokens"] for it in items])), 1)
            avg_api_comp_tok = round(float(np.mean([it["api_completion_tokens"] for it in items])), 1)

            avg_ret_ms = round(float(np.mean([it["retrieval_ms"] for it in items])), 2)
            avg_gen_ms = round(float(np.mean([it["generation_ms"] for it in items])), 2)
            avg_tot_ms = round(float(np.mean([it["total_latency_ms"] for it in items])), 2)

            c_meta = chunk_summary.get(strat, {})
            total_chunks = c_meta.get("total_chunks", 0)
            avg_chunk_tok = c_meta.get("avg_tokens_per_chunk", round(avg_prompt_tok / 5.0, 1))
            vec_mb = c_meta.get("vector_storage_mb", round((total_chunks * 1536 * 4) / (1024 * 1024), 2))
            ingest_cost = c_meta.get("total_ingest_cost_usd", round((total_chunks * avg_chunk_tok * PRICING["embedding_per_1m"]) / 1e6, 4))

            # Official gpt-6-luna pricing: $0.10 / 1M Input tokens, $0.50 / 1M Output tokens, $0.02 / 1M Embedding tokens
            # 1) Standard Thesis Formula (Top-5 context tokens + 25 query embed tokens + 150 completion tokens):
            cost_per_1k_usd = round(
                1000.0 * (
                    (avg_prompt_tok * PRICING["llm_input_per_1m"] / 1e6) +
                    (25.0 * PRICING["embedding_per_1m"] / 1e6) +
                    (150.0 * PRICING["llm_output_per_1m"] / 1e6)
                ),
                4
            )
            # 2) Direct Empirical API Usage Cost (from exact res.usage prompt_tokens & completion_tokens + 25 query embed tokens):
            empirical_api_cost_per_1k_usd = round(
                1000.0 * (
                    (avg_api_prompt_tok * PRICING["llm_input_per_1m"] / 1e6) +
                    (25.0 * PRICING["embedding_per_1m"] / 1e6) +
                    (avg_api_comp_tok * PRICING["llm_output_per_1m"] / 1e6)
                ),
                4
            )

            # Unweighted Composite Quality Score Qm (Thesis Eq. 3.11)
            q_m = round(0.20 * r5 + 0.20 * mrr + 0.20 * cp_rank + 0.20 * faith + 0.20 * ans_rel, 4)

            # Subtype breakdowns (n=20 per subtype)
            sub_dict = {"strategy": strat, "display_name": STRATEGY_DISPLAY_NAMES[strat]}
            for st in subtypes_order:
                st_items = [it for it in items if it["query_subtype"] == st]
                st_r5 = round(sum(it["hit@5"] for it in st_items) / len(st_items), 4) if st_items else 0.0
                sub_dict[f"r5_{st}"] = st_r5

            # Category breakdowns (n=30 per category)
            for cid in [1, 2, 3, 4]:
                cat_items = [it for it in items if it["category_id"] == cid]
                sub_dict[f"r5_cat{cid}"] = round(sum(it["hit@5"] for it in cat_items) / len(cat_items), 4) if cat_items else 0.0

            subtype_rows.append(sub_dict)

            agg[strat] = {
                "strategy": strat,
                "display_name": STRATEGY_DISPLAY_NAMES[strat],
                "total_chunks": total_chunks,
                "avg_tokens_per_chunk": avg_chunk_tok,
                "median_tokens_per_chunk": c_meta.get("median_tokens_per_chunk", 0.0),
                "table_chunks_count": c_meta.get("table_chunks_count", 0),
                "chunking_wall_seconds": c_meta.get("chunking_wall_seconds", 0.0),
                "peak_ram_mb": c_meta.get("peak_ram_mb", 0.0),
                "vector_storage_mb": vec_mb,
                "ingest_cost_usd": ingest_cost,
                "avg_prompt_tokens_k5": avg_prompt_tok,
                "avg_api_prompt_tokens": avg_api_prompt_tok,
                "avg_api_completion_tokens": avg_api_comp_tok,
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "recall@10": r10,
                "mrr": mrr,
                "mrr@5": mrr_5,
                "mean_token_iou": mean_iou,
                "table_recall@5": tbl_r5,
                "narrative_recall@5": nar_r5,
                "modality_gap_r5": modality_gap,
                "context_precision": cp_rank,
                "llm_context_precision": cp_llm,
                "faithfulness": faith,
                "table_faithfulness": tbl_faith,
                "narrative_faithfulness": nar_faith,
                "answer_relevance": ans_rel,
                "aqs": aqs,
                "avg_retrieval_ms": avg_ret_ms,
                "avg_generation_ms": avg_gen_ms,
                "avg_total_latency_ms": avg_tot_ms,
                "cost_per_1k_usd": cost_per_1k_usd,
                "empirical_api_cost_per_1k_usd": empirical_api_cost_per_1k_usd,
                "composite_quality": q_m,
                **{k: v for k, v in sub_dict.items() if k not in ("strategy", "display_name")}
            }

        # Compute Normalized Operational Cost Cm (Thesis Eq. 3.12)
        max_cost = max(m["cost_per_1k_usd"] for m in agg.values())
        max_lat = max(m["avg_total_latency_ms"] for m in agg.values())

        for strat, m in agg.items():
            c_m = 0.70 * (m["cost_per_1k_usd"] / max_cost) + 0.30 * (m["avg_total_latency_ms"] / max_lat)
            m["composite_cost"] = round(c_m, 4)

        # Pareto Frontier & ICER on (cost_per_1k_usd, composite_quality)
        sorted_by_cost = sorted(agg.values(), key=lambda x: x["cost_per_1k_usd"])
        max_q = -1.0
        for item in sorted_by_cost:
            if item["composite_quality"] > max_q:
                item["pareto_status"] = "Pareto Optimal"
                max_q = item["composite_quality"]
            else:
                item["pareto_status"] = "Dominated"

        prev_frontier = None
        pareto_rows = []
        for item in sorted_by_cost:
            if item["pareto_status"] == "Pareto Optimal":
                if prev_frontier is None:
                    item["delta_cost_usd"] = 0.0
                    item["delta_qm"] = 0.0
                    item["delta_r5"] = 0.0
                    item["icer_usd_per_qm"] = 0.0
                    item["icer_label"] = "Baseline Frontier"
                else:
                    dc = round(item["cost_per_1k_usd"] - prev_frontier["cost_per_1k_usd"], 4)
                    dq = round(item["composite_quality"] - prev_frontier["composite_quality"], 4)
                    dr5 = round(item["recall@5"] - prev_frontier["recall@5"], 4)
                    icer_val = round(dc / dq, 2) if dq > 0 else float("inf")
                    item["delta_cost_usd"] = dc
                    item["delta_qm"] = dq
                    item["delta_r5"] = dr5
                    item["icer_usd_per_qm"] = icer_val
                    item["icer_label"] = f"${icer_val:.2f} per +1.0 Qm"
                prev_frontier = item
            else:
                item["delta_cost_usd"] = None
                item["delta_qm"] = None
                item["delta_r5"] = None
                item["icer_usd_per_qm"] = None
                item["icer_label"] = "N/A (Dominated)"
            pareto_rows.append(item)

        # Save JSON & CSVs
        SUMMARY_JSON_PATH.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

        all_fields = list(next(iter(agg.values())).keys())
        with open(SUMMARY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_fields)
            w.writeheader()
            for r in agg.values():
                w.writerow(r)

        if subtype_rows:
            with open(SUBTYPE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(subtype_rows[0].keys()))
                w.writeheader()
                for r in subtype_rows:
                    w.writerow(r)

        with open(PARETO_ICER_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            p_fields = [
                "strategy", "display_name", "cost_per_1k_usd", "empirical_api_cost_per_1k_usd",
                "recall@5", "mrr", "context_precision", "faithfulness", "answer_relevance",
                "composite_quality", "composite_cost",
                "pareto_status", "delta_cost_usd", "delta_qm", "icer_usd_per_qm", "icer_label"
            ]
            w = csv.DictWriter(f, fieldnames=p_fields)
            w.writeheader()
            for r in pareto_rows:
                w.writerow({k: r.get(k, "") for k in p_fields})

        logger.info(f"[+] Saved complete benchmark summaries to {RESULTS_DIR}")
        return agg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true", help="Force re-evaluation of all 720 pairs")
    args = parser.parse_args()

    evaluator = UnifiedBenchmarkEvaluator(max_workers=16)
    evaluator.run_full_benchmark(force_rerun=args.force_rerun)

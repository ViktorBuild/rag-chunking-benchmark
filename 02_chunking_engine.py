# -*- coding: utf-8 -*-
"""
STAGE 2: Unified Multi-Strategy Document Chunking Engine
Executes all 6 chunking strategies across the 115 Markdown documents:
  1. fixed_128   (128 tokens, 20 overlap)
  2. fixed_512   (512 tokens, 77 overlap)
  3. fixed_1024  (1024 tokens, 205 overlap)
  4. recursive   (LangChain RecursiveCharacterTextSplitter, 512 tokens, 77 overlap)
  5. structural  (Markdown AST heading breadcrumbs + atomic tables, 512 target, 800 table ceiling)
  6. semantic    (Sentence-level embeddings via OpenAI `text-embedding-3-small`,
                  P90 cosine distance threshold, min_tokens=80, max_tokens=600, 0 overlap)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import re
import csv
import json
import time
import psutil
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
import numpy as np
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    SOURCE_MARKDOWN_DIR,
    CHUNKS_DIR,
    SENTENCE_CACHE_DIR,
    RESULTS_DIR,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    PRICING,
    STRATEGIES,
    STRATEGY_DISPLAY_NAMES,
    STRATEGY_PARAMS,
    setup_logger
)

logger = setup_logger("02_chunking_engine")
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text, disallowed_special=()))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    category_id: int
    strategy: str
    chunk_index: int
    text: str
    token_count: int
    char_count: int
    has_table: bool
    section_path: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_frontmatter(md_content: str) -> Tuple[Dict[str, Any], str]:
    meta = {}
    body = md_content
    if md_content.startswith("---"):
        parts = md_content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if val.isdigit():
                        meta[key] = int(val)
                    elif val.lower() == "true":
                        meta[key] = True
                    elif val.lower() == "false":
                        meta[key] = False
                    else:
                        meta[key] = val
    return meta, body


# ==============================================================================
# 1-3. FIXED TOKEN CHUNKING (128, 512, 1024)
# ==============================================================================
def chunk_fixed_tokens(
    text: str,
    doc_id: str,
    category_id: int,
    strategy_name: str,
    chunk_size: int,
    overlap: int,
    base_meta: Dict[str, Any]
) -> List[Chunk]:
    tokens = TOKENIZER.encode(text, disallowed_special=())
    if not tokens:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    chunk_idx = 0

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = TOKENIZER.decode(chunk_tokens)
        cid = f"{strategy_name}__{doc_id}__c{chunk_idx:04d}"
        has_table = "|---" in chunk_text or ("|" in chunk_text and "\n|" in chunk_text)

        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                category_id=category_id,
                strategy=strategy_name,
                chunk_index=chunk_idx,
                text=chunk_text,
                token_count=len(chunk_tokens),
                char_count=len(chunk_text),
                has_table=has_table,
                section_path="",
                metadata={
                    **base_meta,
                    "token_start": i,
                    "token_end": i + len(chunk_tokens)
                }
            )
        )
        chunk_idx += 1
        if i + chunk_size >= len(tokens):
            break

    return chunks


# ==============================================================================
# 4. RECURSIVE CHARACTER / TOKEN CHUNKING (512 tokens, 77 overlap)
# ==============================================================================
def chunk_recursive(
    text: str,
    doc_id: str,
    category_id: int,
    base_meta: Dict[str, Any],
    chunk_size: int = 512,
    chunk_overlap: int = 77
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_splits = splitter.split_text(text)

    chunks = []
    for idx, raw_text in enumerate(raw_splits):
        tok_cnt = count_tokens(raw_text)
        cid = f"recursive__{doc_id}__c{idx:04d}"
        has_table = "|---" in raw_text

        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                category_id=category_id,
                strategy="recursive",
                chunk_index=idx,
                text=raw_text,
                token_count=tok_cnt,
                char_count=len(raw_text),
                has_table=has_table,
                section_path="",
                metadata=base_meta
            )
        )
    return chunks


# ==============================================================================
# 5. STRUCTURAL MARKDOWN & TABLE-AWARE CHUNKING
# ==============================================================================
def chunk_structural(
    text: str,
    doc_id: str,
    category_id: int,
    base_meta: Dict[str, Any],
    target_chunk_size: int = 512,
    max_chunk_size: int = 800
) -> List[Chunk]:
    lines = text.split("\n")
    elements: List[Dict[str, Any]] = []

    current_headers = {}
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            htitle = heading_match.group(2).strip()
            for l in list(current_headers.keys()):
                if l >= level:
                    del current_headers[l]
            current_headers[level] = htitle
            i += 1
            continue

        path_crumbs = [current_headers[k] for k in sorted(current_headers.keys())]
        section_path = " > ".join(path_crumbs) if path_crumbs else "General"

        if stripped.startswith("|") and i + 1 < n and ("|---" in lines[i+1] or "| ---" in lines[i+1] or "|:" in lines[i+1]):
            table_lines = []
            while i < n and (lines[i].strip().startswith("|") or lines[i].strip() == ""):
                if lines[i].strip():
                    table_lines.append(lines[i].rstrip())
                i += 1
            table_md = "\n".join(table_lines)
            elements.append({
                "type": "table",
                "text": table_md,
                "section_path": section_path
            })
            continue

        if stripped:
            para_lines = [line]
            i += 1
            while i < n and lines[i].strip() and not lines[i].strip().startswith("#") and not lines[i].strip().startswith("|"):
                para_lines.append(lines[i])
                i += 1
            para_text = "\n".join(para_lines).strip()
            elements.append({
                "type": "prose",
                "text": para_text,
                "section_path": section_path
            })
        else:
            i += 1

    chunks: List[Chunk] = []
    chunk_idx = 0

    def add_chunk(chunk_content: str, s_path: str, is_tbl: bool):
        nonlocal chunk_idx
        if s_path and not chunk_content.startswith(f"### Section: {s_path}"):
            full_text = f"### Section: {s_path}\n\n{chunk_content}"
        else:
            full_text = chunk_content

        tok_cnt = count_tokens(full_text)
        cid = f"structural__{doc_id}__c{chunk_idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                category_id=category_id,
                strategy="structural",
                chunk_index=chunk_idx,
                text=full_text,
                token_count=tok_cnt,
                char_count=len(full_text),
                has_table=is_tbl,
                section_path=s_path,
                metadata=base_meta
            )
        )
        chunk_idx += 1

    current_acc: List[str] = []
    current_path = ""
    current_toks = 0

    for elem in elements:
        if elem["type"] == "table":
            if current_acc:
                add_chunk("\n\n".join(current_acc), current_path, is_tbl=False)
                current_acc = []
                current_toks = 0

            table_str = elem["text"]
            tbl_toks = count_tokens(table_str)

            if tbl_toks <= max_chunk_size:
                add_chunk(table_str, elem["section_path"], is_tbl=True)
            else:
                t_lines = table_str.split("\n")
                if len(t_lines) >= 3:
                    header_rows = t_lines[:2]
                    header_str = "\n".join(header_rows)
                    data_rows = t_lines[2:]

                    sub_batch = []
                    sub_toks = count_tokens(header_str)
                    for r in data_rows:
                        r_toks = count_tokens(r)
                        if sub_toks + r_toks > max_chunk_size and sub_batch:
                            batch_table = header_str + "\n" + "\n".join(sub_batch)
                            add_chunk(batch_table, elem["section_path"], is_tbl=True)
                            sub_batch = [r]
                            sub_toks = count_tokens(header_str) + r_toks
                        else:
                            sub_batch.append(r)
                            sub_toks += r_toks
                    if sub_batch:
                        batch_table = header_str + "\n" + "\n".join(sub_batch)
                        add_chunk(batch_table, elem["section_path"], is_tbl=True)
                else:
                    add_chunk(table_str, elem["section_path"], is_tbl=True)
        else:
            p_text = elem["text"]
            p_toks = count_tokens(p_text)
            if (elem["section_path"] != current_path and current_acc) or (current_toks + p_toks > target_chunk_size and current_acc):
                add_chunk("\n\n".join(current_acc), current_path, is_tbl=False)
                current_acc = []
                current_toks = 0

            current_path = elem["section_path"]
            current_acc.append(p_text)
            current_toks += p_toks

    if current_acc:
        add_chunk("\n\n".join(current_acc), current_path, is_tbl=False)

    return chunks


# ==============================================================================
# 6. UNIFIED SEMANTIC BREAKPOINT CHUNKING VIA OPENAI `text-embedding-3-small`
# ==============================================================================
def get_sentence_distances_openai(
    client: OpenAI,
    doc_id: str,
    sentences: List[str]
) -> Tuple[List[float], int, float]:
    """
    Computes consecutive sentence cosine distances using OpenAI `text-embedding-3-small`.
    Uses persistent disk cache in SENTENCE_CACHE_DIR so distances & token usage are reproducible.
    Returns (distances_list, total_embedding_tokens, api_elapsed_seconds).
    """
    cache_file = SENTENCE_CACHE_DIR / f"{doc_id}_openai_distances.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("sentence_count") == len(sentences) and cached.get("model") == EMBEDDING_MODEL:
                return cached["distances"], cached.get("total_tokens", 0), cached.get("api_seconds", 0.0)
        except Exception:
            pass

    t0 = time.perf_counter()
    # Prepare safe inputs for embedding (truncate any massive single table string > 4000 tokens to 4000 tokens)
    safe_inputs = []
    total_tokens = 0
    for s in sentences:
        toks = TOKENIZER.encode(s, disallowed_special=())
        if len(toks) > 4000:
            toks = toks[:4000]
            s_safe = TOKENIZER.decode(toks)
        else:
            s_safe = s
        total_tokens += len(toks)
        safe_inputs.append(s_safe)

    # Batch API requests safely (max 400 sentences or 120,000 tokens per batch)
    all_embeddings = []
    batch = []
    batch_tokens = 0

    def flush_batch(b: List[str]):
        for attempt in range(5):
            try:
                res = client.embeddings.create(input=b, model=EMBEDDING_MODEL)
                return [item.embedding for item in res.data]
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        return []

    for s_in in safe_inputs:
        s_tok = count_tokens(s_in)
        if batch and (len(batch) >= 400 or batch_tokens + s_tok > 120000):
            all_embeddings.extend(flush_batch(batch))
            batch = []
            batch_tokens = 0
        batch.append(s_in)
        batch_tokens += s_tok

    if batch:
        all_embeddings.extend(flush_batch(batch))

    api_seconds = time.perf_counter() - t0

    # OpenAI text-embedding-3-small vectors are unit-normalized; cosine distance = 1 - dot(e_i, e_{i+1})
    emb_arr = np.array(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_arr = emb_arr / norms

    distances = []
    for j in range(len(emb_arr) - 1):
        cos_sim = float(np.dot(emb_arr[j], emb_arr[j + 1]))
        distances.append(round(1.0 - cos_sim, 6))

    cache_payload = {
        "doc_id": doc_id,
        "model": EMBEDDING_MODEL,
        "sentence_count": len(sentences),
        "total_tokens": total_tokens,
        "api_seconds": round(api_seconds, 4),
        "distances": distances
    }
    cache_file.write_text(json.dumps(cache_payload), encoding="utf-8")
    return distances, total_tokens, api_seconds


def chunk_semantic_openai(
    client: OpenAI,
    text: str,
    doc_id: str,
    category_id: int,
    base_meta: Dict[str, Any],
    min_tokens: int = 80,
    max_tokens: int = 600,
    breakpoint_percentile: float = 90.0
) -> Tuple[List[Chunk], int, float]:
    """
    Identical algorithm to `pipeline/chunking.py:chunk_semantic`,
    replacing `all-MiniLM-L6-v2` with OpenAI `text-embedding-3-small`.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]

    if not sentences:
        return [], 0, 0.0
    if len(sentences) <= 2:
        tok_cnt = count_tokens(text)
        cid = f"semantic__{doc_id}__c0000"
        return [
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                category_id=category_id,
                strategy="semantic",
                chunk_index=0,
                text=text,
                token_count=tok_cnt,
                char_count=len(text),
                has_table="|---" in text,
                section_path="",
                metadata=base_meta
            )
        ], tok_cnt, 0.0

    distances, embed_tokens, api_sec = get_sentence_distances_openai(client, doc_id, sentences)

    if distances:
        threshold = float(np.percentile(distances, breakpoint_percentile))
    else:
        threshold = 0.5

    chunks: List[Chunk] = []
    curr_sentences: List[str] = [sentences[0]]
    curr_tokens = count_tokens(sentences[0])
    chunk_idx = 0

    for idx, dist in enumerate(distances):
        next_sentence = sentences[idx + 1]
        next_tok = count_tokens(next_sentence)

        is_semantic_breakpoint = dist >= threshold
        is_too_big = (curr_tokens + next_tok) > max_tokens
        has_min_size = curr_tokens >= min_tokens

        if (is_semantic_breakpoint and has_min_size) or is_too_big:
            chunk_text = " ".join(curr_sentences)
            cid = f"semantic__{doc_id}__c{chunk_idx:04d}"
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    category_id=category_id,
                    strategy="semantic",
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    token_count=curr_tokens,
                    char_count=len(chunk_text),
                    has_table="|---" in chunk_text,
                    section_path="",
                    metadata=base_meta
                )
            )
            chunk_idx += 1
            curr_sentences = [next_sentence]
            curr_tokens = next_tok
        else:
            curr_sentences.append(next_sentence)
            curr_tokens += next_tok

    if curr_sentences:
        chunk_text = " ".join(curr_sentences)
        cid = f"semantic__{doc_id}__c{chunk_idx:04d}"
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                category_id=category_id,
                strategy="semantic",
                chunk_index=chunk_idx,
                text=chunk_text,
                token_count=curr_tokens,
                char_count=len(chunk_text),
                has_table="|---" in chunk_text,
                section_path="",
                metadata=base_meta
            )
        )

    return chunks, embed_tokens, api_sec


def run_all_chunking() -> Dict[str, Any]:
    md_files = sorted(SOURCE_MARKDOWN_DIR.rglob("*.md"))
    logger.info(f"[*] Found {len(md_files)} Markdown documents in {SOURCE_MARKDOWN_DIR}")
    if len(md_files) != 115:
        raise RuntimeError(f"Expected 115 Markdown files, found {len(md_files)}")

    # Pre-load all 115 documents into memory so I/O doesn't skew strategy timing
    docs_loaded = []
    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(content)
        doc_id = str(meta.get("document_id", md_path.stem))
        category_id = int(meta.get("category_id", 1))
        docs_loaded.append((doc_id, category_id, meta, body))

    client = OpenAI(api_key=OPENAI_API_KEY)
    summary = {}

    for strat in STRATEGIES:
        strat_dir = CHUNKS_DIR / strat
        strat_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[*] Running chunking strategy: {strat} ({STRATEGY_DISPLAY_NAMES[strat]})...")

        tracemalloc.start()
        t0 = time.perf_counter()

        all_chunks: List[Chunk] = []
        semantic_embed_tokens = 0
        semantic_api_sec = 0.0

        if strat == "semantic":
            # Parallelize OpenAI sentence embedding requests across 8 workers for speed
            def _proc_sem(item):
                did, cid, meta, body = item
                ch_list, emb_toks, api_s = chunk_semantic_openai(client, body, did, cid, meta, 80, 600, 90.0)
                out_file = strat_dir / f"{did}.json"
                out_file.write_text(json.dumps([c.to_dict() for c in ch_list], ensure_ascii=False, indent=2), encoding="utf-8")
                return ch_list, emb_toks, api_s

            with ThreadPoolExecutor(max_workers=8) as ex:
                futures = [ex.submit(_proc_sem, item) for item in docs_loaded]
                done_cnt = 0
                for fut in as_completed(futures):
                    ch_list, emb_toks, api_s = fut.result()
                    all_chunks.extend(ch_list)
                    semantic_embed_tokens += emb_toks
                    semantic_api_sec += api_s
                    done_cnt += 1
                    if done_cnt % 25 == 0 or done_cnt == len(docs_loaded):
                        logger.info(f"    [semantic] Processed {done_cnt}/{len(docs_loaded)} documents...")
        else:
            for did, cid, meta, body in docs_loaded:
                if strat == "fixed_128":
                    ch_list = chunk_fixed_tokens(body, did, cid, "fixed_128", 128, 20, meta)
                elif strat == "fixed_512":
                    ch_list = chunk_fixed_tokens(body, did, cid, "fixed_512", 512, 77, meta)
                elif strat == "fixed_1024":
                    ch_list = chunk_fixed_tokens(body, did, cid, "fixed_1024", 1024, 205, meta)
                elif strat == "recursive":
                    ch_list = chunk_recursive(body, did, cid, meta, 512, 77)
                elif strat == "structural":
                    ch_list = chunk_structural(body, did, cid, meta, 512, 800)
                else:
                    raise ValueError(f"Unknown strategy: {strat}")

                all_chunks.extend(ch_list)
                out_file = strat_dir / f"{did}.json"
                out_file.write_text(json.dumps([c.to_dict() for c in ch_list], ensure_ascii=False, indent=2), encoding="utf-8")

        wall_seconds = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_ram_mb = round(peak_bytes / (1024 * 1024), 2)

        tok_counts = [c.token_count for c in all_chunks]
        total_chunks = len(all_chunks)
        total_tokens = int(sum(tok_counts))
        avg_tokens = round(float(np.mean(tok_counts)), 1) if tok_counts else 0.0
        median_tokens = round(float(np.median(tok_counts)), 1) if tok_counts else 0.0
        std_tokens = round(float(np.std(tok_counts)), 1) if tok_counts else 0.0
        table_chunks = sum(1 for c in all_chunks if c.has_table)

        # Embedding cost:
        # For all strategies, indexing chunks into ChromaDB embeds `total_tokens` via text-embedding-3-small.
        # For `semantic`, sentence-level breakpoint detection ALSO embeds `semantic_embed_tokens` via text-embedding-3-small.
        index_embed_cost_usd = round((total_tokens / 1_000_000.0) * PRICING["embedding_per_1m"], 4)
        chunking_embed_cost_usd = round((semantic_embed_tokens / 1_000_000.0) * PRICING["embedding_per_1m"], 4) if strat == "semantic" else 0.0
        total_ingest_cost_usd = round(index_embed_cost_usd + chunking_embed_cost_usd, 4)

        summary[strat] = {
            "strategy": strat,
            "display_name": STRATEGY_DISPLAY_NAMES[strat],
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": avg_tokens,
            "median_tokens_per_chunk": median_tokens,
            "std_tokens_per_chunk": std_tokens,
            "table_chunks_count": table_chunks,
            "chunking_wall_seconds": round(wall_seconds, 2),
            "semantic_api_cumulative_seconds": round(semantic_api_sec, 2),
            "peak_ram_mb": peak_ram_mb,
            "semantic_sentence_embed_tokens": semantic_embed_tokens,
            "chunking_embed_cost_usd": chunking_embed_cost_usd,
            "index_embed_cost_usd": index_embed_cost_usd,
            "total_ingest_cost_usd": total_ingest_cost_usd,
            "vector_storage_mb": round((total_chunks * 1536 * 4) / (1024 * 1024), 2)
        }

        logger.info(
            f"[+] {STRATEGY_DISPLAY_NAMES[strat]:12s} | Chunks: {total_chunks:6,d} | "
            f"Avg Tok: {avg_tokens:6.1f} | Med Tok: {median_tokens:6.1f} | "
            f"Time: {wall_seconds:6.2f}s | Peak RAM: {peak_ram_mb:6.2f} MB | "
            f"Ingest Cost: ${total_ingest_cost_usd:.4f}"
        )

    sum_json = RESULTS_DIR / "chunking_summary.json"
    sum_csv = RESULTS_DIR / "chunking_summary.csv"
    sum_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fields = list(next(iter(summary.values())).keys())
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary.values():
            writer.writerow(row)

    logger.info(f"[+] Stage 2 Complete! Chunking summary saved to {sum_json} and {sum_csv}")
    return summary


if __name__ == "__main__":
    run_all_chunking()

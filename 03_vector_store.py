# -*- coding: utf-8 -*-
"""
STAGE 3: Isolated ChromaDB Vector Store Manager (`text-embedding-3-small`, 1536-d, cosine)
1. Syncs the 5 deterministic collections (`fixed_128`, `fixed_512`, `fixed_1024`, `recursive`, `structural`)
   into the isolated `data/chroma_db` directory and verifies their exact chunk counts & metadata.
2. Drops the legacy `strat_semantic` collection and indexes the newly generated unified `semantic` chunks
   (segmented using `text-embedding-3-small`) into `strat_semantic` using `text-embedding-3-small`.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
import chromadb
from openai import OpenAI

from config import (
    SOURCE_CHROMA_DIR,
    CHROMADB_DIR,
    CHUNKS_DIR,
    RESULTS_DIR,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    STRATEGIES,
    COLLECTION_NAMES,
    setup_logger
)

logger = setup_logger("03_vector_store")
TOKENIZER = tiktoken.get_encoding("cl100k_base")


class UnifiedVectorStore:
    def __init__(self, persist_dir: Path = CHROMADB_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=4)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._query_embed_cache: Dict[str, List[float]] = {}

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        safe_texts = []
        for t in texts:
            t_clean = t.replace("\x00", " ").strip()
            if not t_clean:
                t_clean = "empty"
            toks = TOKENIZER.encode(t_clean, disallowed_special=())
            if len(toks) > 6000:
                t_clean = TOKENIZER.decode(toks[:6000])
            safe_texts.append(t_clean)

        for attempt in range(5):
            try:
                res = self.openai_client.embeddings.create(input=safe_texts, model=EMBEDDING_MODEL)
                return [item.embedding for item in res.data]
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2.0 * (attempt + 1))
        return []

    def embed_query(self, query_text: str) -> List[float]:
        if query_text in self._query_embed_cache:
            return self._query_embed_cache[query_text]
        emb = self.embed_texts([query_text])[0]
        self._query_embed_cache[query_text] = emb
        return emb

    def retrieve(
        self,
        strategy: str,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        cname = COLLECTION_NAMES[strategy]
        col = self.client.get_collection(name=cname)
        q_emb = query_embedding if query_embedding is not None else self.embed_query(query)

        res = col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        out = []
        if res and res.get("ids") and res["ids"][0]:
            ids = res["ids"][0]
            docs = res["documents"][0] if res.get("documents") else [""] * len(ids)
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)

            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                out.append({
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta or {},
                    "distance": round(float(dist), 5),
                    "similarity": round(float(1.0 - dist), 5)
                })
        return out

    def get_all_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for strat, cname in COLLECTION_NAMES.items():
            try:
                col = self.client.get_collection(name=cname)
                stats[strat] = {
                    "collection": cname,
                    "count": col.count(),
                    "metadata": col.metadata
                }
            except Exception as e:
                stats[strat] = {"collection": cname, "count": 0, "error": str(e)}
        return stats


def build_and_verify_vector_store(force_reindex_semantic: bool = True) -> Dict[str, Any]:
    target_sqlite = CHROMADB_DIR / "chroma.sqlite3"
    if not target_sqlite.exists() and SOURCE_CHROMA_DIR.exists():
        logger.info(f"[*] Copying base ChromaDB store from {SOURCE_CHROMA_DIR} to isolated {CHROMADB_DIR}...")
        for item in SOURCE_CHROMA_DIR.iterdir():
            dest = CHROMADB_DIR / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        logger.info("[+] Base ChromaDB store copied successfully.")

    vs = UnifiedVectorStore(CHROMADB_DIR)

    # Load chunking summary to verify exact counts
    chunk_sum_path = RESULTS_DIR / "chunking_summary.json"
    chunk_summary = json.loads(chunk_sum_path.read_text(encoding="utf-8")) if chunk_sum_path.exists() else {}

    for strat in STRATEGIES:
        cname = COLLECTION_NAMES[strat]
        strat_chunk_dir = CHUNKS_DIR / strat
        chunk_files = sorted(strat_chunk_dir.glob("*.json"))

        all_chunks = []
        for cf in chunk_files:
            all_chunks.extend(json.loads(cf.read_text(encoding="utf-8")))
        expected_cnt = len(all_chunks)

        need_index = False
        if strat == "semantic" and force_reindex_semantic:
            # Check if strat_semantic already matches the new chunk count and has unified_openai flag
            try:
                col = vs.client.get_collection(name=cname)
                meta = col.metadata or {}
                if col.count() == expected_cnt and meta.get("semantic_encoder") == EMBEDDING_MODEL:
                    logger.info(f"[*] Collection {cname} already indexed with unified {EMBEDDING_MODEL} ({expected_cnt:,} vectors).")
                else:
                    logger.info(f"[*] Dropping legacy {cname} collection (had {col.count():,} vectors) to re-index {expected_cnt:,} unified semantic chunks...")
                    vs.client.delete_collection(name=cname)
                    need_index = True
            except Exception:
                need_index = True
        else:
            try:
                col = vs.client.get_collection(name=cname)
                if col.count() != expected_cnt:
                    logger.info(f"[*] Collection {cname} count mismatch ({col.count()} vs {expected_cnt}); re-indexing...")
                    vs.client.delete_collection(name=cname)
                    need_index = True
                else:
                    logger.info(f"[+] Verified deterministic collection {cname}: {col.count():,} vectors ({EMBEDDING_MODEL}).")
            except Exception:
                need_index = True

        if need_index:
            logger.info(f"[*] Indexing {expected_cnt:,} chunks into {cname} via {EMBEDDING_MODEL}...")
            t0 = time.perf_counter()
            col = vs.client.get_or_create_collection(
                name=cname,
                metadata={
                    "hnsw:space": "cosine",
                    "model": EMBEDDING_MODEL,
                    "semantic_encoder": EMBEDDING_MODEL,
                    "dimension": EMBEDDING_DIM
                }
            )

            # Prepare batches of 250 chunks
            batch_size = 250
            batches = []
            for i in range(0, len(all_chunks), batch_size):
                batches.append((i, all_chunks[i : i + batch_size]))

            # Parallelize OpenAI embedding generation across 8 threads
            embedded_batches = {}
            with ThreadPoolExecutor(max_workers=8) as ex:
                fut_map = {
                    ex.submit(vs.embed_texts, [c["text"] for c in b_chunks]): (b_idx, b_chunks)
                    for b_idx, b_chunks in batches
                }
                done_b = 0
                for fut in as_completed(fut_map):
                    b_idx, b_chunks = fut_map[fut]
                    embs = fut.result()
                    embedded_batches[b_idx] = (b_chunks, embs)
                    done_b += 1
                    if done_b % 10 == 0 or done_b == len(batches):
                        logger.info(f"    [{cname}] Embedded {done_b}/{len(batches)} batches...")

            # Insert into ChromaDB sequentially (thread-safe SQLite/HNSW writes)
            for b_idx in sorted(embedded_batches.keys()):
                b_chunks, embs = embedded_batches[b_idx]
                ids = [c["chunk_id"] for c in b_chunks]
                docs = [c["text"] for c in b_chunks]
                metas = []
                for c in b_chunks:
                    metas.append({
                        "doc_id": str(c.get("doc_id", "")),
                        "category_id": int(c.get("category_id", 1)),
                        "strategy": str(c.get("strategy", strat)),
                        "chunk_index": int(c.get("chunk_index", 0)),
                        "token_count": int(c.get("token_count", 0)),
                        "char_count": int(c.get("char_count", 0)),
                        "has_table": bool(c.get("has_table", False)),
                        "section_path": str(c.get("section_path", ""))[:500]
                    })
                col.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)

            elapsed_idx = time.perf_counter() - t0
            logger.info(f"[+] Indexed {col.count():,} vectors into {cname} in {elapsed_idx:.2f}s!")

    stats = vs.get_all_collection_stats()
    out_path = RESULTS_DIR / "vector_store_summary.json"
    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(f"[+] Stage 3 Complete! Vector store summary saved to {out_path}")
    return stats


if __name__ == "__main__":
    build_and_verify_vector_store(force_reindex_semantic=True)

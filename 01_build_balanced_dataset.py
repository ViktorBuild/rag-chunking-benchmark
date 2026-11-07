# -*- coding: utf-8 -*-
"""
STAGE 1: Strictly Balanced 120-Query Benchmark Generator (60 Table / 60 Narrative)
Fixes the Category 1 narrative candidate cap from the legacy generator,
replaces the 4 fallback `exact_cell_lookup` items in Category 1 with 4 genuine
`regulatory_obligation` items from Category 1 documents refined by `gpt-6-luna`,
and verifies the exact 4 x 6 x 5 = 120 stratification matrix.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import csv
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple
from openai import OpenAI

from config import (
    SOURCE_MARKDOWN_DIR,
    SOURCE_EVAL_JSON,
    DATASET_JSON,
    DATASET_CSV,
    OPENAI_API_KEY,
    LLM_MODEL,
    RANDOM_SEED,
    CATEGORIES,
    setup_logger
)

logger = setup_logger("01_build_balanced_dataset")


def parse_frontmatter(md_text: str) -> Tuple[Dict[str, str], str]:
    meta = {}
    if md_text.startswith("---"):
        parts = md_text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            return meta, parts[2].strip()
    return meta, md_text


def extract_key_facts(text: str, max_facts: int = 3) -> List[str]:
    facts = []
    num_matches = re.findall(
        r'\b\d+(?:[.,]\d+)?(?:\s*(?:%|MW|GW|kW|kWh|MWh|GWh|GJ|MJ|tCO2|CO2|m/s|m|mm|km|bar|Pa|°C|K|years?|hours?|days?|USD|EUR|\$|€))?\b',
        text
    )
    cleaned_nums = [n.strip() for n in num_matches if len(n.strip()) >= 2 and not n.strip().startswith("202")]
    for n in cleaned_nums:
        if n not in facts:
            facts.append(n)
            if len(facts) >= 2:
                break

    words = re.findall(r'\b[A-Za-z][A-Za-z0-9_-]{4,}\b', text)
    stop = {
        "which", "their", "there", "these", "those", "where", "while", "should", "would", "could",
        "about", "above", "after", "again", "against", "below", "between", "during", "before",
        "under", "further", "then", "once", "here", "when", "both", "each", "more", "most",
        "other", "some", "such", "only", "same", "than", "very", "will", "with", "within",
        "without", "through", "table", "figure", "section", "chapter", "report", "document",
        "energy", "system", "systems", "using", "based", "total", "value", "values"
    }
    for w in words:
        wl = w.lower()
        if wl not in stop and wl not in [f.lower() for f in facts]:
            facts.append(w)
            if len(facts) >= max_facts:
                break
    return facts[:max_facts]


def extract_cat1_regulatory_candidates() -> List[Dict[str, Any]]:
    """
    Scans all Category 1 Markdown documents without the legacy `prose_index < 4` cap
    to find genuine regulatory/compliance/mandatory obligation passages.
    """
    cat1_dir = SOURCE_MARKDOWN_DIR / CATEGORIES[1]["name"]
    md_files = sorted(cat1_dir.glob("*.md"))
    candidates = []

    reg_keywords = [
        "must", "shall", "required", "requirement", "compliance",
        "mandatory", "obligation", "standard", "certification", "minimum", "accordance"
    ]

    for md_path in md_files:
        content = md_path.read_text(encoding="utf-8", errors="replace")
        meta, body = parse_frontmatter(content)
        doc_id = meta.get("document_id", md_path.stem)
        doc_title = meta.get("title", doc_id.replace("_", " ").title())

        lines = body.split("\n")
        current_sec = "General"
        buf = []
        reg_idx = 0

        for line in lines:
            s = line.strip()
            if s.startswith("#"):
                current_sec = re.sub(r'^#+\s*', '', s).strip()
                continue
            if s.startswith("|"):
                continue
            if not s:
                if buf:
                    para = " ".join(buf)
                    buf = []
                    if 280 <= len(para) <= 1200:
                        pl = para.lower()
                        # Check for strong regulatory/compliance obligations and numeric/technical specificity
                        kw_hits = sum(1 for kw in reg_keywords if kw in pl)
                        has_num = bool(re.search(r'\b\d+\b', para))
                        if kw_hits >= 2 and has_num:
                            sents = [st.strip() for st in re.split(r'(?<=[.!?])\s+', para) if len(st.strip()) > 30]
                            if len(sents) >= 2:
                                reg_idx += 1
                                facts = extract_key_facts(para, max_facts=3)
                                gt_answer = " ".join(sents[:2])
                                base_q = (
                                    f"According to {doc_title} (Section '{current_sec}'), what specific compliance "
                                    f"or technical requirements are stipulated regarding {facts[0] if facts else 'the system'}?"
                                )
                                candidates.append({
                                    "query_id": f"eval_{doc_id}_reg_{reg_idx:02d}",
                                    "category_id": 1,
                                    "category_name": CATEGORIES[1]["name"],
                                    "doc_id": doc_id,
                                    "doc_title": doc_title,
                                    "query_type": "narrative_text",
                                    "query_subtype": "regulatory_obligation",
                                    "question": base_q,
                                    "ground_truth_answer": gt_answer,
                                    "source_passage_text": para[:900],
                                    "key_facts": facts if facts else [doc_id.split("_")[0]],
                                    "source_section": current_sec
                                })
            else:
                buf.append(s)

    return candidates


def refine_question_with_llm(client: OpenAI, item: Dict[str, Any]) -> str:
    prompt = (
        f"You are creating an evaluation benchmark for a Technical & Regulatory RAG system.\n"
        f"Rewrite the following template question into a natural, concise, unambiguous expert question "
        f"(1 sentence, max 28 words) that asks for the exact facts in the Ground Truth Answer.\n"
        f"IMPORTANT: Keep specific technical terms, units, or context identifiers so the question is self-contained.\n\n"
        f"Document Title: {item['doc_title']}\n"
        f"Section: {item['source_section']}\n"
        f"Source Passage: {item['source_passage_text'][:450]}\n"
        f"Ground Truth Answer: {item['ground_truth_answer']}\n"
        f"Template Question: {item['question']}\n\n"
        f"Output ONLY the rewritten question string without quotes:"
    )
    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=100,
        reasoning_effort="low"
    )
    new_q = (res.choices[0].message.content or "").strip().strip('"').strip("'")
    if len(new_q) > 15 and "?" in new_q:
        return new_q
    return item["question"]


def build_balanced_dataset() -> List[Dict[str, Any]]:
    logger.info(f"[*] Loading base evaluation dataset from {SOURCE_EVAL_JSON}...")
    base_data = json.loads(SOURCE_EVAL_JSON.read_text(encoding="utf-8"))

    # Keep all items except the 4 excess fallback `exact_cell_lookup` items in Category 1
    kept_items = []
    cat1_exact_count = 0
    removed_ids = []

    for item in base_data:
        if item["category_id"] == 1 and item["query_subtype"] == "exact_cell_lookup":
            cat1_exact_count += 1
            if cat1_exact_count > 5:
                removed_ids.append(item["query_id"])
                continue
        kept_items.append(item)

    logger.info(f"[*] Kept {len(kept_items)} verified items; removed {len(removed_ids)} excess Cat-1 fallback items: {removed_ids}")

    # Find genuine Category 1 regulatory_obligation candidates
    existing_ids = {it["query_id"] for it in kept_items}
    existing_passages = {it["source_passage_text"][:100] for it in kept_items}

    all_reg_candidates = extract_cat1_regulatory_candidates()
    valid_reg_candidates = [
        c for c in all_reg_candidates
        if c["query_id"] not in existing_ids and c["source_passage_text"][:100] not in existing_passages
    ]

    logger.info(f"[*] Found {len(valid_reg_candidates)} distinct Category 1 regulatory_obligation candidates.")
    if len(valid_reg_candidates) < 4:
        raise RuntimeError(f"Insufficient Category 1 regulatory_obligation candidates found ({len(valid_reg_candidates)} < 4)!")

    # Deterministically select 4 candidates across distinct documents where possible
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(valid_reg_candidates)

    selected_new = []
    used_docs = set()
    for c in valid_reg_candidates:
        if c["doc_id"] not in used_docs:
            selected_new.append(c)
            used_docs.add(c["doc_id"])
            if len(selected_new) == 4:
                break
    for c in valid_reg_candidates:
        if len(selected_new) == 4:
            break
        if c not in selected_new:
            selected_new.append(c)

    # Refine the 4 new questions using gpt-6-luna
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info(f"[*] Refining 4 new Category 1 regulatory_obligation questions via {LLM_MODEL}...")
    for item in selected_new:
        refined_q = refine_question_with_llm(client, item)
        item["question"] = refined_q
        logger.info(f"    + [{item['query_id']}] ({item['doc_id']}): {item['question']}")

    # Insert the 4 new items into Category 1 right after existing Category 1 items
    final_dataset = []
    cat1_inserted = False
    for item in kept_items:
        if item["category_id"] != 1 and not cat1_inserted:
            final_dataset.extend(selected_new)
            cat1_inserted = True
        final_dataset.append(item)
    if not cat1_inserted:
        final_dataset.extend(selected_new)

    # Strict validation of 60/60 balance and 4 x 6 x 5 stratification
    assert len(final_dataset) == 120, f"Expected 120 queries, got {len(final_dataset)}"
    type_counts = Counter(q["query_type"] for q in final_dataset)
    assert type_counts["table_numeric"] == 60, f"Expected 60 table_numeric, got {type_counts['table_numeric']}"
    assert type_counts["narrative_text"] == 60, f"Expected 60 narrative_text, got {type_counts['narrative_text']}"

    cell_counts = Counter((q["category_id"], q["query_subtype"]) for q in final_dataset)
    for cid in [1, 2, 3, 4]:
        for st in ["exact_cell_lookup", "cross_comparison", "range_threshold", "scope_definition", "procedural_steps", "regulatory_obligation"]:
            cnt = cell_counts[(cid, st)]
            assert cnt == 5, f"Stratification violation at Cat {cid} / {st}: expected 5, got {cnt}"

    # Save JSON and CSV
    DATASET_JSON.write_text(json.dumps(final_dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = ["query_id", "category_id", "category_name", "doc_id", "doc_title", "query_type", "query_subtype", "question", "ground_truth_answer", "source_section"]
    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for q in final_dataset:
            writer.writerow({k: q.get(k, "") for k in csv_fields})

    logger.info(f"[+] Successfully built & verified strictly balanced 120-query dataset!")
    logger.info(f"    - Saved JSON: {DATASET_JSON}")
    logger.info(f"    - Saved CSV:  {DATASET_CSV}")
    logger.info(f"    - Distribution: {dict(type_counts)}")
    return final_dataset


if __name__ == "__main__":
    build_balanced_dataset()

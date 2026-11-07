# -*- coding: utf-8 -*-
"""
Unified & Reproducible RAG Chunking Benchmark Configuration
THWS Bachelor Thesis: "Chunking Strategy Comparison for RAG Systems"

Key Methodological Guarantees:
1. Unified Embedding Model: OpenAI `text-embedding-3-small` (1536-d, cosine)
   used for BOTH Semantic Chunking (sentence-level breakpoint detection)
   AND Vector Store Indexing / Retrieval across all 6 strategies.
2. Unified LLM Generator & RAGAS Judge: OpenAI `gpt-6-luna` (reasoning_effort="low").
3. No Cross-Encoder Re-ranker: Pure bi-encoder vector retrieval via ChromaDB HNSW (cosine).
4. Strictly Balanced 120-Query Dataset: 60 Table-numeric + 60 Narrative-text queries
   (4 categories x 6 subtypes x 5 queries = 120 queries).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_WORKSPACE = EXPERIMENT_DIR.parent

# Load .env with override=True so system env vars do not shadow workspace .env
load_dotenv(ROOT_WORKSPACE / ".env", override=True)

# Source read-only corpus directory (115 converted Markdown files)
SOURCE_MARKDOWN_DIR = (EXPERIMENT_DIR / "data" / "processed_markdown") if (EXPERIMENT_DIR / "data" / "processed_markdown").exists() else (ROOT_WORKSPACE / "processed_markdown")
SOURCE_EVAL_JSON = ROOT_WORKSPACE / "evaluation_dataset.json"
SOURCE_CHROMA_DIR = ROOT_WORKSPACE / "chroma_db"

# Isolated Experiment Data & Output Directories
DATA_DIR = EXPERIMENT_DIR / "data"
CHUNKS_DIR = DATA_DIR / "processed_chunks"
SENTENCE_CACHE_DIR = DATA_DIR / "sentence_embeddings_cache"
CHROMADB_DIR = DATA_DIR / "chroma_db"
DATASET_JSON = DATA_DIR / "evaluation_dataset_120_balanced.json"
DATASET_CSV = DATA_DIR / "evaluation_dataset_120_balanced.csv"

RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
LOG_FILE = EXPERIMENT_DIR / "pipeline_execution.log"

for d in [DATA_DIR, CHUNKS_DIR, SENTENCE_CACHE_DIR, CHROMADB_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API & Model Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-6-luna").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
EMBEDDING_DIM = 1536
RANDOM_SEED = 42
TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 5

# Official Pricing Constants (USD per 1,000,000 tokens)
# Model: gpt-6-luna ($0.10 / 1M Input tokens, $0.50 / 1M Output tokens)
# Embedding: text-embedding-3-small ($0.02 / 1M tokens)
PRICING = {
    "embedding_per_1m": 0.02,   # text-embedding-3-small ($0.02 / MTok)
    "llm_input_per_1m": 0.10,   # gpt-6-luna ($0.10 / Input MTok)
    "llm_output_per_1m": 0.50,  # gpt-6-luna ($0.50 / Output MTok)
}

CATEGORIES = {
    1: {"name": "01_engineering_optimization_guides", "title": "Engineering & Technical Optimization Guides"},
    2: {"name": "02_decarbonization_roadmaps_iea_irena", "title": "Global Decarbonization & Energy Transition Roadmaps"},
    3: {"name": "03_standards_energy_audits_green_building", "title": "Energy Audits, Management & Green Building Standards"},
    4: {"name": "04_carbon_accounting_esg_methodologies", "title": "Carbon Accounting, ESG & GHG Methodologies"},
}

STRATEGIES = ["fixed_128", "fixed_512", "fixed_1024", "recursive", "structural", "semantic"]

STRATEGY_DISPLAY_NAMES = {
    "fixed_128": "Fixed-128",
    "fixed_512": "Fixed-512",
    "fixed_1024": "Fixed-1024",
    "recursive": "Recursive",
    "structural": "Structural",
    "semantic": "Semantic"
}

COLLECTION_NAMES = {
    "fixed_128": "strat_fixed_128",
    "fixed_512": "strat_fixed_512",
    "fixed_1024": "strat_fixed_1024",
    "recursive": "strat_recursive",
    "structural": "strat_structural",
    "semantic": "strat_semantic"
}

STRATEGY_PARAMS = {
    "fixed_128": {
        "window_tokens": 128,
        "overlap_tokens": 20,
        "overlap_pct": "15%",
        "description": "Fixed token window (128 tokens, 20 tokens / ~15% overlap)"
    },
    "fixed_512": {
        "window_tokens": 512,
        "overlap_tokens": 77,
        "overlap_pct": "15%",
        "description": "Fixed token window (512 tokens, 77 tokens / 15% overlap)"
    },
    "fixed_1024": {
        "window_tokens": 1024,
        "overlap_tokens": 205,
        "overlap_pct": "20%",
        "description": "Fixed token window (1024 tokens, 205 tokens / 20% overlap)"
    },
    "recursive": {
        "target_tokens": 512,
        "overlap_tokens": 77,
        "overlap_pct": "15%",
        "delimiters": ["\\n\\n", "\\n", ". ", " "],
        "description": "Hierarchical syntactic splitting (target 512 tokens, 77 tokens / 15% overlap)"
    },
    "structural": {
        "target_tokens": 512,
        "max_table_tokens": 800,
        "overlap_tokens": 0,
        "overlap_pct": "0%",
        "description": "Markdown AST heading & atomic table segmentation (target 512, table ceiling 800 tokens, 0% overlap)"
    },
    "semantic": {
        "min_tokens": 80,
        "max_tokens": 600,
        "breakpoint_percentile": 90.0,
        "overlap_tokens": 0,
        "overlap_pct": "0%",
        "embedding_model": EMBEDDING_MODEL,
        "description": "Sentence-level semantic breakpoint chunking via OpenAI text-embedding-3-small (P90 cosine distance, 80-600 tokens, 0% overlap)"
    }
}

def setup_logger(name: str = "unified_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

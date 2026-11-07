# -*- coding: utf-8 -*-
"""
Master Reproducible Orchestrator for the Unified RAG Chunking Benchmark Pipeline.
Executes Stages 1 to 5 sequentially:
  - Stage 1: Build & Verify Strictly Balanced 120-Query Dataset (60 Table / 60 Narrative)
  - Stage 2: Execute All 6 Chunking Strategies Across 115 Docs (with `text-embedding-3-small` for Semantic)
  - Stage 3: Build & Index Isolated ChromaDB Vector Store (`text-embedding-3-small`, 1536-d, cosine)
  - Stage 4: Execute Full 720-Run Retrieval & RAGAS Evaluation (`gpt-6-luna`)
  - Stage 5: Generate High-Resolution Charts & Comprehensive Markdown Report
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from importlib import import_module
from config import setup_logger

logger = setup_logger("run_all")

if __name__ == "__main__":
    logger.info("==================================================================")
    logger.info("   STARTING UNIFIED REPRODUCIBLE RAG CHUNKING BENCHMARK PIPELINE")
    logger.info("==================================================================")

    s1 = import_module("01_build_balanced_dataset")
    s1.build_balanced_dataset()

    s2 = import_module("02_chunking_engine")
    s2.run_all_chunking()

    s3 = import_module("03_vector_store")
    s3.build_and_verify_vector_store(force_reindex_semantic=True)

    s4 = import_module("04_run_benchmark_and_ragas")
    evaluator = s4.UnifiedBenchmarkEvaluator(max_workers=16)
    evaluator.run_full_benchmark(force_rerun=False)

    s5 = import_module("05_generate_charts_and_report")
    s5.generate_all_charts_and_report()

    logger.info("==================================================================")
    logger.info("   ALL 5 PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    logger.info("==================================================================")

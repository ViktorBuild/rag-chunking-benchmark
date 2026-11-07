# -*- coding: utf-8 -*-
"""
STAGE 5: Publication-Grade Chart Generator & Transparent Markdown Report Builder
Uses official `gpt-6-luna` pricing ($0.10 / Input MTok, $0.50 / Output MTok)
and Full Top-5 Context RAGAS evaluation.
Generates all figures directly in `figures/` and writes `results/UNIFIED_BENCHMARK_REPORT.md`.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import (
    RESULTS_DIR,
    FIGURES_DIR,
    STRATEGIES,
    STRATEGY_DISPLAY_NAMES,
    LLM_MODEL,
    EMBEDDING_MODEL,
    PRICING,
    setup_logger
)

logger = setup_logger("05_generate_charts_and_report")


def draw_decision_tree_figure(data: Dict[str, Any], out_path: Path):
    """Generates Figure 5.1 (Engineering Decision Tree) with updated empirical numbers."""
    fig, ax = plt.subplots(figsize=(12.5, 9.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle_lines, edge_color, fill_color, title_color, badge_text=None, badge_color=None):
        box = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.3,rounding_size=2.0",
            linewidth=2.2, edgecolor=edge_color, facecolor=fill_color, zorder=3
        )
        ax.add_patch(box)
        ax.text(x, y + h/2 - 3.2, title, ha="center", va="top", fontsize=11, fontweight="bold", color=title_color, zorder=4)
        for idx, line in enumerate(subtitle_lines):
            ax.text(x, y + h/2 - 7.0 - idx * 3.0, line, ha="center", va="top", fontsize=9.2, color="#333333", zorder=4)

        if badge_text:
            bw = 13.5
            bh = 2.6
            bx = x
            by = y - h/2
            b_patch = FancyBboxPatch(
                (bx - bw/2, by - bh/2), bw, bh,
                boxstyle="round,pad=0.1,rounding_size=0.5",
                linewidth=1.6, edgecolor=badge_color, facecolor="#FFFFFF", zorder=5
            )
            ax.add_patch(b_patch)
            ax.text(bx, by, badge_text, ha="center", va="center", fontsize=8.0, fontweight="bold", color=badge_color, zorder=6)

    # Top Node
    draw_box(
        50, 90, 52, 9.5,
        "INGESTION PIPELINE INITIATION",
        ["Technical & Regulatory Document Corpus (PDFs)"],
        "#1B4F72", "#EAEDED", "#1B4F72"
    )

    # Level 2 Node
    draw_box(
        50, 71, 72, 10.5,
        "Corpus Tabular Modality Check",
        ["Does the corpus contain semi-structured tables, regulatory matrices, or section trees?"],
        "#B7950B", "#FEF9E7", "#7D6608"
    )

    # Arrow Top -> L2
    ax.annotate("", xy=(50, 76.5), xytext=(50, 85.0), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1C2833"))

    # Level 3 Nodes
    draw_box(
        26, 47, 45, 12.5,
        "Parser Capability Check",
        [
            "Is a layout-aware parser (IBM Docling TableFormer) available",
            "to extract structural Markdown AST hierarchies?"
        ],
        "#2874A6", "#EBF5FB", "#1B4F72"
    )

    draw_box(
        74, 47, 45, 12.5,
        "Resource Constraint Check",
        [
            "Is operational token budget",
            "the primary binding constraint?"
        ],
        "#2874A6", "#EBF5FB", "#1B4F72"
    )

    # Arrows L2 -> L3
    ax.annotate("", xy=(32, 53.5), xytext=(32, 65.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1C2833"))
    ax.text(30, 60, "YES (Semi-Structured)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1B4F72")

    ax.annotate("", xy=(68, 53.5), xytext=(68, 65.5), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1C2833"))
    ax.text(70, 60, "NO (Pure Prose)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1B4F72")

    # Leaf Nodes (Level 4)
    s_struct = data["structural"]
    s_rec = data["recursive"]
    s_f128 = data["fixed_128"]

    leaves = [
        (
            14.5, 17.5, 22.5, 16.5,
            "DEPLOY STRUCTURAL\n(Docling AST)",
            [
                f"Quality: Q = {s_struct['composite_quality']:.3f}",
                f"Cost: ${s_struct['cost_per_1k_usd']:.2f} / 1k Q",
                f"Table Recall: {s_struct['table_recall@5']:.3f}"
            ],
            "#1E8449", "#EAFAF1", "#196F3D",
            "[PARETO OPTIMAL]", "#1E8449"
        ),
        (
            38.0, 17.5, 22.5, 16.5,
            "DEPLOY RECURSIVE\n(512 Tokens)",
            [
                f"Quality: Q = {s_rec['composite_quality']:.3f}",
                f"Cost: ${s_rec['cost_per_1k_usd']:.2f} / 1k Q",
                "Syntactic Paragraphs"
            ],
            "#7F8C8D", "#F8F9F9", "#1C2833",
            "[DOMINATED FALLBACK]", "#7F8C8D"
        ),
        (
            62.0, 17.5, 22.5, 16.5,
            "DEPLOY FIXED-128\n(Sliding Window)",
            [
                f"Quality: Q = {s_f128['composite_quality']:.3f}",
                f"Cost: ${s_f128['cost_per_1k_usd']:.2f} / 1k Q",
                f"Faithfulness: {s_f128['faithfulness']:.3f}"
            ],
            "#1E8449", "#EAFAF1", "#196F3D",
            "[PARETO OPTIMAL]", "#1E8449"
        ),
        (
            85.5, 17.5, 22.5, 16.5,
            "DEPLOY RECURSIVE\n(512 Tokens)",
            [
                f"Quality: Q = {s_rec['composite_quality']:.3f}",
                f"Cost: ${s_rec['cost_per_1k_usd']:.2f} / 1k Q",
                "High Context Precision"
            ],
            "#7F8C8D", "#F8F9F9", "#1C2833",
            "[BALANCED DEFAULT]", "#2874A6"
        )
    ]

    for lx, ly, lw, lh, ltitle, lsubs, ec, fc, tc, btxt, bcol in leaves:
        box = FancyBboxPatch(
            (lx - lw/2, ly - lh/2), lw, lh,
            boxstyle="round,pad=0.3,rounding_size=2.0",
            linewidth=2.2, edgecolor=ec, facecolor=fc, zorder=3
        )
        ax.add_patch(box)
        ax.text(lx, ly + lh/2 - 2.2, ltitle, ha="center", va="top", fontsize=9.8, fontweight="bold", color=tc, zorder=4)
        for idx, line in enumerate(lsubs):
            ax.text(lx, ly + lh/2 - 8.2 - idx * 2.5, line, ha="center", va="top", fontsize=8.8, color="#333333", zorder=4)
        bw = 13.5
        bh = 2.4
        b_patch = FancyBboxPatch(
            (lx - bw/2, ly - lh/2 - bh/2), bw, bh,
            boxstyle="round,pad=0.1,rounding_size=0.4",
            linewidth=1.5, edgecolor=bcol, facecolor="#FFFFFF", zorder=5
        )
        ax.add_patch(b_patch)
        ax.text(lx, ly - lh/2, btxt, ha="center", va="center", fontsize=7.5, fontweight="bold", color=bcol, zorder=6)

    # Arrows L3 -> L4
    ax.annotate("", xy=(16.5, 26.0), xytext=(20.0, 40.5), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1C2833"))
    ax.text(16.0, 34.0, "YES", fontsize=9.5, fontweight="bold", color="#1B4F72", ha="center")

    ax.annotate("", xy=(38.0, 26.0), xytext=(44.0, 40.5), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1C2833"))
    ax.text(43.0, 34.0, "NO", fontsize=9.5, fontweight="bold", color="#1B4F72", ha="center")

    ax.annotate("", xy=(63.5, 26.0), xytext=(67.0, 40.5), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1C2833"))
    ax.text(62.5, 34.0, "YES", fontsize=9.5, fontweight="bold", color="#1B4F72", ha="center")

    ax.annotate("", xy=(85.0, 26.0), xytext=(87.0, 40.5), arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#1C2833"))
    ax.text(87.5, 34.0, "NO", fontsize=9.5, fontweight="bold", color="#1B4F72", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_all_charts_and_report():
    sum_path = RESULTS_DIR / "unified_benchmark_summary.json"
    if not sum_path.exists():
        raise FileNotFoundError(f"Summary not found at {sum_path}. Run 04_run_benchmark_and_ragas.py first.")

    # Clean up any old figures first
    for old_f in FIGURES_DIR.rglob("*"):
        if old_f.is_file():
            old_f.unlink()

    data = json.loads(sum_path.read_text(encoding="utf-8"))
    strats = [s for s in STRATEGIES if s in data]
    labels = [data[s]["display_name"] for s in strats]
    x = np.arange(len(strats))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # 1. Figure 4.1: Recall@k Performance Curves Across Top-k Horizons
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=300)
    ks = [1, 3, 5, 10]
    palette = {
        "fixed_128": "#E74C3C",
        "fixed_512": "#E67E22",
        "fixed_1024": "#27AE60",
        "recursive": "#2980B9",
        "structural": "#8E44AD",
        "semantic": "#16A085"
    }
    for s in strats:
        m = data[s]
        vals = [m["recall@1"], m["recall@3"], m["recall@5"], m["recall@10"]]
        ax.plot(ks, vals, marker="o", linewidth=2.2, markersize=7, color=palette[s], label=m["display_name"].lower())

    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Retrieval Horizon (Top-k)", fontweight="bold")
    ax.set_ylabel("Recall Score", fontweight="bold")
    ax.set_title("Recall@k Performance Curves Across Top-k Horizons", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_1_recall_curves.png", dpi=300)
    plt.close()

    # 2. Figure 4.2: RAGAS Answer & Retrieval Quality across 6 Chunking Strategies (gpt-6-luna)
    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=300)
    w = 0.20
    r5 = [data[s]["recall@5"] for s in strats]
    cp = [data[s]["context_precision"] for s in strats]
    fth = [data[s]["faithfulness"] for s in strats]
    rel = [data[s]["answer_relevance"] for s in strats]

    ax.bar(x - 1.5*w, r5, w, label="Recall@5", color="#1f77b4")
    ax.bar(x - 0.5*w, cp, w, label="Context Precision", color="#2ca02c")
    ax.bar(x + 0.5*w, fth, w, label="Faithfulness", color="#ff7f0e")
    ax.bar(x + 1.5*w, rel, w, label="Answer Relevance", color="#9467bd")

    ax.set_ylabel("Score [0.0 - 1.0]", fontweight="bold")
    ax.set_title(f"RAGAS Answer & Retrieval Quality across 6 Chunking Strategies ({LLM_MODEL})", fontsize=12, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_2_ragas_breakdown.png", dpi=300)
    plt.close()

    # 3. Figure 4.3: Domain Retrieval Quality: Table-Heavy vs. Narrative Text Queries (60 vs 60)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=300)
    t_r5 = [data[s]["table_recall@5"] for s in strats]
    n_r5 = [data[s]["narrative_recall@5"] for s in strats]

    b1 = ax.bar(x - 0.18, t_r5, 0.36, label="Table Queries Recall@5 (n=60)", color="#E74C3C", edgecolor="#222222", alpha=0.9)
    b2 = ax.bar(x + 0.18, n_r5, 0.36, label="Narrative Queries Recall@5 (n=60)", color="#3498DB", edgecolor="#222222", alpha=0.9)

    for b, v in zip(b1, t_r5):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    for b, v in zip(b2, n_r5):
        ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(strats, fontweight="bold")
    ax.set_ylim(0.0, 1.06)
    ax.set_ylabel("Recall@5 Score", fontweight="bold")
    ax.set_title("Domain Retrieval Quality: Table-Heavy (n=60) vs. Narrative Text (n=60) Queries", fontsize=12, fontweight="bold", pad=14)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_3_table_vs_narrative.png", dpi=300)
    plt.close()

    # 4. Figure 4.4: Empirical Pareto Efficiency Frontier (Cost per 1,000 Queries vs Composite Quality Qm)
    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=300)
    c_frontier = "#1B4F72"
    c_dominated = "#7F8C8D"
    c_line = "#2874A6"

    for s in strats:
        m = data[s]
        c = m["cost_per_1k_usd"]
        cm = m["composite_cost"]
        q = m["composite_quality"]
        name = m["display_name"]
        st = m["pareto_status"]
        if st == "Pareto Optimal":
            ax.scatter(c, q, color=c_frontier, s=160, zorder=5, label="Pareto Optimal (Non-dominated)" if "Pareto Optimal (Non-dominated)" not in [l.get_label() for l in ax.collections] else "")
            xy_off = (10, -6) if s == "structural" else (10, 4)
            ax.annotate(f"{name}\n(${c:.2f}, Cm={cm:.3f}, Qm={q:.3f})", (c, q), textcoords="offset points", xytext=xy_off, fontweight="bold", color=c_frontier, fontsize=9.0)
        else:
            ax.scatter(c, q, color=c_dominated, s=110, zorder=4, alpha=0.75, label="Dominated Configuration" if "Dominated Configuration" not in [l.get_label() for l in ax.collections] else "")
            if s == "fixed_1024":
                xy_off = (-115, -18)
            elif s == "semantic":
                xy_off = (-45, 10)
            elif s == "fixed_512":
                xy_off = (8, 6)
            else:
                xy_off = (8, -14)
            ax.annotate(f"{name}\n(${c:.2f}, Cm={cm:.3f}, Qm={q:.3f})", (c, q), textcoords="offset points", xytext=xy_off, color="#444444", fontsize=8.2)

    f_strats = sorted([s for s in strats if data[s]["pareto_status"] == "Pareto Optimal"], key=lambda k: data[k]["cost_per_1k_usd"])
    ax.plot([data[s]["cost_per_1k_usd"] for s in f_strats], [data[s]["composite_quality"] for s in f_strats], color=c_line, linestyle="--", linewidth=2.2, zorder=3, label="Empirical Pareto Frontier")

    all_costs = [data[s]["cost_per_1k_usd"] for s in strats]
    ax.set_xlim(min(all_costs) - 0.04, max(all_costs) + 0.08)
    ax.set_ylim(0.73, 0.89)
    ax.set_title("Empirical Pareto Efficiency Frontier: Chunking Strategy Quality vs. Operational Cost", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Operational Inference Cost ($ USD per 1,000 Queries)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Composite Quality Score ($Q_m$)", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_4_pareto_frontier.png", dpi=300)
    plt.close()

    # 5. Figure 4.5: Incremental Cost-Effectiveness Ratio (ICER) Across Frontier Transitions
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=300)
    s1, s2 = "fixed_128", "structural"
    dc = data[s2]["cost_per_1k_usd"] - data[s1]["cost_per_1k_usd"]
    dq = data[s2]["composite_quality"] - data[s1]["composite_quality"]
    icer_val = dc / dq

    bars = ax.bar(["Fixed-128 -> Structural"], [icer_val], color="#2E86C1", edgecolor="#1B4F72", width=0.42, zorder=3)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + icer_val * 0.04, f"ICER = ${icer_val:.2f}\n(+\u0394Q = {dq:.3f})", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylim(0, icer_val * 1.38)
    ax.set_title("Incremental Cost-Effectiveness Ratio (ICER) Across Frontier Transitions", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylabel("ICER ($ USD per +1.0 Composite Quality Gain)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure_4_5_icer_breakdown.png", dpi=300)
    plt.close()

    # 6. Figure 5.1: Engineering Decision Tree
    draw_decision_tree_figure(data, FIGURES_DIR / "figure_5_1_decision_tree.png")

    # Build Markdown Report
    lines = [
        "# Unified & Reproducible RAG Chunking Benchmark Report",
        "",
        "## 1. Methodological & Economic Configuration",
        f"- **Corpus**: 115 Technical & Regulatory PDF Documents converted via Docling to structured Markdown (4 categories, 24,088,389 characters, 2,246 reconstructed tables comprising 27,773 tabular rows).",
        f"- **Evaluation Benchmark**: Strictly balanced **120 queries** (**60 `table_numeric`** and **60 `narrative_text`**; 4 categories × 6 subtypes × 5 queries = 120).",
        f"- **Unified Embedding Model**: OpenAI `{EMBEDDING_MODEL}` (1536 dimensions, cosine similarity, `$0.02 / 1M tokens`) used for **both** Semantic Chunking (sentence-level breakpoint detection) **and** ChromaDB vector indexing/retrieval across all 6 strategies.",
        f"- **Unified LLM Generator & RAGAS Judge**: OpenAI `{LLM_MODEL}` (`reasoning_effort='low'`, **Full Top-5 Context Judge**, Input: **`$0.10 / 1M tokens`**, Output: **`$0.50 / 1M tokens`**).",
        f"- **Retrieval Architecture**: Pure bi-encoder cosine similarity search in ChromaDB (`hnsw:space = cosine`, no cross-encoder re-ranker).",
        "",
        "## 2. Corpus Chunking & Ingestion Footprint (Table 4.1)",
        "",
        "| Strategy | Total Chunks | Avg Tokens | Median Tokens | Table Chunks | Chunking Time (s) | Peak RAM (MB) | Vector DB (MB) | Ingest Cost ($) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for s in strats:
        m = data[s]
        lines.append(
            f"| **{m['display_name']}** | {m['total_chunks']:,} | {m['avg_tokens_per_chunk']:.1f} | {m['median_tokens_per_chunk']:.1f} | "
            f"{m['table_chunks_count']:,} | {m['chunking_wall_seconds']:.2f}s | {m['peak_ram_mb']:.2f} | {m['vector_storage_mb']:.2f} | ${m['ingest_cost_usd']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Tier 1 Retrieval Metrics Across 120 Balanced Queries (Table 4.2)",
        "",
        "| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Token IoU | Table R@5 (n=60) | Narrative R@5 (n=60) | Modality Gap (Δ) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ])
    for s in strats:
        m = data[s]
        lines.append(
            f"| **{m['display_name']}** | {m['recall@1']:.4f} | {m['recall@3']:.4f} | **{m['recall@5']:.4f}** | {m['recall@10']:.4f} | "
            f"**{m['mrr']:.4f}** | {m['mean_token_iou']:.4f} | {m['table_recall@5']:.4f} | {m['narrative_recall@5']:.4f} | {m['modality_gap_r5']:+.4f} |"
        )

    lines.extend([
        "",
        "## 4. Granular Subtype Recall@5 Breakdown (20 Queries per Subtype, Table 4.3)",
        "",
        "| Strategy | Exact Cell (n=20) | Cross Comp (n=20) | Range Thresh (n=20) | Scope Def (n=20) | Proc Steps (n=20) | Reg Oblig (n=20) |",
        "|:---|---:|---:|---:|---:|---:|---:|"
    ])
    for s in strats:
        m = data[s]
        lines.append(
            f"| **{m['display_name']}** | {m['r5_exact_cell_lookup']:.3f} | {m['r5_cross_comparison']:.3f} | {m['r5_range_threshold']:.3f} | "
            f"{m['r5_scope_definition']:.3f} | {m['r5_procedural_steps']:.3f} | {m['r5_regulatory_obligation']:.3f} |"
        )

    lines.extend([
        "",
        f"## 5. Tier 2 Generative RAGAS Evaluation via `{LLM_MODEL}` — Full Top-5 Context (Table 4.4)",
        "",
        "| Strategy | Context Precision@5 | Faithfulness (All) | Table Faith (n=60) | Narrative Faith (n=60) | Answer Relevance | AQS |",
        "|:---|---:|---:|---:|---:|---:|---:|"
    ])
    for s in strats:
        m = data[s]
        lines.append(
            f"| **{m['display_name']}** | {m['context_precision']:.4f} | **{m['faithfulness']:.4f}** | "
            f"{m['table_faithfulness']:.4f} | {m['narrative_faithfulness']:.4f} | **{m['answer_relevance']:.4f}** | {m['aqs']:.4f} |"
        )

    lines.extend([
        "",
        f"## 6. Operational Cost (`{LLM_MODEL}`: `$0.10/M` In, `$0.50/M` Out), Pareto Frontier & ICER (Tables 4.5 & 4.6)",
        "",
        "| Strategy | Context Tok (k=5) | Retrieval (ms) | Total Latency (ms) | Cost / 1k Queries ($) | Empirical API Cost / 1k ($) | Quality $Q_m$ | Cost Score $C_m$ | Pareto Status | ICER ($\Delta \text{Cost} / \Delta Q_m$) |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|:---|:---|"
    ])
    for s in strats:
        m = data[s]
        lines.append(
            f"| **{m['display_name']}** | {m['avg_prompt_tokens_k5']:.1f} | {m['avg_retrieval_ms']:.1f} | {m['avg_total_latency_ms']:.1f} | "
            f"**${m['cost_per_1k_usd']:.4f}** | ${m['empirical_api_cost_per_1k_usd']:.4f} | **{m['composite_quality']:.4f}** | "
            f"{m['composite_cost']:.4f} | **{m['pareto_status']}** | {m['icer_label']} |"
        )

    report_path = RESULTS_DIR / "UNIFIED_BENCHMARK_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[+] Generated all 6 thesis figures in {FIGURES_DIR} and full report at {report_path}")


if __name__ == "__main__":
    generate_all_charts_and_report()

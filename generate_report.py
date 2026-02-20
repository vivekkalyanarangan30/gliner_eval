"""PDF report generation with matplotlib tables and charts."""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _fmt(val: float) -> str:
    return f"{val * 100:.1f}"


def _load_results(output_dir: str) -> dict:
    """Load combined results JSON."""
    path = Path(output_dir) / "results.json"
    if not path.exists():
        logger.error(f"Results file not found: {path}. Run run_benchmark.py first.")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _add_title_page(pdf: PdfPages, results: dict, output_dir: str) -> None:
    """Add title page with date and config summary."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    ax.text(0.5, 0.7, "GLiNER-relex Benchmark Report", fontsize=24, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.6, f"Generated: {date.today().isoformat()}", fontsize=14,
            ha="center", va="center", transform=ax.transAxes, color="gray")

    datasets = sorted(results.keys())
    models = set()
    for ds in results.values():
        models.update(ds.keys())
    models = sorted(models)

    summary = f"Datasets: {', '.join(datasets)}\nModels: {', '.join(models)}"
    ax.text(0.5, 0.45, summary, fontsize=12, ha="center", va="center",
            transform=ax.transAxes, family="monospace")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_overall_table(pdf: PdfPages, results: dict) -> None:
    """Table 1: Overall Results — Dataset × Model with Micro-F1, Macro-F1, P, R."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Table 1: Overall Results", fontsize=16, fontweight="bold", pad=20)

    col_labels = ["Dataset", "Model", "Micro-F1", "Macro-F1", "Precision", "Recall"]
    cell_data = []
    cell_colors = []

    for ds_name in sorted(results.keys()):
        for model_name in sorted(results[ds_name].keys()):
            m = results[ds_name][model_name]
            cell_data.append([
                ds_name,
                model_name,
                _fmt(m["micro"]["f1"]),
                _fmt(m["macro"]["f1"]),
                _fmt(m["micro"]["precision"]),
                _fmt(m["micro"]["recall"]),
            ])
            cell_colors.append(["#f0f0f0"] * 6 if len(cell_data) % 2 == 0 else ["white"] * 6)

    if not cell_data:
        cell_data = [["No results"] + [""] * 5]
        cell_colors = [["white"] * 6]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#4472C4"] * 6,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header text color
    for j in range(len(col_labels)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_per_label_table(pdf: PdfPages, results: dict) -> None:
    """Table 2: Per-Label F1 (top 10 per dataset)."""
    for ds_name in sorted(results.keys()):
        model_names = sorted(results[ds_name].keys())

        # Collect all labels and find top 10 by max F1
        all_labels = set()
        for mn in model_names:
            all_labels.update(results[ds_name][mn].get("per_label", {}).keys())

        def max_f1(label):
            return max(
                results[ds_name][mn].get("per_label", {}).get(label, {}).get("f1", 0)
                for mn in model_names
            )

        top_labels = sorted(all_labels, key=max_f1, reverse=True)[:10]

        if not top_labels:
            continue

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title(f"Table 2: Per-Label F1 — {ds_name} (top 10)", fontsize=16,
                      fontweight="bold", pad=20)

        col_labels = ["Relation"] + [f"{mn} F1" for mn in model_names]
        cell_data = []
        cell_colors = []

        for i, label in enumerate(top_labels):
            row = [label[:30]]  # truncate long labels
            for mn in model_names:
                f1 = results[ds_name][mn].get("per_label", {}).get(label, {}).get("f1", 0)
                row.append(_fmt(f1))
            cell_data.append(row)
            cell_colors.append(["#f0f0f0"] * len(col_labels) if i % 2 == 0 else ["white"] * len(col_labels))

        table = ax.table(
            cellText=cell_data,
            colLabels=col_labels,
            cellColours=cell_colors,
            colColours=["#4472C4"] * len(col_labels),
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        for j in range(len(col_labels)):
            table[0, j].set_text_props(color="white", fontweight="bold")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def _add_cardinality_table(pdf: PdfPages, results: dict) -> None:
    """Table 3: Cardinality Analysis."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title("Table 3: Cardinality Analysis (F1)", fontsize=16, fontweight="bold", pad=20)

    col_labels = ["Dataset", "Model", "One-to-One", "One-to-Many", "Many-to-Many"]
    cell_data = []
    cell_colors = []

    for ds_name in sorted(results.keys()):
        for model_name in sorted(results[ds_name].keys()):
            m = results[ds_name][model_name]
            card = m.get("cardinality", {})
            cell_data.append([
                ds_name,
                model_name,
                _fmt(card.get("one-to-one", {}).get("f1", 0)),
                _fmt(card.get("one-to-many", {}).get("f1", 0)),
                _fmt(card.get("many-to-many", {}).get("f1", 0)),
            ])
            cell_colors.append(["#f0f0f0"] * 5 if len(cell_data) % 2 == 0 else ["white"] * 5)

    if not cell_data:
        cell_data = [["No results"] + [""] * 4]
        cell_colors = [["white"] * 5]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#4472C4"] * 5,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    for j in range(len(col_labels)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _add_bar_chart(pdf: PdfPages, results: dict) -> None:
    """Bar chart: Side-by-side Micro-F1 comparison across models per dataset."""
    datasets = sorted(results.keys())
    all_models = set()
    for ds in results.values():
        all_models.update(ds.keys())
    model_names = sorted(all_models)

    if not datasets or not model_names:
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    import numpy as np
    x = np.arange(len(datasets))
    width = 0.8 / max(len(model_names), 1)
    colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]

    for i, mn in enumerate(model_names):
        values = []
        for ds in datasets:
            f1 = results[ds].get(mn, {}).get("micro", {}).get("f1", 0)
            values.append(f1 * 100)
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=mn, color=colors[i % len(colors)])
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Micro-F1 (%)", fontsize=12)
    ax.set_title("Micro-F1 Comparison Across Models and Datasets", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    results = _load_results(output_dir)

    pdf_path = Path(output_dir) / "benchmark_report.pdf"
    logger.info(f"Generating PDF report: {pdf_path}")

    with PdfPages(str(pdf_path)) as pdf:
        _add_title_page(pdf, results, output_dir)
        _add_overall_table(pdf, results)
        _add_per_label_table(pdf, results)
        _add_cardinality_table(pdf, results)
        _add_bar_chart(pdf, results)

    logger.info(f"PDF report generated: {pdf_path}")
    print(f"\nPDF report saved to: {pdf_path.absolute()}")


if __name__ == "__main__":
    main()

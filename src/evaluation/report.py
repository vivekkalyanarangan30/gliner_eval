"""Console tables + JSON/CSV export."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from tabulate import tabulate

from .metrics import MetricsResult

logger = logging.getLogger(__name__)


def _format_pct(val: float) -> str:
    return f"{val * 100:.1f}"


def generate_report(
    results: dict[str, dict[str, MetricsResult]],
    output_dir: str = "outputs",
) -> None:
    """Generate console, JSON, and CSV reports.

    Args:
        results: Nested dict of results[dataset_name][model_name] = MetricsResult.
        output_dir: Directory for output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # === Console: Overall Results Table ===
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    headers = ["Dataset", "Model", "Micro-F1", "Macro-F1", "Precision", "Recall"]
    rows = []
    for ds_name, model_results in sorted(results.items()):
        for model_name, metrics in sorted(model_results.items()):
            rows.append([
                ds_name,
                model_name,
                _format_pct(metrics.micro.f1),
                _format_pct(metrics.macro.f1),
                _format_pct(metrics.micro.precision),
                _format_pct(metrics.micro.recall),
            ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # === Console: Cardinality Analysis ===
    print("\n" + "=" * 80)
    print("CARDINALITY ANALYSIS")
    print("=" * 80)

    card_headers = ["Dataset", "Model", "One-to-One F1", "One-to-Many F1", "Many-to-Many F1"]
    card_rows = []
    for ds_name, model_results in sorted(results.items()):
        for model_name, metrics in sorted(model_results.items()):
            card_rows.append([
                ds_name,
                model_name,
                _format_pct(metrics.cardinality.get("one-to-one", _ZeroF1()).f1),
                _format_pct(metrics.cardinality.get("one-to-many", _ZeroF1()).f1),
                _format_pct(metrics.cardinality.get("many-to-many", _ZeroF1()).f1),
            ])

    print(tabulate(card_rows, headers=card_headers, tablefmt="grid"))

    # === JSON Export ===
    json_data = {}
    for ds_name, model_results in results.items():
        json_data[ds_name] = {}
        for model_name, metrics in model_results.items():
            json_data[ds_name][model_name] = {
                "micro": {"precision": metrics.micro.precision, "recall": metrics.micro.recall, "f1": metrics.micro.f1},
                "macro": {"precision": metrics.macro.precision, "recall": metrics.macro.recall, "f1": metrics.macro.f1},
                "per_label": {
                    label: {"precision": s.precision, "recall": s.recall, "f1": s.f1}
                    for label, s in metrics.per_label.items()
                },
                "cardinality": {
                    card: {"precision": s.precision, "recall": s.recall, "f1": s.f1}
                    for card, s in metrics.cardinality.items()
                },
                "total_gold": metrics.total_gold,
                "total_predicted": metrics.total_predicted,
                "total_tp": metrics.total_tp,
            }

    json_path = out / "results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"JSON results saved to {json_path}")

    # === CSV Export ===
    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    logger.info(f"CSV results saved to {csv_path}")

    # === Per-label top-10 to console ===
    for ds_name, model_results in sorted(results.items()):
        print(f"\n{'=' * 80}")
        print(f"PER-LABEL F1 (top 10) — {ds_name}")
        print("=" * 80)

        # Collect all labels across models
        all_labels = set()
        for metrics in model_results.values():
            all_labels.update(metrics.per_label.keys())

        # Sort by max F1 across models
        def max_f1(label):
            return max(
                (m.per_label.get(label, _ZeroF1()).f1 for m in model_results.values()),
                default=0,
            )

        top_labels = sorted(all_labels, key=max_f1, reverse=True)[:10]
        model_names = sorted(model_results.keys())

        label_headers = ["Relation"] + [f"{m} F1" for m in model_names]
        label_rows = []
        for label in top_labels:
            row = [label]
            for mn in model_names:
                f1 = model_results[mn].per_label.get(label, _ZeroF1()).f1
                row.append(_format_pct(f1))
            label_rows.append(row)

        print(tabulate(label_rows, headers=label_headers, tablefmt="grid"))

    print(f"\nResults saved to {out.absolute()}")


class _ZeroF1:
    """Sentinel for missing F1 scores."""
    precision = 0.0
    recall = 0.0
    f1 = 0.0

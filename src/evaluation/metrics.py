"""Micro/macro F1, per-label metrics, and cardinality analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.datasets.base import GoldRelation, RelationSample
from src.models.base import PredictedRelation

from .matching import MatchResult, match_predictions, normalize


@dataclass
class F1Score:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class MetricsResult:
    micro: F1Score = field(default_factory=F1Score)
    macro: F1Score = field(default_factory=F1Score)
    per_label: dict[str, F1Score] = field(default_factory=dict)
    cardinality: dict[str, F1Score] = field(default_factory=dict)
    total_gold: int = 0
    total_predicted: int = 0
    total_tp: int = 0
    per_sample_matches: list[MatchResult] = field(default_factory=list)


def _compute_f1(tp: int, fp: int, fn: int) -> F1Score:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return F1Score(precision=precision, recall=recall, f1=f1)


def _classify_cardinality(relations: list[GoldRelation]) -> str:
    """Classify a sample's relations into cardinality buckets.

    - one-to-one: each head and tail appears in exactly 1 relation
    - one-to-many: some head has multiple tails
    - many-to-many: both head and tail participate in multiple relations
    """
    if len(relations) <= 1:
        return "one-to-one"

    head_counts = defaultdict(int)
    tail_counts = defaultdict(int)
    for r in relations:
        head_counts[normalize(r.head.text)] += 1
        tail_counts[normalize(r.tail.text)] += 1

    multi_head = any(c > 1 for c in head_counts.values())
    multi_tail = any(c > 1 for c in tail_counts.values())

    if multi_head and multi_tail:
        return "many-to-many"
    elif multi_head or multi_tail:
        return "one-to-many"
    else:
        return "one-to-one"


def compute_metrics(
    samples: list[RelationSample],
    predictions: list[list[PredictedRelation]],
    mode: str = "strict",
) -> MetricsResult:
    """Compute all metrics for a model on a dataset.

    Args:
        samples: Gold standard samples.
        predictions: List of predicted relations per sample (aligned by index).
        mode: Matching mode ("strict" or "relaxed").

    Returns:
        MetricsResult with micro/macro F1, per-label, and cardinality analysis.
    """
    # Aggregate counts for micro F1
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Per-label counts
    label_tp: dict[str, int] = defaultdict(int)
    label_fp: dict[str, int] = defaultdict(int)
    label_fn: dict[str, int] = defaultdict(int)

    # Cardinality buckets
    card_tp: dict[str, int] = defaultdict(int)
    card_fp: dict[str, int] = defaultdict(int)
    card_fn: dict[str, int] = defaultdict(int)

    per_sample_matches = []

    for sample, preds in zip(samples, predictions):
        match = match_predictions(sample.relations, preds, mode=mode)
        per_sample_matches.append(match)

        total_tp += match.true_positives
        total_fp += match.false_positives
        total_fn += match.false_negatives

        # Per-label: count TPs by relation type
        for gold, _pred in match.tp_triples:
            label_tp[normalize(gold.relation)] += 1

        for pred in match.fp_triples:
            label_fp[normalize(pred.relation)] += 1

        for gold in match.fn_triples:
            label_fn[normalize(gold.relation)] += 1

        # Cardinality analysis
        card = _classify_cardinality(sample.relations)
        card_tp[card] += match.true_positives
        card_fp[card] += match.false_positives
        card_fn[card] += match.false_negatives

    # Micro F1
    micro = _compute_f1(total_tp, total_fp, total_fn)

    # Per-label F1
    all_labels = set(label_tp) | set(label_fp) | set(label_fn)
    per_label = {}
    for label in sorted(all_labels):
        per_label[label] = _compute_f1(
            label_tp.get(label, 0),
            label_fp.get(label, 0),
            label_fn.get(label, 0),
        )

    # Macro F1 (average over per-label)
    if per_label:
        macro_p = sum(s.precision for s in per_label.values()) / len(per_label)
        macro_r = sum(s.recall for s in per_label.values()) / len(per_label)
        macro_f1 = sum(s.f1 for s in per_label.values()) / len(per_label)
        macro = F1Score(precision=macro_p, recall=macro_r, f1=macro_f1)
    else:
        macro = F1Score()

    # Cardinality F1
    cardinality = {}
    for card in ["one-to-one", "one-to-many", "many-to-many"]:
        if card_tp.get(card, 0) + card_fp.get(card, 0) + card_fn.get(card, 0) > 0:
            cardinality[card] = _compute_f1(
                card_tp.get(card, 0),
                card_fp.get(card, 0),
                card_fn.get(card, 0),
            )

    return MetricsResult(
        micro=micro,
        macro=macro,
        per_label=per_label,
        cardinality=cardinality,
        total_gold=total_tp + total_fn,
        total_predicted=total_tp + total_fp,
        total_tp=total_tp,
        per_sample_matches=per_sample_matches,
    )

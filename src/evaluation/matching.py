"""Text normalization and strict/relaxed triple matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.datasets.base import GoldRelation
from src.models.base import PredictedRelation


def normalize(text: str) -> str:
    """Lowercase, strip, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


@dataclass
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    tp_triples: list[tuple[GoldRelation, PredictedRelation]]
    fp_triples: list[PredictedRelation]
    fn_triples: list[GoldRelation]


def _strict_match(gold: GoldRelation, pred: PredictedRelation) -> bool:
    """Exact match on normalized (head_text, tail_text, relation)."""
    return (
        normalize(gold.head.text) == normalize(pred.head_text)
        and normalize(gold.tail.text) == normalize(pred.tail_text)
        and normalize(gold.relation) == normalize(pred.relation)
    )


def _relaxed_match(gold: GoldRelation, pred: PredictedRelation) -> bool:
    """Substring containment for entities + exact relation match."""
    g_head = normalize(gold.head.text)
    g_tail = normalize(gold.tail.text)
    p_head = normalize(pred.head_text)
    p_tail = normalize(pred.tail_text)

    head_match = g_head in p_head or p_head in g_head
    tail_match = g_tail in p_tail or p_tail in g_tail
    rel_match = normalize(gold.relation) == normalize(pred.relation)

    return head_match and tail_match and rel_match


def match_predictions(
    gold_relations: list[GoldRelation],
    predicted_relations: list[PredictedRelation],
    mode: str = "strict",
) -> MatchResult:
    """Match predictions against gold labels using greedy matching.

    Each gold triple is matched at most once to prevent double-counting.

    Args:
        gold_relations: Gold standard relations.
        predicted_relations: Model predictions.
        mode: "strict" for exact match, "relaxed" for substring containment.

    Returns:
        MatchResult with TP/FP/FN counts and matched triples.
    """
    match_fn = _strict_match if mode == "strict" else _relaxed_match

    matched_gold = set()
    matched_pred = set()
    tp_triples = []

    # Greedy matching: iterate predictions, try to match each to an unmatched gold
    for pi, pred in enumerate(predicted_relations):
        for gi, gold in enumerate(gold_relations):
            if gi in matched_gold:
                continue
            if match_fn(gold, pred):
                matched_gold.add(gi)
                matched_pred.add(pi)
                tp_triples.append((gold, pred))
                break

    tp = len(tp_triples)
    fp_triples = [p for i, p in enumerate(predicted_relations) if i not in matched_pred]
    fn_triples = [g for i, g in enumerate(gold_relations) if i not in matched_gold]

    return MatchResult(
        true_positives=tp,
        false_positives=len(fp_triples),
        false_negatives=len(fn_triples),
        tp_triples=tp_triples,
        fp_triples=fp_triples,
        fn_triples=fn_triples,
    )

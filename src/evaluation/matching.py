"""Text normalization and strict triple matching."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.datasets.base import GoldRelation
from src.models.base import PredictedRelation


def normalize(text: str) -> str:
    """Lowercase, strip, collapse whitespace, normalize punctuation, strip articles."""
    t = text.lower().strip()
    # Remove spaces before punctuation: "Dec 10 , 1260" → "Dec 10, 1260"
    t = re.sub(r"\s+([,.:;!?)])", r"\1", t)
    # Remove spaces after opening brackets
    t = re.sub(r"([([\[])\s+", r"\1", t)
    # Collapse remaining whitespace
    t = re.sub(r"\s+", " ", t)
    # Strip leading articles (standard NLP normalization)
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return t


def _entity_matches(gold_entity_text: str, gold_aliases: list[str], pred_text: str) -> bool:
    """Check if predicted entity text matches gold entity or any of its aliases."""
    norm_pred = normalize(pred_text)
    # Check primary text
    if normalize(gold_entity_text) == norm_pred:
        return True
    # Check all aliases (coreference mentions for document-level RE)
    for alias in gold_aliases:
        if normalize(alias) == norm_pred:
            return True
    return False


@dataclass
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    tp_triples: list[tuple[GoldRelation, PredictedRelation]]
    fp_triples: list[PredictedRelation]
    fn_triples: list[GoldRelation]


def _strict_match(gold: GoldRelation, pred: PredictedRelation) -> bool:
    """Exact match on normalized (head_text, tail_text, relation).

    For head/tail, checks against primary text AND all aliases (coreference
    handling for document-level RE, standard in DocRED evaluation).
    """
    return (
        _entity_matches(gold.head.text, gold.head.aliases, pred.head_text)
        and _entity_matches(gold.tail.text, gold.tail.aliases, pred.tail_text)
        and normalize(gold.relation) == normalize(pred.relation)
    )


def match_predictions(
    gold_relations: list[GoldRelation],
    predicted_relations: list[PredictedRelation],
    mode: str = "strict",
) -> MatchResult:
    """Match predictions against gold labels using greedy matching.

    Each gold triple is matched at most once to prevent double-counting.
    Uses strict exact-match on normalized (head_text, tail_text, relation).
    Entity matching considers all aliases (coreference mentions).

    Args:
        gold_relations: Gold standard relations.
        predicted_relations: Model predictions.
        mode: Only "strict" is supported (exact match after normalization).

    Returns:
        MatchResult with TP/FP/FN counts and matched triples.
    """
    matched_gold = set()
    matched_pred = set()
    tp_triples = []

    # Greedy matching: iterate predictions, try to match each to an unmatched gold
    for pi, pred in enumerate(predicted_relations):
        for gi, gold in enumerate(gold_relations):
            if gi in matched_gold:
                continue
            if _strict_match(gold, pred):
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

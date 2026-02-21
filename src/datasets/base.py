"""Unified data structures for relation extraction benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field

# Tokens that should NOT have a space inserted before them
_NO_SPACE_BEFORE = {
    ".", ",", ";", ":", "!", "?", "'", ")", "]", "}",
    "'s", "n't", "'re", "'ve", "'ll", "'d", "'m",
    "%", "''",
}
# Tokens that should NOT have a space inserted after them
_NO_SPACE_AFTER = {"(", "[", "{", "``"}


def reconstruct_text(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens into natural text, tracking character offsets per token.

    Handles punctuation properly: no space before commas/periods, etc.
    Returns (full_text, offsets) where offsets[i] = (start, end) for token i.
    """
    parts: list[str] = []
    tok_offsets: list[tuple[int, int]] = []
    pos = 0
    prev_tok = None
    for tok in tokens:
        if prev_tok is not None:
            if tok in _NO_SPACE_BEFORE or prev_tok in _NO_SPACE_AFTER:
                pass
            else:
                parts.append(" ")
                pos += 1
        start = pos
        end = pos + len(tok)
        tok_offsets.append((start, end))
        parts.append(tok)
        pos = end
        prev_tok = tok
    return "".join(parts), tok_offsets


@dataclass
class Entity:
    text: str
    type: str
    start_char: int = -1
    end_char: int = -1
    aliases: list[str] = field(default_factory=list)


@dataclass
class GoldRelation:
    head: Entity
    tail: Entity
    relation: str


@dataclass
class RelationSample:
    id: str
    text: str
    entities: list[Entity] = field(default_factory=list)
    relations: list[GoldRelation] = field(default_factory=list)
    dataset: str = ""
    domain: str = ""  # for CrossRE


@dataclass
class DatasetInfo:
    """Metadata about a loaded dataset split."""
    name: str
    samples: list[RelationSample]
    entity_types: list[str]
    relation_types: list[str]

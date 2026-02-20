"""Unified data structures for relation extraction benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    text: str
    type: str
    start_char: int = -1
    end_char: int = -1


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

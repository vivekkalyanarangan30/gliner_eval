"""Base model ABC, PredictedRelation dataclass, and shared prompt template."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.datasets.base import RelationSample


@dataclass
class PredictedRelation:
    head_text: str
    head_type: str
    tail_text: str
    tail_type: str
    relation: str
    score: float = 1.0


PROMPT_TEMPLATE = """Extract all relations between entities from the text below.

Allowed entity types: {entity_types}
Allowed relation types: {relation_types}

Text: "{text}"

Return JSON: {{"relations": [{{"head_text": "...", "head_type": "...", "tail_text": "...", "tail_type": "...", "relation": "..."}}]}}
Only use the provided types. If none found, return {{"relations": []}}."""


class BaseRelationModel(ABC):
    """Abstract base class for relation extraction models."""

    name: str = "base"

    @abstractmethod
    def predict(
        self,
        sample: RelationSample,
        entity_types: list[str],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        """Extract relations from a sample.

        Args:
            sample: The input sample with text.
            entity_types: Allowed entity types.
            relation_types: Allowed relation types.

        Returns:
            List of predicted relations.
        """
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Free model memory."""
        ...

    def format_prompt(self, text: str, entity_types: list[str], relation_types: list[str]) -> str:
        return PROMPT_TEMPLATE.format(
            entity_types=", ".join(entity_types),
            relation_types=", ".join(relation_types),
            text=text,
        )

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


PROMPT_TEMPLATE = """You are a relation extraction system. Extract all relations between entities from the given text.

CRITICAL RULES:
1. Entity text: Copy the EXACT substring from the text. Do not paraphrase, abbreviate, or reword entity mentions.
2. Relation labels: You MUST use ONLY labels from the allowed relation types list below. Use the label EXACTLY as written — do not rephrase, synonymize, or abbreviate.
3. Entity types: You MUST use ONLY types from the allowed entity types list below.
4. Extract ALL relations you can identify. Do not stop early.

ALLOWED ENTITY TYPES:
{entity_types}

ALLOWED RELATION TYPES:
{relation_types}

TEXT:
\"\"\"{text}\"\"\"

Return a JSON object with a "relations" array. Each element must have exactly these fields:
- "head_text": exact text span from the input for the head entity
- "head_type": one of the allowed entity types
- "tail_text": exact text span from the input for the tail entity
- "tail_type": one of the allowed entity types
- "relation": one of the allowed relation types (EXACTLY as listed above)

If no relations found, return {{"relations": []}}."""


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
        entity_list = "\n".join(f"  - {t}" for t in entity_types)
        relation_list = "\n".join(f"  - {t}" for t in relation_types)
        return PROMPT_TEMPLATE.format(
            entity_types=entity_list,
            relation_types=relation_list,
            text=text,
        )

"""GPT-5 via OpenAI API with structured output."""

from __future__ import annotations

import json
import logging
import os
import time

import openai

from src.datasets.base import RelationSample

from .base import BaseRelationModel, PredictedRelation

logger = logging.getLogger(__name__)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "relation_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "head_text": {"type": "string"},
                            "head_type": {"type": "string"},
                            "tail_text": {"type": "string"},
                            "tail_type": {"type": "string"},
                            "relation": {"type": "string"},
                        },
                        "required": ["head_text", "head_type", "tail_text", "tail_type", "relation"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["relations"],
            "additionalProperties": False,
        },
    },
}


class OpenAIModel(BaseRelationModel):
    name = "gpt5"

    def __init__(
        self,
        model_name: str = "gpt-5",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def load(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        self._client = openai.OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized for model: {self.model_name}")

    def unload(self) -> None:
        self._client = None
        logger.info("OpenAI client released")

    def predict(
        self,
        sample: RelationSample,
        entity_types: list[str],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        if self._client is None:
            raise RuntimeError("Client not initialized. Call load() first.")

        prompt = self.format_prompt(sample.text, entity_types, relation_types)

        for attempt in range(3):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": self.max_tokens,
                    "response_format": RESPONSE_SCHEMA,
                }
                # GPT-5 only supports default temperature (1)
                if self.temperature > 0:
                    kwargs["temperature"] = self.temperature
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                break
            except openai.RateLimitError:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except Exception as e:
                logger.error(f"OpenAI API error for {sample.id}: {e}")
                return []
        else:
            logger.error(f"All retries failed for {sample.id}")
            return []

        if not content:
            finish = response.choices[0].finish_reason
            logger.error(f"Empty response from OpenAI for {sample.id} (finish_reason={finish})")
            return []

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Failed to parse OpenAI response for {sample.id}: {content[:200]}")
            return []

        predictions = []
        for rel in data.get("relations", []):
            predictions.append(
                PredictedRelation(
                    head_text=rel.get("head_text", ""),
                    head_type=rel.get("head_type", ""),
                    tail_text=rel.get("tail_text", ""),
                    tail_type=rel.get("tail_type", ""),
                    relation=rel.get("relation", ""),
                )
            )

        return predictions

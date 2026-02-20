"""Qwen3 via Ollama OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re

import openai
import requests

from src.datasets.base import RelationSample

from .base import BaseRelationModel, PredictedRelation

logger = logging.getLogger(__name__)


def _extract_json_fallback(text: str) -> dict | None:
    """Attempt to extract JSON from malformed LLM output."""
    # Try to find JSON block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


class OllamaModel(BaseRelationModel):
    name = "qwen3"

    def __init__(
        self,
        model_name: str = "qwen3:0.6b",
        ollama_base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None

    def _health_check(self) -> bool:
        """Verify Ollama is running and model is available."""
        try:
            base = self.ollama_base_url.replace("/v1", "")
            resp = requests.get(f"{base}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if self.model_name not in models:
                # Check without tag
                base_names = [m.split(":")[0] for m in models]
                model_base = self.model_name.split(":")[0]
                if model_base not in base_names:
                    logger.error(
                        f"Model {self.model_name} not found in Ollama. "
                        f"Available: {models}. Run: ollama pull {self.model_name}"
                    )
                    return False
            return True
        except requests.ConnectionError:
            logger.error("Ollama is not running. Start it with: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def load(self) -> None:
        if not self._health_check():
            raise RuntimeError("Ollama health check failed")
        self._client = openai.OpenAI(
            base_url=self.ollama_base_url,
            api_key="ollama",
        )
        logger.info(f"Ollama client initialized for model: {self.model_name}")

    def unload(self) -> None:
        self._client = None
        logger.info("Ollama client released")

    def predict(
        self,
        sample: RelationSample,
        entity_types: list[str],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        if self._client is None:
            raise RuntimeError("Client not initialized. Call load() first.")

        prompt = self.format_prompt(sample.text, entity_types, relation_types)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ollama API error for {sample.id}: {e}")
            return []

        # Parse response with fallback
        data = None
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = _extract_json_fallback(content or "")

        if data is None:
            logger.error(f"Failed to parse Ollama response for {sample.id}")
            return []

        predictions = []
        for rel in data.get("relations", []):
            if not isinstance(rel, dict):
                continue
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

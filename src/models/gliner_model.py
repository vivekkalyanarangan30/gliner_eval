"""GLiNER-relex model wrapper."""

from __future__ import annotations

import gc
import logging

from src.datasets.base import RelationSample

from .base import BaseRelationModel, PredictedRelation

logger = logging.getLogger(__name__)


class GLiNERModel(BaseRelationModel):
    name = "gliner"

    def __init__(
        self,
        model_id: str = "knowledgator/gliner-relex-large-v0.5",
        entity_threshold: float = 0.3,
        relation_threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.entity_threshold = entity_threshold
        self.relation_threshold = relation_threshold
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading GLiNER model: {self.model_id}")
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(self.model_id)
        logger.info("GLiNER model loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("GLiNER model unloaded")

    def predict(
        self,
        sample: RelationSample,
        entity_types: list[str],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Truncate text if too long (GLiNER typically handles ~512 subword tokens)
        text = sample.text
        if len(text) > 3000:
            logger.warning(f"Truncating text from {len(text)} to 3000 chars for sample {sample.id}")
            text = text[:3000]

        try:
            _, relations = self._model.inference(
                texts=[text],
                labels=entity_types,
                relations=relation_types,
                threshold=self.entity_threshold,
                relation_threshold=self.relation_threshold,
                flat_ner=False,
                return_relations=True,
            )
        except Exception as e:
            logger.error(f"GLiNER inference failed for {sample.id}: {e}")
            return []

        predictions = []
        for rel in relations[0] if relations else []:
            predictions.append(
                PredictedRelation(
                    head_text=rel["head"]["text"],
                    head_type=rel["head"].get("type", ""),
                    tail_text=rel["tail"]["text"],
                    tail_type=rel["tail"].get("type", ""),
                    relation=rel["relation"],
                    score=rel.get("score", 1.0),
                )
            )

        return predictions

"""GLiNER2 model wrapper."""

from __future__ import annotations

import gc
import logging

from src.datasets.base import RelationSample

from .base import BaseRelationModel, PredictedRelation

logger = logging.getLogger(__name__)


class GLiNER2Model(BaseRelationModel):
    name = "gliner2"

    def __init__(
        self,
        model_id: str = "fastino/gliner2-large-v1",
        relation_threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.relation_threshold = relation_threshold
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        logger.info(f"Loading GLiNER2 model: {self.model_id}")
        from gliner2 import GLiNER2

        self._model = GLiNER2.from_pretrained(self.model_id)
        logger.info("GLiNER2 model loaded")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            logger.info("GLiNER2 model unloaded")

    def predict(
        self,
        sample: RelationSample,
        entity_types: list[str],
        relation_types: list[str],
    ) -> list[PredictedRelation]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Truncate text if too long
        text = sample.text
        if len(text) > 3000:
            logger.warning(f"Truncating text from {len(text)} to 3000 chars for sample {sample.id}")
            text = text[:3000]

        try:
            results = self._model.extract_relations(
                text,
                relation_types,
                include_confidence=True,
            )
        except Exception as e:
            logger.error(f"GLiNER2 inference failed for {sample.id}: {e}")
            return []

        predictions = []
        # Output format: {'relation_extraction': {'rel_type': [{'head': {'text': ..., 'confidence': ...}, 'tail': {'text': ..., 'confidence': ...}}, ...]}}
        rel_dict = results.get("relation_extraction", {})
        for relation_label, extractions in rel_dict.items():
            for item in extractions:
                head_text = item["head"]["text"]
                tail_text = item["tail"]["text"]
                score = min(item["head"].get("confidence", 1.0), item["tail"].get("confidence", 1.0))
                if score < self.relation_threshold:
                    continue
                predictions.append(
                    PredictedRelation(
                        head_text=head_text,
                        head_type="",
                        tail_text=tail_text,
                        tail_type="",
                        relation=relation_label,
                        score=score,
                    )
                )

        # Deduplicate: keep only the top-3 highest-scoring relations per (head, tail) pair.
        from collections import defaultdict
        pair_preds: dict[tuple[str, str], list[PredictedRelation]] = defaultdict(list)
        for pred in predictions:
            key = (pred.head_text, pred.tail_text)
            pair_preds[key].append(pred)

        deduped = []
        for key, preds in pair_preds.items():
            preds.sort(key=lambda p: p.score, reverse=True)
            deduped.extend(preds[:3])

        return deduped

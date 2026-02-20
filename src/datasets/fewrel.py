"""FewRel dataset loader via GitHub raw download."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import requests

from .base import DatasetInfo, Entity, GoldRelation, RelationSample

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/thunlp/FewRel/master/data"
CACHE_DIR = Path.home() / ".cache" / "gliner_eval" / "fewrel"


def _download_file(filename: str) -> Path:
    """Download a FewRel file and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / filename

    if cached.exists():
        logger.info(f"Using cached FewRel file: {cached}")
        return cached

    url = f"{BASE_URL}/{filename}"
    logger.info(f"Downloading FewRel: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_text(resp.text, encoding="utf-8")
    return cached


def load_fewrel(
    split: str = "val_wiki",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> DatasetInfo:
    """Load FewRel dataset.

    Args:
        split: Data split (default: val_wiki).
        sample_size: Number of samples to use (None for all).
        random_seed: Random seed for sampling.

    Returns:
        DatasetInfo with samples, entity types, and relation types.
    """
    data_path = _download_file(f"{split}.json")
    pid2name_path = _download_file("pid2name.json")

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(pid2name_path, encoding="utf-8") as f:
        pid2name = json.load(f)

    samples = []
    all_entity_types = set()
    all_relation_types = set()
    sample_idx = 0

    for pid, instances in data.items():
        # Map PID to human-readable relation name
        rel_name = pid2name.get(pid, [pid])[0].lower() if pid in pid2name else pid.lower()
        all_relation_types.add(rel_name)

        for inst in instances:
            tokens = inst["tokens"]
            text = " ".join(tokens)

            # Head entity
            h_name = inst["h"][0]
            h_indices = inst["h"][2][0] if inst["h"][2] else []
            h_type = "entity"  # FewRel doesn't provide fine-grained entity types
            all_entity_types.add(h_type)

            # Compute head char offsets from token indices
            if h_indices:
                h_start_char = len(" ".join(tokens[: h_indices[0]])) + (1 if h_indices[0] > 0 else 0)
                h_end_char = h_start_char + len(" ".join(tokens[i] for i in h_indices))
            else:
                h_start_char = -1
                h_end_char = -1

            head = Entity(text=h_name, type=h_type, start_char=h_start_char, end_char=h_end_char)

            # Tail entity
            t_name = inst["t"][0]
            t_indices = inst["t"][2][0] if inst["t"][2] else []
            t_type = "entity"
            all_entity_types.add(t_type)

            if t_indices:
                t_start_char = len(" ".join(tokens[: t_indices[0]])) + (1 if t_indices[0] > 0 else 0)
                t_end_char = t_start_char + len(" ".join(tokens[i] for i in t_indices))
            else:
                t_start_char = -1
                t_end_char = -1

            tail = Entity(text=t_name, type=t_type, start_char=t_start_char, end_char=t_end_char)

            relation = GoldRelation(head=head, tail=tail, relation=rel_name)

            samples.append(
                RelationSample(
                    id=f"fewrel_{sample_idx}",
                    text=text,
                    entities=[head, tail],
                    relations=[relation],
                    dataset="fewrel",
                )
            )
            sample_idx += 1

    if sample_size is not None and sample_size < len(samples):
        rng = random.Random(random_seed)
        samples = rng.sample(samples, sample_size)

    logger.info(f"Loaded {len(samples)} FewRel samples with {len(all_relation_types)} relation types")

    return DatasetInfo(
        name="fewrel",
        samples=samples,
        entity_types=sorted(all_entity_types),
        relation_types=sorted(all_relation_types),
    )

"""MultiTACRED dataset loader from local JSON files (requires LDC agreement)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .base import DatasetInfo, Entity, GoldRelation, RelationSample, reconstruct_text

logger = logging.getLogger(__name__)

# PTB token conversion (same as the official HF loading script)
_PTB_MAPPING = {
    "-lrb-": "(", "-rrb-": ")", "-lsb-": "[",
    "-rsb-": "]", "-lcb-": "{", "-rcb-": "}",
}

# Split name → filename template ({lang} is replaced at runtime)
_SPLIT_FILES = {
    "train": "train_{lang}.json",
    "test": "test_{lang}.json",
    "dev": "dev_{lang}.json",
    "validation": "dev_{lang}.json",
}


def _convert_ptb_token(token: str) -> str:
    return _PTB_MAPPING.get(token.lower(), token)


def load_multitacred(
    data_dir: str,
    language: str = "en",
    split: str = "test",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> DatasetInfo:
    """Load MultiTACRED dataset from local JSON files.

    Args:
        data_dir: Path to the extracted MultiTACRED data directory.
        language: Language code (e.g. "en", "de", "fr").
        split: Data split — "train", "test", or "dev"/"validation".
        sample_size: Number of samples to use (None for all).
        random_seed: Random seed for sampling.

    Returns:
        DatasetInfo with samples, entity types, and relation types.
    """
    filename = _SPLIT_FILES.get(split)
    if filename is None:
        raise ValueError(f"Unknown split '{split}'. Choose from: {list(_SPLIT_FILES.keys())}")

    filepath = Path(data_dir) / language / filename.format(lang=language)
    if not filepath.exists():
        raise FileNotFoundError(
            f"MultiTACRED data not found at {filepath}. "
            f"Download from https://catalog.ldc.upenn.edu/LDC2024T09 and extract to '{data_dir}'."
        )

    logger.info(f"Loading MultiTACRED language={language}, split={split} from {filepath}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    all_entity_types = set()
    all_relation_types = set()

    for idx, inst in enumerate(data):
        relation = inst["relation"]
        if relation == "no_relation":
            continue

        tokens = [_convert_ptb_token(t) for t in inst["token"]]
        text, tok_offsets = reconstruct_text(tokens)

        rel_name = relation.lower()
        all_relation_types.add(rel_name)

        # Raw TACRED JSON uses inclusive end indices
        subj_start = inst["subj_start"]
        subj_end = inst["subj_end"]  # inclusive
        obj_start = inst["obj_start"]
        obj_end = inst["obj_end"]  # inclusive

        h_start_char = tok_offsets[subj_start][0]
        h_end_char = tok_offsets[subj_end][1]
        h_text = text[h_start_char:h_end_char]

        t_start_char = tok_offsets[obj_start][0]
        t_end_char = tok_offsets[obj_end][1]
        t_text = text[t_start_char:t_end_char]

        h_type = inst["subj_type"].lower()
        t_type = inst["obj_type"].lower()
        all_entity_types.add(h_type)
        all_entity_types.add(t_type)

        head = Entity(text=h_text, type=h_type, start_char=h_start_char, end_char=h_end_char)
        tail = Entity(text=t_text, type=t_type, start_char=t_start_char, end_char=t_end_char)
        gold = GoldRelation(head=head, tail=tail, relation=rel_name)

        samples.append(
            RelationSample(
                id=f"multitacred_{inst.get('id', idx)}",
                text=text,
                entities=[head, tail],
                relations=[gold],
                dataset="multitacred",
            )
        )

    if sample_size is not None and sample_size < len(samples):
        rng = random.Random(random_seed)
        samples = rng.sample(samples, sample_size)

    logger.info(f"Loaded {len(samples)} MultiTACRED samples with {len(all_relation_types)} relation types")

    return DatasetInfo(
        name="multitacred",
        samples=samples,
        entity_types=sorted(all_entity_types),
        relation_types=sorted(all_relation_types),
    )

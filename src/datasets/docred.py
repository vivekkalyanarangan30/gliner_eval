"""DocRED dataset loader via hf_hub_download."""

from __future__ import annotations

import gzip
import json
import logging
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from .base import DatasetInfo, Entity, GoldRelation, RelationSample

logger = logging.getLogger(__name__)

REPO_ID = "thunlp/docred"


def _download_file(filename: str) -> Path:
    """Download a file from the DocRED HuggingFace repo."""
    return Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset"))


def _load_gzip_json(path: Path) -> dict | list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _reconstruct_text(sents: list[list[str]]) -> tuple[str, list[list[tuple[int, int]]]]:
    """Join tokenized sentences into full text, tracking char offsets per token.

    Returns (full_text, offsets) where offsets[sent_idx][tok_idx] = (start, end).
    """
    parts = []
    offsets = []
    pos = 0
    for sent_tokens in sents:
        sent_offsets = []
        for i, tok in enumerate(sent_tokens):
            start = pos
            end = pos + len(tok)
            sent_offsets.append((start, end))
            parts.append(tok)
            pos = end
            # Add space after each token (except last token of last sentence)
            parts.append(" ")
            pos += 1
        offsets.append(sent_offsets)
    full_text = "".join(parts).rstrip()
    return full_text, offsets


def load_docred(
    split: str = "validation",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> DatasetInfo:
    """Load DocRED dataset.

    Args:
        split: Only "validation" is supported (no label leakage).
        sample_size: Number of samples to use (None for all).
        random_seed: Random seed for sampling.

    Returns:
        DatasetInfo with samples, entity types, and relation types.
    """
    logger.info("Downloading DocRED data files...")
    dev_path = _download_file("data/dev.json.gz")
    rel_info_path = _download_file("data/rel_info.json.gz")

    data = _load_gzip_json(dev_path)
    rel_info = _load_gzip_json(rel_info_path)

    # Build PID → human-readable name mapping
    pid_to_name = {pid: name for pid, name in rel_info.items()}

    samples = []
    all_entity_types = set()
    all_relation_types = set()

    for idx, doc in enumerate(data):
        text, offsets = _reconstruct_text(doc["sents"])
        vertex_set = doc["vertexSet"]

        # Build entity list from vertex set
        entities = []
        for vert in vertex_set:
            mention = vert[0]  # first mention
            name = mention["name"]
            etype = mention["type"].lower()
            all_entity_types.add(etype)

            # Compute char offsets from token positions
            sent_id = mention["sent_id"]
            tok_start = mention["pos"][0]
            tok_end = mention["pos"][1]  # exclusive
            if sent_id < len(offsets) and tok_start < len(offsets[sent_id]):
                start_char = offsets[sent_id][tok_start][0]
                end_idx = min(tok_end - 1, len(offsets[sent_id]) - 1)
                end_char = offsets[sent_id][end_idx][1]
            else:
                start_char = -1
                end_char = -1

            entities.append(Entity(text=name, type=etype, start_char=start_char, end_char=end_char))

        # Build relations
        relations = []
        for label in doc.get("labels", []):
            head_idx = label["h"]
            tail_idx = label["t"]
            pid = label["r"]
            rel_name = pid_to_name.get(pid, pid).lower()
            all_relation_types.add(rel_name)

            head_entity = entities[head_idx]
            tail_entity = entities[tail_idx]
            relations.append(GoldRelation(head=head_entity, tail=tail_entity, relation=rel_name))

        samples.append(
            RelationSample(
                id=f"docred_{idx}",
                text=text,
                entities=entities,
                relations=relations,
                dataset="docred",
            )
        )

    # Filter to samples that have at least one relation
    samples = [s for s in samples if s.relations]

    if sample_size is not None and sample_size < len(samples):
        rng = random.Random(random_seed)
        samples = rng.sample(samples, sample_size)

    logger.info(f"Loaded {len(samples)} DocRED samples with {len(all_relation_types)} relation types")

    return DatasetInfo(
        name="docred",
        samples=samples,
        entity_types=sorted(all_entity_types),
        relation_types=sorted(all_relation_types),
    )

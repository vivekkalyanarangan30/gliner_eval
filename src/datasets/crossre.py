"""CrossRE dataset loader via GitHub raw download."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import requests

from .base import DatasetInfo, Entity, GoldRelation, RelationSample, reconstruct_text

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/mainlp/CrossRE/main/crossre_data"
CACHE_DIR = Path.home() / ".cache" / "gliner_eval" / "crossre"

# Map concatenated CrossRE entity types to natural language labels
# so models (especially GLiNER) can understand them
_ENTITY_TYPE_MAP = {
    "academicjournal": "academic journal",
    "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound",
    "chemicalelement": "chemical element",
    "programlang": "programming language",
}


def _download_domain(domain: str, split: str) -> Path:
    """Download a CrossRE domain file and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{domain}-{split}.json"
    cached = CACHE_DIR / filename

    if cached.exists():
        logger.info(f"Using cached CrossRE file: {cached}")
        return cached

    url = f"{BASE_URL}/{filename}"
    logger.info(f"Downloading CrossRE: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    cached.write_text(resp.text, encoding="utf-8")
    return cached


def _parse_jsonl(path: Path, domain: str) -> list[RelationSample]:
    """Parse a CrossRE JSONL file into RelationSample list."""
    samples = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            tokens = obj["sentence"]
            text, tok_offsets = reconstruct_text(tokens)

            # Build entity map keyed by (start, end) for dedup
            entity_map: dict[tuple[int, int], Entity] = {}
            for ner_entry in obj.get("ner", []):
                start, end, etype = ner_entry[0], ner_entry[1], ner_entry[2]
                # end is INCLUSIVE in CrossRE
                char_start = tok_offsets[start][0]
                char_end = tok_offsets[end][1]  # end is inclusive
                entity_text = text[char_start:char_end]
                raw_type = etype.lower()
                entity_map[(start, end)] = Entity(
                    text=entity_text,
                    type=_ENTITY_TYPE_MAP.get(raw_type, raw_type),
                    start_char=char_start,
                    end_char=char_end,
                )

            entities = list(entity_map.values())

            # Build relations — ends are INCLUSIVE
            relations = []
            for rel_entry in obj.get("relations", []):
                s1, e1, s2, e2 = rel_entry[0], rel_entry[1], rel_entry[2], rel_entry[3]
                rel_type = rel_entry[4].lower()

                head = entity_map.get((s1, e1))
                tail = entity_map.get((s2, e2))
                if head is None:
                    hc_s = tok_offsets[s1][0]
                    hc_e = tok_offsets[e1][1]
                    head = Entity(text=text[hc_s:hc_e], type="unknown")
                if tail is None:
                    tc_s = tok_offsets[s2][0]
                    tc_e = tok_offsets[e2][1]
                    tail = Entity(text=text[tc_s:tc_e], type="unknown")

                relations.append(GoldRelation(head=head, tail=tail, relation=rel_type))

            samples.append(
                RelationSample(
                    id=f"crossre_{domain}_{line_num}",
                    text=text,
                    entities=entities,
                    relations=relations,
                    dataset="crossre",
                    domain=domain,
                )
            )

    return samples


def load_crossre(
    domains: list[str] | None = None,
    split: str = "test",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> DatasetInfo:
    """Load CrossRE dataset for specified domains.

    Args:
        domains: List of domains to load (default: ai, news, science).
        split: Data split (default: test).
        sample_size: Number of samples to use (None for all).
        random_seed: Random seed for sampling.

    Returns:
        DatasetInfo with combined samples from all domains.
    """
    if domains is None:
        domains = ["ai", "news", "science"]

    all_samples = []
    all_entity_types = set()
    all_relation_types = set()

    for domain in domains:
        path = _download_domain(domain, split)
        domain_samples = _parse_jsonl(path, domain)

        for s in domain_samples:
            for e in s.entities:
                all_entity_types.add(e.type)
            for r in s.relations:
                all_relation_types.add(r.relation)

        all_samples.extend(domain_samples)

    # Filter to samples with relations
    all_samples = [s for s in all_samples if s.relations]

    if sample_size is not None and sample_size < len(all_samples):
        rng = random.Random(random_seed)
        all_samples = rng.sample(all_samples, sample_size)

    logger.info(
        f"Loaded {len(all_samples)} CrossRE samples from {domains} "
        f"with {len(all_relation_types)} relation types"
    )

    return DatasetInfo(
        name="crossre",
        samples=all_samples,
        entity_types=sorted(all_entity_types),
        relation_types=sorted(all_relation_types),
    )

# Multilingual Relation Extraction Benchmarks

A shortlist of multilingual RE benchmarks suitable for evaluating relation extraction models (GLiNER-relex, LLMs, etc.).

## Benchmarks

### RED^FM
- **Languages:** 7 (EN, FR, ES, DE, IT, PT, ZH)
- **Relations:** ~400 Wikidata types | **Size:** Human-revised subset of 40M+ auto-annotated triplets
- **Format:** Sentence-level, entity pairs + relation labels
- **Access:** [ACL Anthology](https://aclanthology.org/2023.acl-long.237/) · [arXiv](https://arxiv.org/abs/2306.09802)
- **Notes:** Shares Wikidata relation vocabulary with DocRED/FewRel. Strongest candidate for integration with existing benchmarks.

### MultiTACRED (integrated)
- **Languages:** 12 (AR, ZH, FI, FR, DE, HI, HU, JA, PL, RU, ES, TR)
- **Relations:** 41 (TACRED schema) | **Size:** ~106K instances/language (machine-translated)
- **Format:** Sentence-level, entity-typed relation classification
- **Access:** [HuggingFace](https://huggingface.co/datasets/DFKI-SLT/multitacred) · [arXiv](https://arxiv.org/abs/2305.04582) · [LDC](https://catalog.ldc.upenn.edu/LDC2024T09)
- **Notes:** Broadest language coverage. Requires LDC access ($25 for non-members). **Integrated into evaluation pipeline** — loader at `src/datasets/multitacred.py`, configurable via `config.yaml` (`datasets.multitacred`). Requires local data download from LDC; set `data_dir` to the extracted path.

### SMiLER
- **Languages:** 14 | **Relations:** 36 | **Size:** 1.1M sentences
- **Format:** Joint entity + relation extraction
- **Access:** [GitHub](https://github.com/samsungnlp/smiler) · [ACL Anthology](https://aclanthology.org/2021.eacl-main.166/)
- **Notes:** Largest multilingual RE dataset. Joint entity+RE format maps naturally to GLiNER-style models.

### RELX
- **Languages:** 5 (EN, FR, DE, ES, TR) | **Relations:** 37 (KBP-37) | **Size:** 502 parallel sentences
- **Format:** Relation classification with human-translated parallel data
- **Access:** [GitHub](https://github.com/boun-tabi/RELX) · [ACL Anthology](https://aclanthology.org/2020.findings-emnlp.32/)
- **Notes:** Small but gold-standard parallel data — ideal for controlled cross-lingual comparison.

## Comparison

| Dataset | Languages | Relations | Size | Translation | Best For |
|---------|-----------|-----------|------|-------------|----------|
| RED^FM | 7 | ~400 | Medium | Native | Wikidata-aligned evaluation |
| MultiTACRED | 12 | 41 | Large | Machine | Cross-lingual transfer |
| SMiLER | 14 | 36 | 1.1M | Native | Joint entity+RE at scale |
| RELX | 5 | 37 | 502 | Human | Controlled parallel evaluation |

## References

- [Multilingual Relation Extraction Survey (2025)](https://papers.dice-research.org/2025/IEEEAccess_MRE_Survey/public.pdf)
- [RE Techniques Survey — Springer (2025)](https://link.springer.com/article/10.1007/s10462-025-11280-0)

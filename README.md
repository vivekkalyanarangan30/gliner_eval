# GLiNER-relex Benchmark

Benchmarking **GLiNER-relex** (`knowledgator/gliner-relex-large-v0.5`) and **GLiNER2** (`fastino/gliner2-large-v1`) against **GPT-5** and **Qwen3 (0.6B)** on relation extraction across three datasets: DocRED, CrossRE, and FewRel.

## Prerequisites

- Python >= 3.10
- An OpenAI API key (for GPT-5 evaluation)
- [Ollama](https://ollama.com/) installed and running (for Qwen3 evaluation)

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd gliner_eval
pip install -e .
```

### 2. Set up your OpenAI API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Set up Ollama (for Qwen3)

```bash
# Start the Ollama server (if not already running)
ollama serve

# Pull the Qwen3 model (in a separate terminal)
ollama pull qwen3:0.6b
```

## Running the Benchmark

```bash
python run_benchmark.py
```

This will:
1. Download and cache all enabled datasets
2. Run each enabled model on each dataset sequentially
3. Print results to the console
4. Save per-model results to `outputs/results_<model>_<dataset>.json`
5. Save combined results to `outputs/results.json` and `outputs/results.csv`

### Generate the PDF report

After the benchmark completes:

```bash
python generate_report.py
```

This produces `outputs/benchmark_report.pdf` containing:
- Overall results table (Micro-F1, Macro-F1, Precision, Recall)
- Per-label F1 table (top 10 relations per dataset)
- Cardinality analysis (one-to-one, one-to-many, many-to-many)
- Bar chart comparing Micro-F1 across models

## Configuration

All settings are in `config.yaml`.

### Sample size

Control how many samples to evaluate per dataset:

```yaml
general:
  sample_size: 25    # number of samples per dataset
  random_seed: 42    # for reproducible sampling
```

To run on the **full dataset**, remove the `sample_size` field or set it to a very large number (the loader will cap at available samples):

```yaml
general:
  sample_size: 999999   # effectively uses all available samples
```

> **Note:** Running the full dataset with GPT-5 will take a long time and incur significant API costs. DocRED validation has ~1,000 samples, CrossRE test sets have ~300-500 per domain, and FewRel val_wiki has ~11,200 samples.

### Enable/disable specific models

```yaml
models:
  gliner:
    enabled: true      # toggle to false to skip GLiNER-relex
  gliner2:
    enabled: true      # toggle to false to skip GLiNER2
  gpt5:
    enabled: true      # toggle to false to skip GPT-5
  qwen3:
    enabled: true      # toggle to false to skip Qwen3
```

### Enable/disable specific datasets

```yaml
datasets:
  docred:
    enabled: true      # document-level, 96 Wikidata relations
  crossre:
    enabled: true
    domains: [ai, news, science]   # choose from: ai, literature, music, news, politics, science
  fewrel:
    enabled: true      # sentence-level, 16 Wikidata relations
```

### Model parameters

```yaml
models:
  gliner:
    model_id: knowledgator/gliner-relex-large-v0.5
    entity_threshold: 0.3      # NER confidence threshold
    relation_threshold: 0.5    # relation confidence threshold
  gliner2:
    model_id: fastino/gliner2-large-v1
    relation_threshold: 0.5    # relation confidence threshold
  gpt5:
    model_name: gpt-5
    temperature: 0.0           # 0.0 for reproducibility
    max_tokens: 16384
  qwen3:
    model_name: qwen3:0.6b
    ollama_base_url: http://localhost:11434/v1
    temperature: 0.0
    max_tokens: 2048
```

### Matching mode

```yaml
general:
  matching_mode: strict    # exact normalized match on (head_text, tail_text, relation)
```

### Using a custom config file

```bash
python run_benchmark.py path/to/custom_config.yaml
python generate_report.py path/to/custom_output_dir
```

## Output Files

All outputs are saved to the `outputs/` directory (configurable via `general.output_dir`):

| File | Description |
|------|-------------|
| `results_<model>_<dataset>.json` | Per-model, per-dataset detailed results with all predictions |
| `results.json` | Combined results across all models and datasets |
| `results.csv` | Summary table (convenient for LaTeX import) |
| `benchmark_report.pdf` | Publication-ready PDF with tables and charts |

## Project Structure

```
gliner_eval/
├── config.yaml              # Benchmark configuration
├── run_benchmark.py         # Main entry point
├── generate_report.py       # PDF report generator
├── src/
│   ├── config.py            # Config loading (pydantic + dotenv)
│   ├── datasets/
│   │   ├── base.py          # Shared data structures (RelationSample, Entity, GoldRelation)
│   │   ├── docred.py        # DocRED loader
│   │   ├── crossre.py       # CrossRE loader
│   │   └── fewrel.py        # FewRel loader
│   ├── models/
│   │   ├── base.py          # Base model ABC + shared prompt template
│   │   ├── gliner_model.py  # GLiNER-relex wrapper
│   │   ├── gliner2_model.py # GLiNER2 wrapper
│   │   ├── openai_model.py  # GPT-5 via OpenAI API
│   │   └── ollama_model.py  # Qwen3 via Ollama
│   └── evaluation/
│       ├── matching.py      # Strict triple matching with normalization
│       ├── metrics.py       # Micro/macro F1, per-label, cardinality analysis
│       └── report.py        # Console + JSON + CSV output
└── outputs/                 # Generated results (gitignored)
```

## Typical Workflow

```bash
# Quick test run (5 samples, GLiNER only)
# Edit config.yaml: sample_size: 5, disable gpt5 and qwen3
python run_benchmark.py

# Full benchmark (all models, 25 samples)
# Edit config.yaml: sample_size: 25, enable all models
python run_benchmark.py
python generate_report.py
```

## Hardware Notes

- **GLiNER** loads the model into CPU memory (~1.2 GB). Runs fast even on an 8 GB MacBook.
- **GLiNER2** loads the model into CPU memory. Similar footprint to GLiNER-relex.
- **GPT-5** calls the OpenAI API. No local compute needed, but expect ~45 min for 25 DocRED samples due to document length.
- **Qwen3 (0.6B)** runs locally via Ollama. Light enough for any machine. Expect ~15 min total for 25 samples across all datasets.
- Models are loaded and unloaded sequentially to keep memory usage low.

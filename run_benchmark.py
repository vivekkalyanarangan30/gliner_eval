"""Main orchestrator for GLiNER-relex benchmarking."""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from src.config import load_config
from src.datasets.base import DatasetInfo
from src.datasets.crossre import load_crossre
from src.datasets.docred import load_docred
from src.datasets.fewrel import load_fewrel
from src.evaluation.metrics import MetricsResult, compute_metrics
from src.evaluation.report import generate_report
from src.models.base import BaseRelationModel, PredictedRelation
from src.models.gliner_model import GLiNERModel
from src.models.ollama_model import OllamaModel
from src.models.openai_model import OpenAIModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_datasets(cfg) -> dict[str, DatasetInfo]:
    """Load all enabled datasets."""
    datasets = {}

    if cfg.datasets.docred.enabled:
        logger.info("Loading DocRED...")
        datasets["docred"] = load_docred(
            split=cfg.datasets.docred.split,
            sample_size=cfg.general.sample_size,
            random_seed=cfg.general.random_seed,
        )

    if cfg.datasets.crossre.enabled:
        logger.info("Loading CrossRE...")
        datasets["crossre"] = load_crossre(
            domains=cfg.datasets.crossre.domains,
            split=cfg.datasets.crossre.split,
            sample_size=cfg.general.sample_size,
            random_seed=cfg.general.random_seed,
        )

    if cfg.datasets.fewrel.enabled:
        logger.info("Loading FewRel...")
        datasets["fewrel"] = load_fewrel(
            split=cfg.datasets.fewrel.split,
            sample_size=cfg.general.sample_size,
            random_seed=cfg.general.random_seed,
        )

    return datasets


def build_models(cfg) -> list[BaseRelationModel]:
    """Build list of enabled models."""
    models = []

    if cfg.models.gliner.enabled:
        models.append(
            GLiNERModel(
                model_id=cfg.models.gliner.model_id,
                entity_threshold=cfg.models.gliner.entity_threshold,
                relation_threshold=cfg.models.gliner.relation_threshold,
            )
        )

    if cfg.models.gpt5.enabled:
        models.append(
            OpenAIModel(
                model_name=cfg.models.gpt5.model_name,
                temperature=cfg.models.gpt5.temperature,
                max_tokens=cfg.models.gpt5.max_tokens,
            )
        )

    if cfg.models.qwen3.enabled:
        models.append(
            OllamaModel(
                model_name=cfg.models.qwen3.model_name,
                ollama_base_url=cfg.models.qwen3.ollama_base_url,
                temperature=cfg.models.qwen3.temperature,
                max_tokens=cfg.models.qwen3.max_tokens,
            )
        )

    return models


def run_model_on_dataset(
    model: BaseRelationModel,
    dataset: DatasetInfo,
) -> list[list[PredictedRelation]]:
    """Run a model on all samples in a dataset."""
    all_predictions = []

    for sample in tqdm(dataset.samples, desc=f"  {model.name} on {dataset.name}", leave=False):
        try:
            preds = model.predict(sample, dataset.entity_types, dataset.relation_types)
        except Exception as e:
            logger.error(f"Prediction failed for {sample.id}: {e}")
            preds = []
        all_predictions.append(preds)

    return all_predictions


def save_intermediate(
    model_name: str,
    dataset_name: str,
    metrics: MetricsResult,
    predictions: list[list[PredictedRelation]],
    output_dir: str,
) -> None:
    """Save intermediate results after each model×dataset combo."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_name,
        "dataset": dataset_name,
        "micro_f1": metrics.micro.f1,
        "macro_f1": metrics.macro.f1,
        "precision": metrics.micro.precision,
        "recall": metrics.micro.recall,
        "total_gold": metrics.total_gold,
        "total_predicted": metrics.total_predicted,
        "total_tp": metrics.total_tp,
        "predictions": [
            [
                {
                    "head_text": p.head_text,
                    "head_type": p.head_type,
                    "tail_text": p.tail_text,
                    "tail_type": p.tail_type,
                    "relation": p.relation,
                    "score": p.score,
                }
                for p in sample_preds
            ]
            for sample_preds in predictions
        ],
    }

    path = out / f"results_{model_name}_{dataset_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Intermediate results saved: {path}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    cfg = load_config(config_path)

    logger.info(f"Config: sample_size={cfg.general.sample_size}, seed={cfg.general.random_seed}")
    logger.info(f"Matching mode: {cfg.general.matching_mode}")

    # Step 1: Load datasets
    datasets = load_datasets(cfg)
    if not datasets:
        logger.error("No datasets enabled. Check config.yaml.")
        sys.exit(1)

    logger.info(f"Loaded {len(datasets)} datasets: {list(datasets.keys())}")
    for name, ds in datasets.items():
        logger.info(f"  {name}: {len(ds.samples)} samples, {len(ds.entity_types)} entity types, {len(ds.relation_types)} relation types")

    # Step 2: Build models
    models = build_models(cfg)
    if not models:
        logger.error("No models enabled. Check config.yaml.")
        sys.exit(1)

    logger.info(f"Models to evaluate: {[m.name for m in models]}")

    # Step 3: Run evaluation (models sequentially for memory)
    results: dict[str, dict[str, MetricsResult]] = {}

    for model in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model.name}")
        logger.info(f"{'='*60}")

        try:
            model.load()
        except Exception as e:
            logger.error(f"Failed to load model {model.name}: {e}")
            continue

        for ds_name, dataset in datasets.items():
            logger.info(f"Running {model.name} on {ds_name}...")

            predictions = run_model_on_dataset(model, dataset)
            metrics = compute_metrics(
                dataset.samples,
                predictions,
                mode=cfg.general.matching_mode,
            )

            if ds_name not in results:
                results[ds_name] = {}
            results[ds_name][model.name] = metrics

            logger.info(
                f"  {model.name}/{ds_name}: "
                f"Micro-F1={metrics.micro.f1:.3f}, Macro-F1={metrics.macro.f1:.3f}, "
                f"P={metrics.micro.precision:.3f}, R={metrics.micro.recall:.3f}"
            )

            save_intermediate(model.name, ds_name, metrics, predictions, cfg.general.output_dir)

        # Free model memory
        model.unload()
        gc.collect()

    # Step 4: Generate combined report
    if results:
        generate_report(results, output_dir=cfg.general.output_dir)
    else:
        logger.error("No results collected. All models may have failed to load.")

    logger.info("Benchmarking complete.")


if __name__ == "__main__":
    main()

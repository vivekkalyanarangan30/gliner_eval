"""Configuration loading via pydantic + dotenv."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class GeneralConfig(BaseModel):
    sample_size: int = 5
    random_seed: int = 42
    output_dir: str = "outputs"
    matching_mode: str = "strict"


class DocREDConfig(BaseModel):
    enabled: bool = True
    split: str = "validation"


class CrossREConfig(BaseModel):
    enabled: bool = True
    domains: list[str] = ["ai", "news", "science"]
    split: str = "test"


class FewRelConfig(BaseModel):
    enabled: bool = True
    split: str = "val_wiki"


class DatasetsConfig(BaseModel):
    docred: DocREDConfig = DocREDConfig()
    crossre: CrossREConfig = CrossREConfig()
    fewrel: FewRelConfig = FewRelConfig()


class GLiNERConfig(BaseModel):
    enabled: bool = True
    model_id: str = "knowledgator/gliner-relex-large-v0.5"
    entity_threshold: float = 0.3
    relation_threshold: float = 0.5
    batch_size: int = 1


class GPT5Config(BaseModel):
    enabled: bool = True
    model_name: str = "gpt-5"
    temperature: float = 0.0
    max_tokens: int = 2048


class Qwen3Config(BaseModel):
    enabled: bool = True
    model_name: str = "qwen3:0.6b"
    ollama_base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.0
    max_tokens: int = 2048


class ModelsConfig(BaseModel):
    gliner: GLiNERConfig = GLiNERConfig()
    gpt5: GPT5Config = GPT5Config()
    qwen3: Qwen3Config = Qwen3Config()


class BenchmarkConfig(BaseModel):
    general: GeneralConfig = GeneralConfig()
    datasets: DatasetsConfig = DatasetsConfig()
    models: ModelsConfig = ModelsConfig()


def load_config(config_path: str = "config.yaml", env_path: Optional[str] = None) -> BenchmarkConfig:
    """Load configuration from YAML file and .env."""
    if env_path is None:
        env_path = os.path.join(os.path.dirname(config_path) or ".", ".env")
    load_dotenv(env_path)

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    return BenchmarkConfig(**raw)

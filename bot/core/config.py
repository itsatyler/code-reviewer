from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
  """Where to find (or download) the GGUF model."""
  path: str
  hf_repo: str
  hf_filename: str
  n_ctx: int = 4096
  n_gpu_layers: int = 0
  verbose: bool = False
  chat_format: Optional[str] = None


@dataclass(frozen=True)
class SamplingConfig:
  """Sampling parameters forwarded to create_chat_completion."""
  temperature: float = 0.7
  top_p: float = 0.9
  top_k: int = 40
  max_tokens: int = 1024
  repeat_penalty: float = 1.1
  stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptConfig:
  """System prompt and personality."""
  system: str = (
    "You are a Senior Code Reviewer. "
    "Point out bugs, style issues, and improvements."
  )


@dataclass(frozen=True)
class Config:
  """Top-level config loaded from config.yaml."""
  model: ModelConfig
  sampling: SamplingConfig = field(default_factory=SamplingConfig)
  prompt: PromptConfig = field(default_factory=PromptConfig)

  @classmethod
  def from_yaml(cls, path: Path | str = "config.yaml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
      raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
      raw = yaml.safe_load(f)

    model_raw = raw.get("model", {})
    sampling_raw = raw.get("sampling", {})
    prompt_raw = raw.get("prompt", {})

    # Convert stop list to tuple for frozen dataclass
    if "stop" in sampling_raw and isinstance(sampling_raw["stop"], list):
      sampling_raw["stop"] = tuple(sampling_raw["stop"])

    return cls(
      model=ModelConfig(**model_raw),
      sampling=SamplingConfig(**sampling_raw),
      prompt=PromptConfig(**prompt_raw),
    )

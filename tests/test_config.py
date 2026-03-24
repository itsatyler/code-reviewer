import pytest
from pathlib import Path

from bot.core.config import Config, ModelConfig, SamplingConfig, PromptConfig


class TestModelConfig:

  def test_frozen(self):
    mc = ModelConfig(path="test.gguf", hf_repo="r", hf_filename="f")
    with pytest.raises(Exception):
      mc.path = "other"  # type: ignore[misc]

  def test_defaults(self):
    mc = ModelConfig(path="test.gguf", hf_repo="r", hf_filename="f")
    assert mc.n_ctx == 4096
    assert mc.n_gpu_layers == 0
    assert mc.verbose is False
    assert mc.chat_format is None


class TestSamplingConfig:

  def test_defaults(self):
    sc = SamplingConfig()
    assert sc.temperature == 0.7
    assert sc.top_p == 0.9
    assert sc.top_k == 40
    assert sc.max_tokens == 1024
    assert sc.repeat_penalty == 1.1
    assert sc.stop == ()

  def test_frozen(self):
    sc = SamplingConfig()
    with pytest.raises(Exception):
      sc.temperature = 0.5  # type: ignore[misc]


class TestConfig:

  def test_from_yaml(self, tmp_path: Path):
    yaml_content = """
model:
  path: "models/test.gguf"
  hf_repo: "test/repo"
  hf_filename: "test.gguf"
  n_ctx: 2048

sampling:
  temperature: 0.5
  stop: ["</s>", "[END]"]

prompt:
  system: "Test prompt."
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = Config.from_yaml(config_file)

    assert config.model.path == "models/test.gguf"
    assert config.model.n_ctx == 2048
    assert config.sampling.temperature == 0.5
    assert config.sampling.stop == ("</s>", "[END]")
    assert config.prompt.system == "Test prompt."

  def test_from_yaml_defaults(self, tmp_path: Path):
    yaml_content = """
model:
  path: "models/test.gguf"
  hf_repo: "test/repo"
  hf_filename: "test.gguf"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = Config.from_yaml(config_file)

    assert config.sampling.temperature == 0.7
    assert "reviewer" in config.prompt.system.lower()

  def test_from_yaml_missing_file(self):
    with pytest.raises(FileNotFoundError):
      Config.from_yaml("/nonexistent/config.yaml")

  def test_frozen(self, sample_config: Config):
    with pytest.raises(Exception):
      sample_config.model = None  # type: ignore[misc]

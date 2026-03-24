from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bot.core.config import Config, ModelConfig, SamplingConfig, PromptConfig
from bot.core.pipeline import ChatPipeline, LlamaCppPipeline
from bot.core.resolver import ModelResolver
from tests.conftest import FakePipeline


class TestChatPipelineProtocol:

  def test_fake_satisfies_protocol(self):
    fake = FakePipeline()
    assert isinstance(fake, ChatPipeline)

  def test_protocol_methods(self):
    fake = FakePipeline()
    assert fake.loaded is False
    fake.load("/tmp/test.gguf")
    assert fake.loaded is True
    result = fake.chat([{"role": "user", "content": "test"}])
    assert isinstance(result, str)


class TestLlamaCppPipeline:

  def test_not_loaded_raises(self, sample_config: Config):
    pipeline = LlamaCppPipeline(sample_config)
    assert pipeline.loaded is False
    with pytest.raises(RuntimeError, match="not loaded"):
      pipeline.chat([{"role": "user", "content": "test"}])

  @patch("bot.core.pipeline.Llama")
  def test_load_and_chat(self, mock_llama_cls, sample_config: Config):
    mock_llm = MagicMock()
    mock_llama_cls.return_value = mock_llm
    mock_llm.create_chat_completion.return_value = {
      "choices": [{"message": {"content": "Hello!"}}]
    }

    pipeline = LlamaCppPipeline(sample_config)
    pipeline.load("/tmp/test.gguf")

    assert pipeline.loaded is True
    result = pipeline.chat([{"role": "user", "content": "hello"}])
    assert result == "Hello!"
    mock_llm.create_chat_completion.assert_called_once()

  @patch("bot.core.pipeline.Llama")
  def test_stream(self, mock_llama_cls, sample_config: Config):
    mock_llm = MagicMock()
    mock_llama_cls.return_value = mock_llm
    mock_llm.create_chat_completion.return_value = iter([
      {"choices": [{"delta": {"content": "Hell"}}]},
      {"choices": [{"delta": {"content": "o!"}}]},
    ])

    pipeline = LlamaCppPipeline(sample_config)
    pipeline.load("/tmp/test.gguf")
    tokens = list(pipeline.chat(
      [{"role": "user", "content": "hello"}],
      stream=True,
    ))
    assert tokens == ["Hell", "o!"]


class TestModelResolver:

  def test_local_file_exists(self, tmp_path: Path):
    model_file = tmp_path / "test.gguf"
    model_file.write_text("fake model")

    config = Config(
      model=ModelConfig(
        path=str(model_file),
        hf_repo="test/repo",
        hf_filename="test.gguf",
      ),
    )
    resolver = ModelResolver(config)
    result = resolver.resolve()
    assert result == str(model_file.resolve())

  @patch("bot.core.resolver.hf_hub_download")
  def test_download_fallback(self, mock_download, tmp_path: Path):
    model_path = tmp_path / "models" / "test.gguf"
    mock_download.return_value = str(model_path)

    config = Config(
      model=ModelConfig(
        path=str(model_path),
        hf_repo="test/repo",
        hf_filename="test.gguf",
      ),
    )
    resolver = ModelResolver(config)
    result = resolver.resolve()

    mock_download.assert_called_once_with(
      repo_id="test/repo",
      filename="test.gguf",
      local_dir=str(model_path.parent),
    )
    assert result == str(model_path)

from typing import Iterator

import pytest

from bot.core.config import Config, ModelConfig, SamplingConfig, PromptConfig


class FakePipeline:
  """In-memory fake that satisfies ChatPipeline Protocol."""

  def __init__(self) -> None:
    self._loaded = False
    self.last_messages: list[dict[str, str]] = []

  @property
  def loaded(self) -> bool:
    return self._loaded

  def load(self, model_path: str) -> None:
    self._loaded = True

  def chat(
    self,
    messages: list[dict[str, str]],
    stream: bool = False,
  ) -> str | Iterator[str]:
    self.last_messages = list(messages)
    reply = "Code review."
    if stream:
      return iter(reply.split())
    return reply


@pytest.fixture
def sample_config() -> Config:
  return Config(
    model=ModelConfig(
      path="/tmp/test-model.gguf",
      hf_repo="test/repo",
      hf_filename="test.gguf",
    ),
    sampling=SamplingConfig(),
    prompt=PromptConfig(system="Test system prompt."),
  )


@pytest.fixture
def fake_pipeline() -> FakePipeline:
  return FakePipeline()

from unittest.mock import patch
import pytest
from bot.core.bot import LocalLlama


class TestLocalLlama:

  def test_init_sets_system_prompt(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    assert len(bot._history) == 1
    assert bot._history[0]["role"] == "system"
    assert bot._history[0]["content"] == sample_config.prompt.system

  def test_review_before_boot_raises(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    with pytest.raises(RuntimeError, match="boot"):
      bot.review("print('hello')")

  def test_review(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    fake_pipeline.load("dummy")
    result = bot.review("def foo(): pass")

    assert isinstance(result, str)
    assert len(bot._history) == 3  # system + user + assistant

  def test_review_stream(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    fake_pipeline.load("dummy")
    tokens = bot.review("def foo(): pass", stream=True)

    collected = list(tokens)
    assert len(collected) > 0
    # Streaming doesn't auto-append to history
    assert len(bot._history) == 2  # system + user

    # Caller appends after collecting
    bot.append_response("".join(collected))
    assert len(bot._history) == 3

  def test_review_file(self, sample_config, fake_pipeline, tmp_path):
    test_file = tmp_path / "test.py"
    test_file.write_text("def foo():\n  return 42\n")

    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    fake_pipeline.load("dummy")
    result = bot.review_file(str(test_file))

    assert isinstance(result, str)
    assert "test.py" in fake_pipeline.last_messages[-1]["content"]

  def test_review_file_not_found(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    fake_pipeline.load("dummy")
    with pytest.raises(FileNotFoundError):
      bot.review_file("/nonexistent/file.py")

  def test_reset(self, sample_config, fake_pipeline):
    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    fake_pipeline.load("dummy")
    bot.review("first")
    bot.review("second")
    assert len(bot._history) == 5  # system + 2*(user + assistant)

    bot.reset()
    assert len(bot._history) == 1
    assert bot._history[0]["role"] == "system"

  @patch("bot.core.bot.ModelResolver")
  def test_boot(self, mock_resolver_cls, sample_config, fake_pipeline):
    mock_resolver = mock_resolver_cls.return_value
    mock_resolver.resolve.return_value = "/tmp/test.gguf"

    bot = LocalLlama(sample_config, pipeline=fake_pipeline)
    bot.boot()

    assert fake_pipeline.loaded is True
    mock_resolver.resolve.assert_called_once()

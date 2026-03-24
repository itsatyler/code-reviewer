from unittest.mock import patch, MagicMock
from bot.cli.app import CLI


class TestCLI:

  @patch("bot.cli.app.LocalLlama")
  def test_run_file_mode(self, mock_bot_cls, sample_config):
    mock_bot = mock_bot_cls.return_value
    mock_bot.review_file.return_value = iter(["Hello!", " Looks", " good!"])
    mock_bot.append_response = MagicMock()

    cli = CLI(sample_config)
    cli._bot = mock_bot
    cli._boot = MagicMock()
    cli._print_banner = MagicMock()

    cli.run(file_path="test.py")

    mock_bot.review_file.assert_called_once_with("test.py", stream=True)

  @patch("bot.cli.app.LocalLlama")
  @patch("builtins.input", side_effect=["exit"])
  def test_interactive_exit(self, mock_input, mock_bot_cls, sample_config):
    mock_bot = mock_bot_cls.return_value

    cli = CLI(sample_config)
    cli._bot = mock_bot
    cli._boot = MagicMock()
    cli._print_banner = MagicMock()

    cli.run(file_path=None)
    # Should exit cleanly without calling review
    mock_bot.review.assert_not_called()

  @patch("bot.cli.app.LocalLlama")
  @patch("builtins.input", side_effect=["reset", "exit"])
  def test_interactive_reset(self, mock_input, mock_bot_cls, sample_config):
    mock_bot = mock_bot_cls.return_value

    cli = CLI(sample_config)
    cli._bot = mock_bot
    cli._boot = MagicMock()
    cli._print_banner = MagicMock()

    cli.run(file_path=None)
    mock_bot.reset.assert_called_once()

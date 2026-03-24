from pathlib import Path
from typing import Iterator

from bot.core.config import Config
from bot.core.pipeline import ChatPipeline, LlamaCppPipeline
from bot.core.resolver import ModelResolver


class LocalLlama:
  """
  Composes model resolution, inference pipeline, and conversation
  management into a single interface. Depends on ChatPipeline Protocol,
  not on any specific backend.
  """

  def __init__(
    self,
    config: Config,
    pipeline: ChatPipeline | None = None,
  ) -> None:
    self._config = config
    self._pipeline = pipeline or LlamaCppPipeline(config)
    self._history: list[dict[str, str]] = [
      {"role": "system", "content": config.prompt.system},
    ]

  def boot(self) -> None:
    """Resolve model path (downloading if needed) and load into pipeline."""
    resolver = ModelResolver(self._config)
    model_path = resolver.resolve()
    self._pipeline.load(model_path)

  def review(self, code: str, *, stream: bool = False) -> str | Iterator[str]:
    """Submit code for review."""
    if not self._pipeline.loaded:
      raise RuntimeError("Call boot() before review().")

    self._history.append({"role": "user", "content": code})
    result = self._pipeline.chat(self._history, stream=stream)

    if isinstance(result, str):
      self._history.append({"role": "assistant", "content": result})

    return result

  def review_file(self, path: str, *, stream: bool = False) -> str | Iterator[str]:
    """Read a file and submit it for review."""
    file_path = Path(path)
    if not file_path.is_file():
      raise FileNotFoundError(f"No such file: {path}")

    code = file_path.read_text(encoding="utf-8")
    header = f"Review this file ({file_path.name}):\n\n```\n{code}\n```"
    return self.review(header, stream=stream)

  def append_response(self, content: str) -> None:
    """Append a streamed response to history after collection."""
    self._history.append({"role": "assistant", "content": content})

  def reset(self) -> None:
    """Clear conversation history, keeping the system prompt."""
    self._history = [self._history[0]]

import logging
from typing import Iterator, Protocol, runtime_checkable

from bot.core.config import Config

try:
  from llama_cpp import Llama
except ImportError:
  Llama = None  # type: ignore[assignment, misc]

log = logging.getLogger(__name__)


@runtime_checkable
class ChatPipeline(Protocol):
  """Backend-agnostic chat inference interface."""

  def load(self, model_path: str) -> None: 
    ...

  def chat(
    self,
    messages: list[dict[str, str]],
    stream: bool = False,
  ) -> str | Iterator[str]: 
    ...

  @property
  def loaded(self) -> bool: 
    ...


class LlamaCppPipeline:
  """llama-cpp-python backend for ChatPipeline."""

  def __init__(self, config: Config) -> None:
    if Llama is None:
      raise ImportError(
        "llama-cpp-python is required for this backend. "
        "Install with: uv add llama-cpp-python"
      )
    self._config = config
    self._llm: Llama | None = None # type: ignore[assignment, misc]

  @property
  def loaded(self) -> bool:
    return self._llm is not None

  def load(self, model_path: str) -> None:
    mc = self._config.model
    log.info("Loading model: %s", model_path)
    self._llm = Llama(
      model_path=model_path,
      n_ctx=mc.n_ctx,
      n_gpu_layers=mc.n_gpu_layers,
      chat_format=mc.chat_format,
      verbose=mc.verbose,
    )

  def chat(
    self,
    messages: list[dict[str, str]],
    stream: bool = False,
  ) -> str | Iterator[str]:
    if self._llm is None:
      raise RuntimeError("Pipeline not loaded. Call load() first.")

    sc = self._config.sampling
    stop = list(sc.stop) if sc.stop else None

    if stream:
      return self._stream(messages, stop)

    response = self._llm.create_chat_completion(
      messages=messages,
      temperature=sc.temperature,
      top_p=sc.top_p,
      top_k=sc.top_k,
      max_tokens=sc.max_tokens,
      repeat_penalty=sc.repeat_penalty,
      stop=stop,
    )
    return response["choices"][0]["message"]["content"] or ""

  def _stream(
    self,
    messages: list[dict[str, str]],
    stop: list[str] | None,
  ) -> Iterator[str]:
    sc = self._config.sampling
    for chunk in self._llm.create_chat_completion(  # type: ignore[union-attr]
      messages=messages,
      temperature=sc.temperature,
      top_p=sc.top_p,
      top_k=sc.top_k,
      max_tokens=sc.max_tokens,
      repeat_penalty=sc.repeat_penalty,
      stop=stop,
      stream=True,
    ):
      delta = chunk["choices"][0].get("delta", {})
      token = delta.get("content")
      if token:
        # Guard against leaked ChatML tokens (including malformed partials)
        if "<|im_" in token or "<|im_start" in token or "<|im_end" in token:
          return
        yield token

import logging
from pathlib import Path

from bot.core.config import Config

try:
  from huggingface_hub import hf_hub_download
except ImportError:
  hf_hub_download = None  # type: ignore[assignment, misc]

log = logging.getLogger(__name__)


class ModelResolver:
  """Resolves a GGUF model path: local file first, HuggingFace fallback."""

  def __init__(self, config: Config) -> None:
    self._model = config.model

  def resolve(self) -> str:
    local = Path(self._model.path).expanduser().resolve()
    if local.is_file():
      log.info("Model found locally: %s", local)
      return str(local)

    if hf_hub_download is None:
      raise ImportError(
        f"Model not found at {local} and huggingface-hub is not installed. "
        "Install with: uv add huggingface-hub"
      )

    log.info(
      "Model not at %s — downloading %s/%s",
      local, self._model.hf_repo, self._model.hf_filename,
    )

    local.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
      repo_id=self._model.hf_repo,
      filename=self._model.hf_filename,
      local_dir=str(local.parent),
    )
    log.info("Downloaded model to: %s", downloaded)
    return downloaded

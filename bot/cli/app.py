from typing import Iterator, Optional

from rich.console import Console
from rich.panel import Panel

from bot.core.config import Config
from bot.core.bot import LocalLlama


class CLI:
  """Interactive CLI — supports paste-and-review and file review modes."""

  PASTE_SENTINEL = "END"

  def __init__(self, config: Config) -> None:
    self._console = Console()
    self._bot = LocalLlama(config)

  def run(self, file_path: Optional[str] = None) -> None:
    self._print_banner()
    self._boot()

    if file_path:
      self._review_file(file_path)
    else:
      self._interactive_loop()

  def _print_banner(self) -> None:
    self._console.print(
      Panel(
        "[bold yellow]Local Llama Code Reviewer...[/]\n"
        "Just paste any code, or give it a file.",
        title="[bold red]Code Reviewer[/]",
        border_style="bright_cyan",
      )
    )

  def _boot(self) -> None:
    with self._console.status("[bold cyan]Loading the model..."):
      self._bot.boot()
    self._console.print("[green]Model loaded. Ready to review![/]\n")

  def _review_file(self, path: str) -> None:
    self._console.print(f"[cyan]Reviewing file:[/] {path}\n")
    tokens = self._bot.review_file(path, stream=True)
    self._stream_review(tokens)

  def _interactive_loop(self) -> None:
    self._console.print(
      f"[dim]Paste code below. Type '{self.PASTE_SENTINEL}' on its own line when done. "
      f"Type 'quit' or 'exit' to leave. Type 'reset' to clear history.[/]\n"
    )

    while True:
      try:
        code = self._read_multiline()
      except (KeyboardInterrupt, EOFError):
        self._console.print("\n[yellow]See Ya![/]")
        break

      cmd = code.strip().lower()
      if cmd in ("quit", "exit"):
        self._console.print("[yellow]See Ya![/]")
        break
      if cmd == "reset":
        self._bot.reset()
        self._console.print("[green]History cleared. Fresh slate![/]\n")
        continue
      if not code.strip():
        continue

      tokens = self._bot.review(code, stream=True)
      self._stream_review(tokens)

  def _read_multiline(self) -> str:
    self._console.print("[bold cyan]code>[/] ", end="")
    first = input()

    if first.strip().lower() in ("quit", "exit", "reset"):
      return first

    lines: list[str] = [first]
    while True:
      try:
        line = input()
      except EOFError:
        break
      if line.strip() == self.PASTE_SENTINEL:
        break
      lines.append(line)

    return "\n".join(lines)

  def _stream_review(self, tokens: str | Iterator[str]) -> None:
    self._console.print("\n[bold red]Reviewer:[/]")

    if isinstance(tokens, str):
      self._console.print(tokens)
      return

    collected: list[str] = []
    for token in tokens:
      self._console.print(token, end="", highlight=False)
      collected.append(token)

    full_response = "".join(collected)
    self._bot.append_response(full_response)
    self._console.print("\n")

import sys
from pathlib import Path

from bot.core.config import Config
from bot.cli.app import CLI


def main() -> None:
  config = Config.from_yaml(Path(__file__).parent / "config.yaml")
  file_arg = sys.argv[1] if len(sys.argv) > 1 else None
  cli = CLI(config)
  cli.run(file_path=file_arg)


if __name__ == "__main__":
  main()

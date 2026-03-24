# Code Reviewer

A local code reviewer powered by Mistral 7B Instruct, running entirely on your machine through llama-cpp-python. No API keys, no cloud, just a quantized model and a CLI.

## What it does

You paste code (or point it at a file), and it gives you a structured review: overall impression, specific issues, and suggestions. It streams the response token by token so you're not staring at a blank screen waiting.

There's an interactive mode where you can keep feeding it code in a loop, and it remembers the conversation context between reviews. Or you can just pass a file path and get a one-shot review.

## How it's built

The core is a `ChatPipeline` protocol that defines what an inference backend needs to look like: load a model, run chat completions, report whether it's loaded. The actual Mistral inference happens through `LlamaCppPipeline`, but the bot code never touches that directly. It just talks to the protocol. This means swapping in a different backend later is straightforward.

Model resolution is local-first. If the GGUF file is already on disk, it uses that. If not, it pulls it from HuggingFace automatically. Both `llama_cpp` and `huggingface_hub` are behind try-import guards so nothing blows up if one isn't installed yet.

The bot itself (`LocalLlama`) is the orchestrator. Construction is cheap, and model loading happens explicitly through a `boot()` call. It manages conversation history and handles both streaming and non-streaming responses.

Config lives in a single `config.yaml` file, loaded into frozen dataclasses. Model params, sampling settings, system prompt, all in one place.

The CLI is built with Rich for some nice terminal formatting. It handles the streaming output, multi-line code input (paste your code, type END), and a few simple commands like reset and quit.

## Project structure

```
bot/
  core/
    pipeline.py   -- ChatPipeline protocol + LlamaCppPipeline
    resolver.py   -- local-first model resolution, HF fallback
    bot.py        -- conversation orchestrator
    config.py     -- frozen dataclass config from YAML
  cli/
    app.py        -- Rich-powered interactive CLI
config.yaml       -- all configuration
main.py           -- entry point
```

## Running it

```bash
# install deps
uv sync

# interactive mode
uv run python main.py

# review a specific file
uv run python main.py path/to/file.py
```

The model (~4GB quantized GGUF) downloads automatically on first run if it's not already in the `models/` directory.

## Stack

- Python 3.12
- llama-cpp-python for inference
- Mistral 7B Instruct v0.2 (Q4_K_M quantization)
- PyYAML for config
- Rich for the terminal UI
- pytest for tests
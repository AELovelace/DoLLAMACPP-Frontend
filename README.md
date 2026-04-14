# DoLLAMACPP Frontend

A small Python + Qt frontend for:

- searching Hugging Face model repos
- listing GGUF files inside a selected repo
- downloading a selected GGUF model locally
- launching `llama-server` against that model
- sending a quick test chat prompt to the running server

## What this first version does

This starter app now covers the full first workflow:

1. point the app at your local `llama-server` executable
2. search Hugging Face for a model repo
3. inspect repo metadata and available `.gguf` files
4. download a selected file into the local `models/` folder
5. launch `llama-server` with that file
6. send a chat request from inside the app

It is meant as a clean foundation we can extend with presets, model management, health checks, and better server controls.

## Requirements

- Python 3.10+
- a working `llama.cpp` build or binary release that includes `llama-server`

You can get `llama.cpp` from the official repo:

- https://github.com/ggml-org/llama.cpp

The upstream README currently shows `llama-server -hf ...` support as well, but this frontend downloads a selected GGUF file locally first and then launches `llama-server` with `-m <file>`.

The `llama.cpp` README also documents its OpenAI-compatible server interface at `http://localhost:8080/v1/chat/completions`. The in-app chat panel uses that endpoint.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

## Notes

- The app stores basic settings in `frontend_config.json`.
- Downloaded models go into `models/`.
- For gated Hugging Face repos, paste a token into the `HF Token` field.
- Model downloads show live transferred bytes, total size when available, and estimated throughput.
- The repository panel shows basic metadata such as author, tags, downloads, and license when available.
- The chat panel is a lightweight tester for a running local server, not a full chat client yet.

## Good first test

Search for small GGUF repos such as:

- `gemma 3 1b gguf`
- `qwen2.5 0.5b gguf`
- `tinyllama gguf`

Then choose a smaller quantized file like `Q4_K_M` or `Q4_0` to validate the workflow quickly.

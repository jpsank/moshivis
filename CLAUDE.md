# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

MoshiVis is Kyutai's Vision-Speech Model: a 7B [Moshi](https://github.com/kyutai-labs/moshi) speech-text foundation model augmented with ~206M cross-attention adapter parameters plus a frozen 400M PaliGemma2 vision encoder. It is **inference-only** code; training/finetuning is not in this repo.

Three independent backend implementations of the same model live side by side:

- `kyuteye_pt/` — Python/PyTorch (full precision; needs ~24GB VRAM)
- `kyuteye_rs/` — Rust/Candle (used in the online demo; supports CUDA + Metal; supports q8 quantization)
- `kyuteye_mlx/` — Python/MLX (Apple Silicon; supports q4 + q8)

Plus the shared web frontend (`client/`) and synthetic visual dialogue dataset code (`ssvd/`).

When working on a backend, **edit only that one** unless the user explicitly asks for cross-backend changes; the three implementations are independent and the PyTorch + MLX backends use different config schemas than the Rust one.

## Running each backend

The web UI on `https://localhost:8088` needs SSL certs at the repo root:

```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout key.pem -out cert.pem
```

Use `--ssl False` (pt) or omit `--ssl` (mlx, http by default) to skip.

```bash
# PyTorch
cd kyuteye_pt && uv run server configs/moshika-vis.yaml --port 8088

# Rust (use --features metal on macOS)
cd kyuteye_rs && cargo run --features cuda --bin moshi-backend -r -- \
    --config configs/config-moshika-vis.json standalone --vis

# MLX (Apple Silicon)
cd kyuteye_mlx && uv run server          # bf16
cd kyuteye_mlx && uv run server -q 8     # q8 quantized
```

The web client is fetched prebuilt from HuggingFace on first run by `scripts/get_static_client.py`, or built locally with `cd client && npm install && npm run build` (produces `client/dist`).

## CI / lint / test

CI runs per-backend, all from each backend's own directory:

```bash
# kyuteye_pt
cd kyuteye_pt && uv run --locked pylint --rcfile=.pylintrc --fail-under=8.5 ./kyuteye
cd kyuteye_pt && uv run --locked sanity-check   # runs server.sanity_check (currently a no-op)

# kyuteye_mlx
cd kyuteye_mlx && uv run ruff format --diff && uv run ruff check --select I
cd kyuteye_mlx && uv run --locked sanity-check

# kyuteye_rs
cd kyuteye_rs && cargo fmt --all -- --check
cd kyuteye_rs && cargo --locked clippy --workspace --tests --examples --locked -- -D warnings
```

There is no real test suite — `kyuteye_pt/tests/hello.py` and `kyuteye_mlx/tests/test_siglip.py` are placeholders. Verify changes by running the backend and exercising the web UI.

## PyTorch backend architecture

`kyuteye_pt/kyuteye/server.py` is the entry point (`uv run server`). One WebSocket per client; everything runs under a single asyncio event loop with an `asyncio.Lock` that serializes the model — the PyTorch backend is **single-session at a time**, unlike MoshiRAG which batches.

The per-frame pipeline in `ServerState.handle_chat.opus_loop`:

1. Read PCM from the opus reader, batch into `frame_size`-sized chunks
2. `mimi.encode(chunk)` → discrete audio codes
3. For each codebook step, `moshi_vis.step(codes, ca_src=...)` → text token + audio tokens
4. `mimi.decode(audio_tokens)` → PCM out, queued for the opus writer
5. Detokenize text via SentencePiece; send each piece over the WS as a `\x07` frame

Vision is injected once per session at the start: `extract_image` reads the first WS message (kind=8), runs it through `ImageProjection` (PaliGemma2 vision encoder + projection layers), and calls `MoshiVisGen.precompte_ca_kv` to precompute the cross-attention K and V. Those tensors are then passed as `ca_src` on every subsequent step.

**Key model classes** (`kyuteye/models/`):

- `MoshiVis` (`moshivis.py`) — the model itself; subclasses `StreamingModule`. `forward_text` runs the LLM backbone (Helium) over text+audio embeddings with optional cross-attention input. `forward_depformer` runs the per-codebook depth-transformer one step at a time.
- `MoshiVisGen` — inference wrapper. `step()` is the autoregressive inner loop that handles the delayed-codebook cache; `depformer_step()` samples the audio codebooks. Always call inside `with moshi_vis.streaming(): ...`.
- `Helium` (`helium.py`) — the underlying 7B LLM backbone.
- `ImageProjection` (`image_projection.py`) — wraps the frozen vision encoder + the trained projection.

**Cross-attention** (`kyuteye/modules/cross_attention.py`) is **shared across all transformer layers** via a `SharedCrossAttention` metaclass (`xa_shared=true` in the YAML config). Each layer has a gating mechanism (`XAGate`, `xa_gating: sigmoid`) that lets the model modulate the visual signal.

The PyTorch config schema is `kyuteye_pt/configs/moshika-vis.yaml` parsed by `KyuteyeConfig.from_yml`. The Rust backend has its own JSON config in `kyuteye_rs/configs/` — schemas are not interchangeable.

### `kyuteye_pt/kyuteye/omni/` — Omni Assistant (RAG + tool calling)

Added on top of MoshiVis as an opt-in pipeline; off unless `--omni-plugin=my.module` is passed to `server`. The package adds:

- **Async retrieval** ported and simplified from `kyutai-labs/moshi-rag`. Default `LLMRetriever` calls an OpenAI-compatible endpoint (`LLM_BASE_URL`); users subclass `BaseRetriever` and register via `omni.set_retriever(...)`.
- **Tool calling** via `[TOOL: name(args)]` patterns in the model's text stream. Register functions with `@omni.tool(name=...)` or `BaseTool` subclasses via `register_tool`.
- **TextStreamMonitor** buffers detokenized text and detects RAG triggers (default substring `<ret>`) and complete `[TOOL: ...]` patterns. Incomplete tool calls stay in the buffer so half-baked patterns never fire.
- **ContextInjector** (experimental) re-encodes retrieved text into cross-attention K/V via the LLM's text-embedding table and concatenates with the image K/V.

Important caveat: MoshiVis was trained with **image patches** as the cross-attention source. Stuffing text embeddings through the same pathway is out-of-distribution — orchestration and detection all work, but the model won't reliably ground answers in the retrieved text via the XA pathway. Retrieved/tool results are always echoed back to the UI regardless, which is the reliable surface. See `kyuteye_pt/kyuteye/omni/README.md` for the full design.

## Conventions

- Use `uv run` (not `pip` / `python`) for the Python backends — both have committed `uv.lock` files and CI uses `--locked`.
- The repo refuses most refactoring PRs (see `CONTRIBUTING.md`); bug fixes are welcome. Don't restructure existing files unless asked.
- Three-backend symmetry is not enforced — adding a feature to one backend does not require porting to the others.

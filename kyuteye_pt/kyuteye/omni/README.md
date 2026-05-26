# Omni Assistant

Adds two asynchronous subsystems on top of the MoshiVis PyTorch server:

* **Asynchronous RAG**, ported from
  [kyutai-labs/moshi-rag](https://github.com/kyutai-labs/moshi-rag).
  When the model emits the configurable trigger pattern (default
  `<ret>`), a background task asks a retrieval backend for a short
  factual reference, then re-encodes the reference and concatenates it
  with the image cross-attention KV cache.
* **Pluggable tool calling**. When the model emits
  `[TOOL: name(arg=value, ...)]`, the named callable runs in a background
  task and its result is surfaced to both the UI and (optionally) the
  cross-attention input.

Both subsystems are configured by **Python plugins**, not YAML. A plugin
module imports `kyuteye.omni`, calls `set_retriever(...)` and/or
`register_tool(...)` / `@tool`, and is passed to the server via the
`--omni-plugin=your.module` CLI flag.

## Quick start

```bash
# 1. Install the optional OpenAI client (needed for the default LLM retriever).
pip install ".[omni]"

# 2. Point the default retriever at an OpenAI-compatible endpoint
#    (a local vLLM, llama.cpp server, or hosted API).
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export LLM_API_KEY=anything  # if the endpoint requires it

# 3. Launch the MoshiVis server with the bundled example plugin.
server \
  --kyuteye-config-path=configs/moshika-vis.yaml \
  --omni-plugin=kyuteye.omni.example_plugin
```

## Writing your own plugin

```python
# my_plugin.py
from kyuteye.omni import BaseRetriever, set_retriever, tool


class MyKnowledgeBase(BaseRetriever):
    async def retrieve(self, context, history, *, timeout=1.5, max_tokens=512):
        # Look up the most recent user turn in your vector store, KB, etc.
        # Return (reference_text, num_turns_consumed).
        ...


set_retriever(MyKnowledgeBase())


@tool(name="weather", description="Get the current weather")
def weather(city: str = "Paris") -> str:
    return f"It is sunny in {city}."


@tool(name="light", description="Switch a smart light")
def light(room: str = "living_room", state: str = "on") -> str:
    # Drive your smart-home hub here.
    return f"Set {room} light to {state}."
```

Then run:

```bash
server \
  --kyuteye-config-path=configs/moshika-vis.yaml \
  --omni-plugin=my_plugin
```

## Trigger patterns

* **RAG**: substring `<ret>` (override with `--omni-rag-trigger`). MoshiVis
  is not trained to emit a dedicated retrieval token, so the substring
  path is the realistic one -- prompt or fine-tune the model to say
  `<ret>` mid-turn when it doesn't know an answer.
* **Tool**: substring `[TOOL: name(args)]` (override with
  `--omni-tool-start` / `--omni-tool-end`). Arguments are parsed as
  comma-separated `key=value` pairs, with bare positional values
  supported. Quote values containing spaces.

## CLI flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--omni-plugin` | (unset) | Python dotted module to import at startup. Enables the pipeline. |
| `--omni-rag-trigger` | `<ret>` | Substring that fires retrieval. |
| `--omni-rag-timeout` | `1.5` | Seconds before retrieval gives up. |
| `--omni-rag-max-tokens` | `512` | Max tokens to request from the retrieval LLM. |
| `--omni-rag-wait-steps` | `0` | Model steps to wait before firing retrieval. |
| `--omni-xa-injection` | `True` | Re-encode retrieved text into cross-attention KV. |
| `--omni-tool-start` | `[TOOL:` | Opening marker. |
| `--omni-tool-end` | `]` | Closing marker. |

## Injection modes

`--omni-injection-mode` controls how a retrieved reference reaches the model:

* **`xa` (default, experimental)** — re-encode the reference via SentencePiece +
  Helium's text-embedding table, project through `MoshiVisGen.precompte_ca_kv`,
  and concatenate K/V with the image K/V. MoshiVis was trained with image
  patches as the cross-attention source, so attending to text via that pathway
  is out of distribution. Mechanically works, semantically weak. Set
  `--no-omni-xa-injection` to skip the concat and only surface the reference
  to the UI.
* **`streaming_sum`** — the MoshiRAG-faithful path. Requires:
  1. A model config with `rag.enabled: true` and conditioners declared (see
     "Combined MoshiVis+RAG model" below).
  2. A running ARC encoder service (HTTP `POST /embed` -> safetensors
     `[1, T, dim]`). Point at it with `--omni-arc-encoder-url=...` or the
     `REFERENCE_ENCODER_URL` env var.
  The retrieved text is forwarded to the ARC encoder, the response tensor is
  pushed into the LM's streaming-sum queue via
  `MoshiVisGen.update_streaming_sum_tensor`, and one row per step is added
  to the LM's input embeddings.
* **`off`** — the reference is surfaced to the UI as `[REF: ...]` but the
  model is not touched. Useful as a control.

Regardless of mode, the retrieved/tool text is always sent to the WebSocket
text channel so a human operator can see what the retriever produced.

## Combined MoshiVis+RAG model

The PyTorch backend now hosts MoshiRAG's full conditioner machinery
(`ConditionProvider`, `ConditionFuser`, `LUTConditioner`, `TensorConditioner`,
`learnt_padding`) under `kyuteye/conditioners/`, and `MoshiVis.forward_text`
accepts `sum_condition` / `streaming_sum_condition` / `sequence_emb` so a
combined fine-tune slots in without code changes. To enable, add a `rag:`
section to the YAML config:

```yaml
rag:
  enabled: true
  rag_token_id: 31999          # learned <ret> token id in the SP vocab
  force_streaming_sum: true    # allocate a zero slot even before any reference
  conditioners:
    first_speaker:
      type: lut
      n_bins: 2
      tokenizer: noop
      possible_values: [SPEAKER_MAIN, SPEAKER_OTHER]
      dim: 16
    reference_with_time:
      type: tensor
      dim: 4096                # ARC encoder output width
  fuse2cond:
    prepend: [first_speaker]
    streaming_sum: [reference_with_time]
```

Loading vanilla MoshiVis checkpoints against a `rag.enabled: true` config is
allowed -- the conditioner weights initialize randomly and the streaming-sum
forward path runs with random offsets. The expected workflow is to fine-tune
on combined visual + RAG data and ship that checkpoint; see kyutai-labs/moshi-rag
for the ARC encoder build and training data format.

### Features intentionally **not** ported from MoshiRAG

The merge targets a single-batch streaming inference backend. The following
MoshiRAG features were skipped on purpose; flag them if your fine-tune
requires any of them:

* **Classifier-free guidance (`cfg_coef != 1.0`)** -- MoshiRAG doubles the
  batch with positive/negative conditions and interpolates logits per step.
  MoshiVis pt has no CFG hook. A CFG-trained fine-tune will still load, but
  inference runs as `cfg_coef=1` (the conditional branch only).
* **`support_out_of_sync`** -- MoshiRAG's per-slot async exec-mask handling.
  Single-batch backend doesn't need it.
* **Depformer streaming_sum / per-codebook conditioning** -- the audio
  depformer in this backend ignores conditioning; only the main transformer
  consumes the streaming-sum row.
* **`on_text_hook`, `on_audio_hook`, `on_text_logits_hook`** -- MoshiRAG's
  per-step callbacks. The Omni `TextStreamMonitor` provides the equivalent
  observability at a higher level.
* **The `cross` fuser slot** -- vision owns cross-attention; the loader
  raises if `rag.fuse2cond` routes anything to ``cross``.
* **Local `ArcEncoder` module + `T5Conditioner`** -- the remote ARC encoder
  service covers the runtime path and avoids pulling T5 / xformers into the
  PyTorch backend.

## Layout

```
kyuteye_pt/kyuteye/omni/
├── __init__.py              # public API
├── llm_client.py            # OpenAI-compatible client (ported from moshi-rag)
├── reference_generator.py   # transcript → "Reference: ..." (ported and simplified)
├── retrievers.py            # BaseRetriever + default LLMRetriever
├── rag_manager.py           # per-channel async retrieval lifecycle
├── tools.py                 # ToolRegistry + @tool decorator + [TOOL: ...] parser
├── text_monitor.py          # token stream → events
├── context_injector.py      # text → cross-attention K/V, concat with image KV
├── example_plugin.py        # example showing the plugin shape
└── prompts/
    └── reference_prompt.txt # ported from moshi-rag (simplified template)
```

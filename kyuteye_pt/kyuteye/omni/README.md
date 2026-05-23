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

## What is "experimental" about XA injection?

MoshiVis was trained with **image patch** embeddings as the cross-attention
source. We piggyback on the same code path to inject text by tokenizing
the retrieved string with SentencePiece, looking up the LLM's text
embedding table, and projecting the result through
`MoshiVisGen.precompte_ca_kv`. The K/V tensors are concatenated with the
image K/V along the sequence dimension.

Mechanically this works -- the model attends to a longer K/V cache --
but the semantics of attending to text via the image-trained pathway are
unsurprising-to-uninspiring. Even when XA injection is off the retrieved
text is always streamed back to the user via the WebSocket text channel,
so a human operator can see the reference. Disable with
`--no-omni-xa-injection` to compare A/B.

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

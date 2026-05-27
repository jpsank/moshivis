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
* **`streaming_sum`** — the MoshiRAG-faithful path. The retrieved text
  (or tool result, see "Tool calling" below) is encoded by the ARC
  encoder into a `[1, T, dim]` tensor that is pushed into the LM's
  streaming-sum queue via `MoshiVisGen.update_streaming_sum_tensor`;
  one row per step is added to the LM's input embeddings.

  Requires a model config with `rag.enabled: true` and conditioners
  declared (see "Combined MoshiVis+RAG model" below). The ARC encoder
  itself can run in either of two modes — controlled by
  `--omni-arc-encoder-mode`:

  | Mode | Behavior |
  | --- | --- |
  | `auto` (default) | Use the in-process conditioner if one is wired into the loaded model; otherwise fall back to HTTP. |
  | `local` | Force the in-process path. Skips injection with a warning if the model has no ARC conditioner (e.g. `rag.enabled=false` or xformers missing). |
  | `http` | Force the remote service. Configure with `--omni-arc-encoder-url=...` or the `REFERENCE_ENCODER_URL` env var. |

  The in-process path calls `kyuteye.conditioners.arc_encoder.ArcEncoderConditioner`
  directly on the same GPU as the LM (no network hop, no separate
  process). The HTTP path POSTs to `{url}/embed` and decodes a
  safetensors response — protocol-compatible with the MoshiRAG ARC
  encoder service. A reference implementation of that service lives in
  this repo at `scripts/serve_arc_encoder.py` (loads the same
  conditioner the in-process path would, wraps it in an aiohttp app):

  ```bash
  python scripts/serve_arc_encoder.py \
      --kyuteye-config configs/moshika-vis.yaml \
      --host 0.0.0.0 --port 8089
  ```

  Both paths produce identical tensors when given the same input; the
  choice is operational (single-process simplicity vs. ability to scale
  encoder and LM independently).
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

### ARC encoder modes -- remote vs in-process

Reference text → conditioning tensors goes through an **ARC encoder**, the
trainable bridge MoshiRAG uses to feed reference embeddings into the LM's
`streaming_sum` slot. Two ways to host it:

* **Remote (default, recommended for serving)**. The ARC encoder runs in
  a separate HTTP service (e.g. MoshiRAG's `Dockerfile.arc_encoder`). The
  client at `kyuteye/omni/arc_encoder_client.py` POSTs the reference text
  to `{URL}/embed`, receives a safetensors-encoded `[1, T, dim]` tensor,
  and pushes it into the LM's per-slot streaming-sum queue. Activate with
  `--omni-injection-mode=streaming_sum` and `--omni-arc-encoder-url=...`
  (or the `REFERENCE_ENCODER_URL` env var). No GPU memory for the ARC
  encoder on the Moshi server; you can place it on a different machine.

* **In-process (recommended for training, or single-machine inference)**.
  The full ARC encoder is now ported into `kyuteye/conditioners/arc_encoder.py`
  -- a faithful port of `moshi-rag/moshi/moshi/conditioners/arc_encoder.py`.
  Configure it as a conditioner in the YAML:

  ```yaml
  rag:
    enabled: true
    rag_token_id: 31999
    force_streaming_sum: true
    conditioners:
      first_speaker:
        type: lut
        n_bins: 2
        tokenizer: noop
        possible_values: [SPEAKER_MAIN, SPEAKER_OTHER]
        dim: 16
      reference_with_time:
        type: arc          # was: type: tensor
        tokenizer_name: meta-llama/Llama-3.2-3B-Instruct
        embedder_params:
          compress_rates: [-4]
        bridge_module:
          in_dim: 3072
          out_dim: 4096          # Moshi LM hidden dim
          hidden_dim: 4096
        # Optional: load pretrained MoshiRAG ARC weights.
        # hf_repo: kyutai/moshika-rag-pytorch-bf16
        output_dim: 4096
    fuse2cond:
      prepend: [first_speaker]
      streaming_sum: [reference_with_time]
  ```

  Install: `pip install '.[arc]'` (adds xformers). xformers is required;
  the conditioner constructor raises a clear error if it's missing.

  At inference, the loader calls `ArcEncoderConditioner.load_weights()`
  if `hf_repo` is set; otherwise the encoder runs at random init, which
  is the **train-from-scratch** path. For training, set
  `finetune: true` on the conditioner config -- the encoder weights are
  then included in the trainable parameter set.

### Combined MoshiVis + RAG training (from scratch)

The architecture in this branch lets you train a combined fine-tune end-
to-end without copying anything back from kyutai-labs/moshi-rag:

1. **Architecture**: enable the `rag:` section above. The ARC encoder is
   in-process under the `reference_with_time` conditioner. The conditioner
   module ships with `finetune=True` support so its parameters are
   trainable. ConditionFuser routes ARC output to `streaming_sum` and the
   speaker LUT to `prepend`. The MoshiVis vision pathway stays on
   `cross_attention_src`, unchanged.

2. **Data**: use `ssvd/rag_augment.py` to generate JSONL training examples
   (visual+RAG via `augment_visual`, text-only RAG via `generate_text`).

3. **Conditioner dropout (for CFG training)**: import
   `dropout_all_conditions` from `kyuteye.conditioners` and apply it
   randomly to your sample batches during training. At inference, set
   `cfg_coef > 1.0` to amplify the conditioning signal.

4. **Training loop**: not in this repo (MoshiVis is inference-only).
   But all the pieces -- ARC encoder, fuser, dropout utilities,
   `force_streaming_sum`, CFG -- are present and trainable. A future
   training repo on top of this branch should be able to do
   `model.train()` + gradient backprop without further architectural
   work.

### Features intentionally **not** ported from MoshiRAG

The merge targets a streaming inference backend. The following MoshiRAG
features were skipped on purpose; flag them if your use case needs any:

* **`T5Conditioner`** -- text encoder using HuggingFace T5 + spacy. The
  ARC encoder is the modern path; T5 is the legacy alternative. Pull in
  if you need to load a T5-conditioned checkpoint.
* **`WhiteSpaceTokenizer` with spacy NLP processing** -- only the
  whitespace-split + hash version is ported; full lemmatization/stopword
  removal needs spacy.
* **Training-loop helpers outside conditioner dropout** -- DataLoaders,
  loss schedules, optimizer setup, etc. Inference-only repo charter.

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

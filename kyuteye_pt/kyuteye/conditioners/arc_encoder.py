"""ARC encoder text conditioner (port of kyutai-labs/moshi-rag).

The ARC encoder is the encoder MoshiRAG trains alongside its LM: it takes a
reference text string and produces a sequence of embeddings shaped
``[1, T_compressed, llm_dim]`` that the LM's ``streaming_sum`` path
additively consumes step-by-step. It is the *trainable bridge* between
"text retrieved from a knowledge base" and "tensor the LM is conditioned
on" -- a co-trained pair.

Ported faithfully from ``moshi-rag/moshi/moshi/conditioners/arc_encoder.py``
so that a future combined MoshiVis+RAG fine-tune effort doesn't have to
re-import. The major pieces:

* :class:`ArcEncoderTransformer` -- 26-layer transformer (3072-dim,
  24/8-head GQA, 8192 FFN) with a :class:`PoolingModule` for the final
  sequence compression that controls the temporal resolution of the
  conditioning stream.
* :class:`EmbProjector` -- 2-layer linear bridge from the encoder's
  ``in_dim`` to the LM's hidden ``out_dim``.
* :class:`ArcEncoderTokenizer` -- thin wrapper around a HuggingFace
  AutoTokenizer (default Llama-3.2-3B-Instruct, like upstream).
* :class:`ArcEncoderConditioner` -- the
  :class:`kyuteye.conditioners.BaseConditioner` subclass that the
  fuser registry instantiates. Plugs into ``streaming_sum`` for RAG.
* :class:`MultiArcEncoderConditioner` -- inference-time variant that
  zeros the output mask on empty references so ``learnt_padding``
  kicks in.

xformers dependency: the attention uses ``xformers.ops.fmha.memory_efficient_attention``
with ``BlockDiagonalMask`` for efficient variable-length batches. xformers
is an OPTIONAL dependency in this repo (``pip install '.[arc]'`` or
``[omni,arc]``). All xformers imports are lazy -- importing this module
without xformers installed succeeds; only constructing an
:class:`ArcEncoderConditioner` actually requires xformers, raising a
clear error with install instructions if missing.

Without retraining, the ARC encoder weights are random and the
streaming_sum injection will produce essentially noise (the encoder and
the LM are not yet co-aligned). Use the pretrained MoshiRAG ARC weights
(``kyutai/...`` on HF) **only** with a matching MoshiRAG LM checkpoint --
or train your own combined fine-tune. See
``kyuteye/omni/README.md`` for the training workflow.
"""

from __future__ import annotations

import logging
import operator
import typing as tp
from functools import reduce

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from kyuteye.conditioners.base import (
    ConditionType,
    TokenizedText,
    _BaseTextConditioner,
)
from kyuteye.conditioners.text import length_to_mask

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# xformers lazy loading.
#
# Imported at the call site (inside :meth:`Attention.forward` and
# :meth:`ArcEncoderTransformer.forward_embedder`) so that this module can be
# imported without xformers installed. The conditioner construction also
# probes for it eagerly so the failure is at config-load time, not first
# forward.
# ----------------------------------------------------------------------------


def _require_xformers() -> tuple[tp.Any, tp.Any, tp.Any]:
    """Return ``(memory_efficient_attention, BlockDiagonalMask, BlockDiagonalCausalMask)``.

    Raises ``ImportError`` with install instructions if xformers is missing.
    """
    try:
        from xformers.ops.fmha import memory_efficient_attention  # type: ignore[import-not-found]
        from xformers.ops.fmha.attn_bias import (  # type: ignore[import-not-found]
            BlockDiagonalCausalMask,
            BlockDiagonalMask,
        )
    except ImportError as e:
        raise ImportError(
            "The ArcEncoderConditioner requires xformers. Install with "
            "`pip install xformers` (or `pip install '.[arc]'` from the "
            "kyuteye_pt project root). xformers needs a matching CUDA + "
            "PyTorch build; see https://github.com/facebookresearch/xformers "
            "if pip resolution fails."
        ) from e
    return memory_efficient_attention, BlockDiagonalMask, BlockDiagonalCausalMask


# ----------------------------------------------------------------------------
# Math helpers -- verbatim from moshi-rag/conditioners/arc_encoder.py:24-87
# ----------------------------------------------------------------------------


def precompute_freqs_cis(
    dim: int, end: int, theta: float, device: tp.Optional[torch.device] = None
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_k: tp.Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis_k is None:
        freqs_cis_k = freqs_cis.clone()

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    freqs_cis_k = freqs_cis_k[:, None, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k)

    return xq_out.type_as(xq).flatten(-2), xk_out.type_as(xk).flatten(-2)


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def positions_from_sizes(sizes: tp.Iterable[int], device: tp.Any) -> torch.Tensor:
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )


def split_integer(x: int, n: int) -> list[int]:
    if n > 0:
        base = x // n
        remainder = x % n
        result = [base] * n
        for i in range(remainder):
            result[i] += 1
        return result
    n = -n
    base = x // n
    remainder = x % n
    if remainder > 0:
        result = (base + 1) * [x // (base + 1)]
        for i in range(x % (base + 1)):
            result[i] += 1
    else:
        result = [n] * base
    assert sum(result) == x, (
        f"Sum of result {sum(result)} must equal x {x} with n {n}"
    )
    return result


# ----------------------------------------------------------------------------
# Architecture -- verbatim from moshi-rag/conditioners/arc_encoder.py:90-407
# ----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class Attention(nn.Module):
    """Grouped-query attention using xformers' memory_efficient_attention.

    ``n_kv_heads`` controls the GQA ratio (``repeats = n_heads // n_kv_heads``).
    All xformers ops are lazy-imported in :meth:`forward` so module import
    doesn't require xformers.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        other_kv: tp.Optional[torch.Tensor] = None,
        freqs_cis: tp.Optional[torch.Tensor] = None,
        freqs_cis_k: tp.Optional[torch.Tensor] = None,
        mask: tp.Any = None,
    ) -> torch.Tensor:
        memory_efficient_attention, _, _ = _require_xformers()
        seqlen_sum, _ = x.shape

        if other_kv is None:
            other_kv = x.clone()
        kv_seqlen, _ = other_kv.shape

        xq = self.wq(x).view(seqlen_sum, self.n_heads, self.head_dim)
        xk = self.wk(other_kv).view(kv_seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(other_kv).view(kv_seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(
                xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k
            )

        key, val = repeat_kv(xk, xv, self.repeats, dim=1)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]

        output = memory_efficient_attention(xq, key, val, mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        other_kv: tp.Optional[torch.Tensor] = None,
        freqs_cis_k: tp.Optional[torch.Tensor] = None,
        mask: tp.Any = None,
    ) -> torch.Tensor:
        r = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            other_kv=None if other_kv is None else self.attention_norm(other_kv),
            freqs_cis_k=freqs_cis_k,
        )
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        return h + r


class PoolingModule(nn.Module):
    """Compresses an arbitrary-length sequence by mean-pooling fixed-size windows.

    Used in the last few layers of the ARC transformer to reduce the
    conditioning stream to a manageable temporal resolution. ``comp_rate``
    is interpreted as in MoshiRAG:

    * ``-1``: no compression (return inputs unchanged).
    * ``0``: collapse to a single token.
    * ``>0``: split into ``comp_rate`` chunks of (roughly) equal size.
    * ``<-1``: split into chunks of size ``abs(comp_rate)`` each.
    """

    def forward(
        self,
        x: torch.Tensor,
        comp_rate: int,
        seqlens: tp.Optional[list[int]] = None,
    ) -> tuple[torch.Tensor, list[int]]:
        new_seqlens: list[int] = []
        pool_size: list[int] = []

        if comp_rate != -1 and seqlens is not None:
            for embed_size in seqlens:
                if comp_rate == 0:
                    compressed_embed_size = [embed_size]
                elif comp_rate > 0 and embed_size // comp_rate == 0:
                    compressed_embed_size = [1] * embed_size
                elif comp_rate < -1 and embed_size // abs(comp_rate) == 0:
                    compressed_embed_size = [embed_size]
                else:
                    compressed_embed_size = split_integer(embed_size, comp_rate)
                pool_size.extend(compressed_embed_size)
                new_seqlens.append(len(compressed_embed_size))
            pool_mask = torch.block_diag(
                *[torch.ones(t) / t for t in pool_size]
            ).to(device=x.device, dtype=x.dtype)
        else:
            new_seqlens = seqlens if seqlens is not None else []
            pool_mask = None

        queries = x if pool_mask is None else pool_mask @ x
        return queries, new_seqlens


class ArcEncoderTransformer(nn.Module):
    """The ARC encoder backbone. Architecture matches MoshiRAG exactly.

    Defaults: 26 layers, 3072 dim, 24/8 GQA heads, 8192 FFN, RMSNorm with
    1e-5 eps, RoPE with theta=500000 up to 128k positions. The last
    ``len(compress_rates)`` layers run with a :class:`PoolingModule`
    cross-attending the compressed query against the pre-pooled key/value
    stream, which is how the encoder achieves variable temporal
    compression of the reference text.
    """

    def __init__(
        self,
        checkpoint: bool = False,
        compression_rate: int = -4,
    ) -> None:
        super().__init__()

        self.vocab_size = 128256
        self.n_layers = 26
        self._precomputed_freqs_cis: tp.Optional[torch.Tensor] = None

        self.tok_embeddings = torch.nn.Embedding(128256, 3072)
        self.for_embedding = True
        self.compress_rates = [compression_rate]
        self.start_compressing = self.n_layers - len(self.compress_rates)
        self.trained_layers = range(0, self.n_layers)
        self.causal = False
        self.pooling_module = PoolingModule()
        self.n_mem_tokens = 0
        self.mem_embeddings: tp.Optional[torch.Tensor] = None

        if checkpoint:
            raise ImportError(
                "torch.distributed activation checkpointing is not supported in this port"
            )

        self.layers = nn.ModuleDict(
            {
                str(i): TransformerBlock(
                    dim=3072,
                    hidden_dim=8192,
                    n_heads=24,
                    n_kv_heads=8,
                    head_dim=128,
                    norm_eps=1e-5,
                )
                for i in range(self.n_layers)
            }
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        try:
            device = next(iter(self.parameters())).device
        except StopIteration:
            device = torch.device("cuda")
        if self._precomputed_freqs_cis is None:
            self._precomputed_freqs_cis = precompute_freqs_cis(
                128, 128_000, theta=500000.0, device=device
            )
        return self._precomputed_freqs_cis

    def forward_embedder(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        _, BlockDiagonalMask, BlockDiagonalCausalMask = _require_xformers()
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        token_embeds = self.tok_embeddings(input_ids)
        h = token_embeds
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        if self.causal:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)
        freqs_cis = self.freqs_cis[positions].to(device=h.device)
        compress_index = 0

        for i in range(self.n_layers):
            if not isinstance(self_att_mask, BlockDiagonalMask):
                self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)
            if i >= self.start_compressing:
                pooled_h, new_seqlens = self.pooling_module(
                    x=h,
                    comp_rate=self.compress_rates[compress_index],
                    seqlens=seqlens,
                )
                positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                new_freqs_cis = self.freqs_cis[positions].to(device=h.device)
                if self.causal:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=seqlens
                    )
                else:
                    self_att_mask = BlockDiagonalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=seqlens
                    )
                h = self.layers[str(i)](
                    x=pooled_h,
                    other_kv=h,
                    freqs_cis=new_freqs_cis,
                    mask=self_att_mask,
                    freqs_cis_k=freqs_cis,
                )
                if self.causal:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                    )
                else:
                    self_att_mask = BlockDiagonalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                    )
                freqs_cis = new_freqs_cis
                seqlens = new_seqlens
                compress_index += 1
            else:
                h = self.layers[str(i)](
                    x=h,
                    freqs_cis=freqs_cis,
                    mask=self_att_mask,
                )

        if self.n_mem_tokens > 0:
            new_h = torch.zeros(
                (self.n_mem_tokens * len(seqlens), h.shape[1]),
                device=h.device,
                dtype=h.dtype,
            )
            ind = 0
            for j, size in enumerate(seqlens):
                new_h[j * self.n_mem_tokens : (j + 1) * self.n_mem_tokens] = h[
                    ind : ind + size
                ][-self.n_mem_tokens :]
                ind += size
            seqlens = [self.n_mem_tokens] * len(seqlens)
            h = new_h.clone()

        return h, seqlens


class EmbProjector(nn.Module):
    """2-layer linear bridge from ARC encoder hidden dim to LM hidden dim."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: tp.Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))


# ----------------------------------------------------------------------------
# Tokenizer + Conditioner -- ports from moshi-rag/conditioners/arc_encoder.py:410-606
# ----------------------------------------------------------------------------


class ArcEncoderTokenizer:
    """Wraps a HuggingFace AutoTokenizer for the ARC encoder."""

    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ArcEncoderTokenizer requires the `transformers` package."
            ) from e
        logger.info("[ARC] Loading tokenizer from %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_idx = -1
        self.bos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.stop_tokens = {self.eos_token}

    def encode(
        self,
        text: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: tp.Union[str, tp.AbstractSet[str]] = set(),
        disallowed_special: tp.Union[str, tp.Collection[str]] = (),
    ) -> list[int]:
        del allowed_special, disallowed_special  # signature parity with moshi-rag
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if bos and self.bos_token is not None:
            tokens.insert(0, self.bos_token)
        if eos and self.eos_token is not None:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class TorchAutocast:
    """Minimal autocast context manager. Ported from moshi-rag/utils/autocast.py."""

    def __init__(self, enabled: bool, *args: tp.Any, **kwargs: tp.Any) -> None:
        self.autocast = torch.autocast(*args, **kwargs) if enabled else None

    def __enter__(self) -> None:
        if self.autocast is not None:
            self.autocast.__enter__()

    def __exit__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        if self.autocast is not None:
            self.autocast.__exit__(*args, **kwargs)


class ArcEncoderConditioner(_BaseTextConditioner[TokenizedText]):
    """ARC encoder text conditioner.

    Plugs into :class:`ConditionFuser` via the ``streaming_sum`` slot:
    given a reference string, tokenize → encode → project → return
    ``[1, T_compressed, llm_dim]`` for the LM to consume one row per step.

    ``embedder_params`` and ``bridge_module`` are required kwargs; they
    parametrize the encoder transformer (``compress_rates``) and the
    bridge projector dimensions respectively. Match MoshiRAG's JSON
    config for stock weights:

    .. code-block:: yaml

        conditioners:
          reference_with_time:
            type: arc
            tokenizer_name: meta-llama/Llama-3.2-3B-Instruct
            embedder_params:
              compress_rates: [-4]
            bridge_module:
              in_dim: 3072
              out_dim: 4096    # MoshiVis LLM dim
              hidden_dim: 4096
            hf_repo: kyutai/moshika-rag-pytorch-bf16  # optional, for pretrained
            output_dim: 4096
            device: cuda
    """

    def __init__(
        self,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "bfloat16",
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        hf_repo: tp.Optional[str] = None,
        **kwargs: tp.Any,
    ) -> None:
        # Probe for xformers up-front so a misconfigured deployment surfaces
        # the error at construction time, not on the first forward.
        _require_xformers()

        self._hf_repo = hf_repo
        self.finetune = finetune

        embedder_params = kwargs.pop("embedder_params", None)
        bridge_module = kwargs.pop("bridge_module", None)
        if embedder_params is None or bridge_module is None:
            raise ValueError(
                "ArcEncoderConditioner: pass embedder_params and bridge_module"
            )
        self.config = {
            "embedder_params": embedder_params,
            "bridge_module": bridge_module,
        }
        self.compression_rate = self.config["embedder_params"]["compress_rates"][0]
        self.bridge_module_params = self.config["bridge_module"]

        super().__init__(dim=self.bridge_module_params["out_dim"], **kwargs)

        device_type = str(self.device).split(":")[0]
        if autocast_dtype is None or device_type == "cpu":
            self.autocast = TorchAutocast(enabled=False)
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            self.autocast = TorchAutocast(
                enabled=True, device_type=device_type, dtype=dtype
            )

        self.tokenizer = ArcEncoderTokenizer(tokenizer_name)
        self._init_modules()

    def _init_modules(self) -> None:
        self.embedder = ArcEncoderTransformer(
            compression_rate=self.compression_rate
        ).to(self.device)
        self.bridge_module = EmbProjector(
            in_dim=self.bridge_module_params["in_dim"],
            out_dim=self.bridge_module_params["out_dim"],
            hidden_dim=self.bridge_module_params.get("hidden_dim"),
        ).to(self.device)
        if self.finetune:
            self.embedder.train()
            self.bridge_module.train()
        else:
            self.embedder.eval()
            self.bridge_module.eval()

    def load_weights(self) -> None:
        """If ``hf_repo`` was set, load ``model.safetensors`` from HF.

        Called from :func:`kyuteye.models.loaders.get_moshi_vis` after the
        main checkpoint load. No-op if ``hf_repo`` is unset.
        """
        if not self._hf_repo:
            return
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError(
                "load_weights requires huggingface_hub and safetensors"
            ) from e
        path = hf_hub_download(self._hf_repo, "model.safetensors")
        state = load_file(path, device=str(self.device))
        self.load_state_dict(state, assign=True, strict=False)
        if self.finetune:
            self.embedder.train()
            self.bridge_module.train()
        else:
            self.embedder.eval()
            self.bridge_module.eval()

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        output: list[torch.Tensor] = []
        lengths: list[int] = []
        for text in entries:
            if text == "":
                output.append(torch.tensor([self.tokenizer.pad_idx]))
                lengths.append(0)
                continue
            tokens = self.tokenizer.encode(text, bos=False, eos=False)
            lengths.append(len(tokens))
            output.append(torch.tensor(tokens))

        mask = length_to_mask(torch.tensor(lengths))
        padded_output = pad_sequence(
            output, padding_value=self.tokenizer.pad_idx, batch_first=True
        ).int()
        return TokenizedText(
            padded_output.to(self.device), mask.to(self.device)
        )

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        batch_size, _ = tokens.shape
        assert batch_size == 1, "ArcEncoderConditioner only supports one reference for now"
        valid_tokens = tokens[0][mask[0]]

        if valid_tokens.shape[0] == 0:
            # All sequences are empty: return a single zero vector with seq
            # length 1 so the additive path has a shape to broadcast against
            # and learnt_padding (if any) can fill in.
            final_embeddings = torch.zeros(
                1, 1, self.bridge_module_params["out_dim"], device=self.device
            )
            final_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            return ConditionType(final_embeddings, final_mask)

        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeddings, embed_seqlens = self.embedder.forward_embedder(
                input_ids=valid_tokens, seqlens=[valid_tokens.shape[0]]
            )
            embeddings = self.bridge_module(embeddings).unsqueeze(0)
            masks = torch.ones(
                1, embed_seqlens[0], dtype=torch.bool, device=self.device
            )
        return ConditionType(embeddings, masks)


class MultiArcEncoderConditioner(ArcEncoderConditioner):
    """ARC encoder variant that zeros the output mask on empty references.

    Identical to :class:`ArcEncoderConditioner` except that ``_get_condition``
    forces ``mask=0`` whenever the original input mask is fully zero -- so
    :class:`BaseConditioner.forward` falls back to ``learnt_padding`` for
    empty-string references instead of feeding the encoder's response to
    the LM. Matches MoshiRAG's ``MultiArcEncoderConditioner`` at
    ``moshi-rag/moshi/moshi/conditioners/arc_encoder.py:552``.
    """

    def __init__(
        self,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "bfloat16",
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        ref_dropout: float = 0.0,
        frame_rate: float = 12.5,
        rag_time_sampling_params: tp.Optional[tp.Dict[str, float]] = None,
        **kwargs: tp.Any,
    ) -> None:
        del ref_dropout, frame_rate, rag_time_sampling_params  # training-only knobs
        super().__init__(
            finetune=finetune,
            autocast_dtype=autocast_dtype,
            tokenizer_name=tokenizer_name,
            **kwargs,
        )

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        embeddings, raw_mask = super()._get_condition(inputs)
        B = inputs.tokens.shape[0]
        assert B == 1, "MultiArcEncoderConditioner only supports one reference for now"

        if inputs.mask.sum() != 0:
            return ConditionType(embeddings, raw_mask)
        # Empty reference: zero the mask so learnt_padding kicks in.
        return ConditionType(
            torch.zeros_like(embeddings), torch.zeros_like(raw_mask)
        )
